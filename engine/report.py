"""Sonuç paylaşma ve rapor dışa aktarma (SÖZLEŞME: imzalar sabit)."""

from __future__ import annotations

import csv
import html
import json
import re
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from . import analysis, storage
from .i18n import get_language, kind_label, kind_short, method_label, tx
from .settings import TEST_KINDS


# ----------------------------------------------------------------- yardımcılar

def _unique_dest(root: Path, name: str) -> Path:
    cand = root / name
    i = 2
    while cand.exists():
        cand = root / f"{name}_{i}"
        i += 1
    return cand


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-") or tx("calisma", "run")


def _kind_file_slug(kind: str) -> str:
    return tx("buyuk", "large") if kind == "buyuk" else kind


def _find_run_dirs(base_dir: Path) -> list[Path]:
    """base_dir altında ayarlar.json içeren en üst seviye çalışma klasörlerini bulur."""
    found: list[Path] = []
    if (base_dir / "ayarlar.json").is_file():
        return [base_dir]

    def _walk(d: Path) -> None:
        for child in sorted(d.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            if (child / "ayarlar.json").is_file():
                found.append(child)
            else:
                _walk(child)

    _walk(base_dir)
    return found


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_resolved = dest_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(
                    tx(
                        f"Güvensiz zip yolu: {member.filename}",
                        f"Unsafe zip path: {member.filename}",
                    )
                )
            target = (dest_resolved / member_path).resolve()
            if target != dest_resolved and dest_resolved not in target.parents:
                raise ValueError(
                    tx(
                        f"Zip hedef dışına çıkıyor: {member.filename}",
                        f"Zip entry escapes target directory: {member.filename}",
                    )
                )
        zf.extractall(dest_resolved)


def _fmt_num(val: Any, digits: int = 3) -> str:
    if val is None or val == "":
        return "-"
    if isinstance(val, bool):
        return tx("Evet", "Yes") if val else tx("Hayır", "No")
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    return str(val)


def _fmt_pct(val: Any) -> str:
    if val is None or val == "":
        return "-"
    return tx(f"%{float(val) * 100:.1f}", f"{float(val) * 100:.1f}%")


def _write_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow(["" if v is None else v for v in row])
    return path


def _save_fig(fig: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path.name


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |")
    return "\n".join(lines)


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    ths = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    trs = []
    for row in rows:
        tds = "".join(f"<td>{html.escape(str(c))}</td>" for c in row)
        trs.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{ths}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def _run_settings_row(run: dict[str, Any], summary: dict[str, Any]) -> dict[str, str]:
    settings = run.get("settings") or {}
    train_cfg = settings.get("train_config") or {}
    env_cfg = settings.get("env") or {}
    rewards = train_cfg.get("reward_overrides") or env_cfg.get("rewards") or {}
    hparams = settings.get("hyperparameters") or train_cfg.get("hparams") or {}
    egitim = summary["egitim"]

    episodes = train_cfg.get("episodes", egitim["bolum"])
    repeats = train_cfg.get("repeats", egitim["tekrar"])
    rewards_str = ", ".join(f"{k}={v}" for k, v in rewards.items()) if rewards else "-"
    hparams_str = json.dumps(hparams, ensure_ascii=False) if hparams else "-"

    return {
        "run_id": summary["run_id"],
        "ajan": summary["agent"] or "-",
        "etiket": summary["label"] or "-",
        "bolum": str(episodes if episodes is not None else "-"),
        "tekrar": str(repeats if repeats is not None else "-"),
        "oduller": rewards_str,
        "hiperparametreler": hparams_str,
        "son_basari": _fmt_pct(egitim["son_basari"]),
        "son_odul": _fmt_num(egitim["son_ortalama_odul"], 2),
        "son_adim": _fmt_num(egitim["son_ortalama_adim"], 1),
    }


def _agent_test_summary(test_dict: dict[str, dict[str, Any]], agent_name: str) -> dict[str, Any] | None:
    if not test_dict:
        return None
    if agent_name in test_dict:
        return test_dict[agent_name]
    for method, stats in test_dict.items():
        if method not in ("random", "sag_el"):
            return stats
    return next(iter(test_dict.values()), None)


# ----------------------------------------------------------------- genel API

def export_run_zip(run_dir: Path, out_path: Path) -> Path:
    """Çalışma klasörünü arkadaşa göndermek için zip'ler."""
    run_dir = Path(run_dir)
    out_path = Path(out_path)
    if not run_dir.is_dir():
        raise ValueError(
            tx(
                f"Çalışma klasörü bulunamadı: {run_dir}",
                f"Run directory not found: {run_dir}",
            )
        )

    settings_path = run_dir / "ayarlar.json"
    if settings_path.is_file():
        settings = storage.read_settings(run_dir)
        run_id = str(settings.get("run_id") or run_dir.name)
    else:
        run_id = run_dir.name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        files_written = 0
        for p in sorted(run_dir.rglob("*")):
            if p.is_file() and p.suffix != ".tmp":
                rel = p.relative_to(run_dir).as_posix()
                zf.write(p, arcname=f"{run_id}/{rel}")
                files_written += 1
        if files_written == 0:
            zf.writestr(f"{run_id}/", "")
    return out_path


def import_runs(sources: Sequence[Path], root: Path | None = None) -> list[Path]:
    """Klasör ya da .zip olarak gelen çalışmaları results/ altına kopyalar."""
    dest_root = Path(root) if root is not None else storage.RESULTS_ROOT
    dest_root.mkdir(parents=True, exist_ok=True)

    imported: list[Path] = []
    for src in sources:
        src = Path(src)
        if not src.exists():
            raise ValueError(tx(f"Kaynak bulunamadı: {src}", f"Source not found: {src}"))

        if src.is_dir():
            run_dirs = _find_run_dirs(src)
            if not run_dirs:
                raise ValueError(
                    tx(
                        f"Klasörde ayarlar.json içeren çalışma bulunamadı: {src}",
                        f"No run containing ayarlar.json found in directory: {src}",
                    )
                )
            for rdir in run_dirs:
                dest = _unique_dest(dest_root, rdir.name)
                shutil.copytree(rdir, dest)
                imported.append(dest)
        elif src.is_file() and zipfile.is_zipfile(src):
            with tempfile.TemporaryDirectory() as tmp_str:
                tmp_dir = Path(tmp_str)
                _safe_extract_zip(src, tmp_dir)
                run_dirs = _find_run_dirs(tmp_dir)
                if not run_dirs:
                    raise ValueError(
                        tx(
                            f"Zip içinde ayarlar.json içeren çalışma bulunamadı: {src}",
                            f"No run containing ayarlar.json found in zip: {src}",
                        )
                    )
                for rdir in run_dirs:
                    if rdir == tmp_dir:
                        settings = storage.read_settings(rdir)
                        folder_name = str(settings.get("run_id") or src.stem)
                    else:
                        folder_name = rdir.name
                    dest = _unique_dest(dest_root, folder_name)
                    shutil.copytree(rdir, dest)
                    imported.append(dest)
        else:
            raise ValueError(
                tx(
                    f"Geçersiz kaynak (klasör veya .zip olmalı): {src}",
                    f"Invalid source (must be a directory or .zip): {src}",
                )
            )

    return imported


def export_report(run_dirs: Sequence[Path], out_dir: Path) -> Path:
    """Seçilen çalışmalardan out_dir içinde zaman damgalı rapor klasörü üretir; HTML yolunu döndürür."""
    if not run_dirs:
        raise ValueError(
            tx(
                "Rapor için en az bir çalışma klasörü gerekli.",
                "At least one run directory is required for a report.",
            )
        )

    loaded_runs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for rd in run_dirs:
        rd_path = Path(rd)
        if not rd_path.is_dir():
            raise ValueError(
                tx(
                    f"Çalışma klasörü bulunamadı: {rd_path}",
                    f"Run directory not found: {rd_path}",
                )
            )
        run = analysis.load_run(rd_path)
        loaded_runs.append(run)
        summaries.append(analysis.summarize_loaded(run))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_dir = _unique_dest(out_dir, f"{tx('rapor', 'report')}_{stamp}")
    figs_dirname = tx("grafikler", "figures")
    tables_dirname = tx("tablolar", "tables")
    figs_dir = report_dir / figs_dirname
    tables_dir = report_dir / tables_dirname
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Tekil ve çoklu çalışma dosya önekleri
    used_slugs: dict[str, int] = {}
    run_prefixes: list[str] = []
    multi = len(loaded_runs) > 1
    for summary in summaries:
        base = _slug(summary["run_id"])
        cnt = used_slugs.get(base, 0) + 1
        used_slugs[base] = cnt
        slug = base if cnt == 1 else f"{base}_{cnt}"
        run_prefixes.append(f"{slug}_" if multi else "")

    # 1. Ayarlar tablosu
    ayar_headers = [
        tx("Çalışma", "Run"),
        tx("Ajan", "Agent"),
        tx("Etiket", "Label"),
        tx("Bölüm", "Episodes"),
        tx("Tekrar", "Repeats"),
        tx("Ödüller", "Rewards"),
        tx("Hiperparametreler", "Hyperparameters"),
        tx("Son %10 Başarı", "Last 10% Success"),
        tx("Son %10 Ödül", "Last 10% Reward"),
        tx("Son %10 Adım", "Last 10% Steps"),
    ]
    ayar_rows: list[list[str]] = []
    for run, summary in zip(loaded_runs, summaries):
        info = _run_settings_row(run, summary)
        ayar_rows.append([
            info["run_id"], info["ajan"], info["etiket"], info["bolum"], info["tekrar"],
            info["oduller"], info["hiperparametreler"], info["son_basari"], info["son_odul"], info["son_adim"],
        ])
    _write_csv(tables_dir / tx("ayarlar.csv", "settings.csv"), ayar_headers, ayar_rows)

    # 2. Öğrenme eğrileri (eğitim verisi olan çalışmalar)
    runs_with_train = [r for r in loaded_runs if r.get("train")]
    learning_imgs: list[tuple[str, str]] = []
    if runs_with_train:
        for metric, fname, title in (
            (
                "success",
                tx("ogrenme_basari.png", "learning_success.png"),
                tx("Öğrenme Eğrisi — Başarı Oranı", "Learning Curve — Success Rate"),
            ),
            (
                "total_reward",
                tx("ogrenme_odul.png", "learning_reward.png"),
                tx("Öğrenme Eğrisi — Toplam Ödül", "Learning Curve — Total Reward"),
            ),
            (
                "steps",
                tx("ogrenme_adim.png", "learning_steps.png"),
                tx("Öğrenme Eğrisi — Adım Sayısı", "Learning Curve — Steps"),
            ),
        ):
            fig = analysis.learning_curve(runs_with_train, metric=metric)
            saved_name = _save_fig(fig, figs_dir / fname)
            learning_imgs.append((title, f"{figs_dirname}/{saved_name}"))

    # 3. Çalışma bazlı test tabloları & grafikleri + genelleme
    per_run_sections: list[dict[str, Any]] = []
    test_headers = [
        tx("Yöntem", "Method"),
        tx("Başarı", "Success"),
        tx("Başarı Std", "Success Std"),
        tx("Ort. Adım (Başarılı)", "Avg. Steps (Successful)"),
        tx("Verimlilik", "Efficiency"),
        tx("Ort. Ödül", "Avg. Reward"),
        tx("Ort. Çarpma", "Avg. Wall Hits"),
        tx("Bölüm", "Episodes"),
    ]

    for run, summary, prefix in zip(loaded_runs, summaries, run_prefixes):
        section: dict[str, Any] = {
            "summary": summary,
            "settings_row": _run_settings_row(run, summary),
            "tests": {},
            "gen_img": None,
        }
        for kind in TEST_KINDS:
            t_dict = summary["test"].get(kind) or {}
            if not t_dict:
                continue
            csv_rows = []
            display_rows = []
            for method, st in t_dict.items():
                mlabel = method_label(method)
                csv_rows.append([
                    mlabel, st["basari"], st["basari_std"], st["adim"],
                    st["verim"], st["odul"], st["carpma"], st["n"],
                ])
                display_rows.append([
                    mlabel,
                    _fmt_pct(st["basari"]),
                    _fmt_num(st["basari_std"], 3),
                    _fmt_num(st["adim"], 1),
                    _fmt_num(st["verim"], 2),
                    _fmt_num(st["odul"], 2),
                    _fmt_num(st["carpma"], 1),
                    str(st["n"]),
                ])
            kind_slug = _kind_file_slug(kind)
            _write_csv(tables_dir / f"{prefix}test_{kind_slug}.csv", test_headers, csv_rows)
            img_name = _save_fig(
                analysis.baseline_bars(run, kind=kind),
                figs_dir / f"{prefix}test_{kind_slug}.png",
            )
            section["tests"][kind] = {
                "rows": display_rows,
                "img": f"{figs_dirname}/{img_name}",
            }

        if "normal" in section["tests"] and "buyuk" in section["tests"]:
            gen_fname = f"{prefix}{tx('genelleme', 'generalization')}.png"
            gen_name = _save_fig(analysis.generalization_plot(run), figs_dir / gen_fname)
            section["gen_img"] = f"{figs_dirname}/{gen_name}"

        per_run_sections.append(section)

    # 4. Çoklu çalışma karşılaştırması
    comp_rows: list[list[str]] = []
    comp_imgs: list[tuple[str, str]] = []
    comp_headers = [
        tx("Çalışma", "Run"),
        tx("Ajan", "Agent"),
        tx("Etiket", "Label"),
        tx("Son %10 Başarı", "Last 10% Success"),
        tx("Normal Başarı", "Normal Success"),
        tx("Normal Verimlilik", "Normal Efficiency"),
        tx("Büyük Başarı", "Large Success"),
        tx("Büyük Verimlilik", "Large Efficiency"),
    ]
    if multi:
        for summary in summaries:
            st_n = _agent_test_summary(summary["test"].get("normal") or {}, summary["agent"])
            st_b = _agent_test_summary(summary["test"].get("buyuk") or {}, summary["agent"])
            comp_rows.append([
                summary["run_id"],
                summary["agent"] or "-",
                summary["label"] or "-",
                _fmt_pct(summary["egitim"]["son_basari"]),
                _fmt_pct(st_n["basari"]) if st_n else "-",
                _fmt_num(st_n["verim"], 2) if st_n else "-",
                _fmt_pct(st_b["basari"]) if st_b else "-",
                _fmt_num(st_b["verim"], 2) if st_b else "-",
            ])
        _write_csv(tables_dir / tx("karsilastirma.csv", "comparison.csv"), comp_headers, comp_rows)

        for kind, title in (
            ("normal", tx("Karşılaştırma — Normal Labirent", "Comparison — Normal Maze")),
            ("buyuk", tx("Karşılaştırma — Büyük Labirent", "Comparison — Large Maze")),
        ):
            runs_k = [r for r in loaded_runs if (r.get("test") or {}).get(kind)]
            if runs_k:
                kind_slug = _kind_file_slug(kind)
                comp_fname = tx(f"karsilastirma_{kind_slug}.png", f"comparison_{kind_slug}.png")
                cname = _save_fig(analysis.compare_runs(runs_k, kind=kind), figs_dir / comp_fname)
                comp_imgs.append((title, f"{figs_dirname}/{cname}"))

    # 5. Markdown rapor
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    report_title = tx("Pekiştirmeli Öğrenme Deney Raporu", "Reinforcement Learning Experiment Report")
    settings_heading = tx("Çalışma Ayarları ve Eğitim Özeti", "Run Settings and Training Summary")
    learning_heading = tx("Öğrenme Eğrileri", "Learning Curves")
    run_heading_prefix = tx("Çalışma", "Run")
    no_test_msg = tx(
        "Bu çalışma için test sonucu bulunmuyor.",
        "No test results available for this run.",
    )
    gen_heading = f"{tx('Genelleme', 'Generalization')} ({kind_short('normal')} vs {kind_short('buyuk')})"
    gen_alt = tx("Genelleme", "Generalization")
    comp_heading = tx("Çalışmalar Arası Karşılaştırma", "Cross-Run Comparison")

    md_parts: list[str] = [
        f"# {report_title}",
        tx(f"_Oluşturulma: {now_str}_", f"_Generated: {now_str}_"),
        "",
        f"## {settings_heading}",
        _md_table(ayar_headers, ayar_rows),
        "",
    ]
    if learning_imgs:
        md_parts.append(f"## {learning_heading}")
        for title, rel in learning_imgs:
            md_parts.append(f"### {title}")
            md_parts.append(f"![{title}]({rel})")
            md_parts.append("")

    for sec in per_run_sections:
        s = sec["summary"]
        title_label = f"{s['agent']} {s['label']}".strip() or s["run_id"]
        md_parts.append(f"## {run_heading_prefix}: {title_label} (`{s['run_id']}`)")
        if not sec["tests"]:
            md_parts.append(f"_{no_test_msg}_")
            md_parts.append("")
            continue
        for kind in TEST_KINDS:
            if kind not in sec["tests"]:
                continue
            tdata = sec["tests"][kind]
            kind_title = tx(f"{kind_label(kind)} Testi", f"{kind_label(kind)} Test")
            md_parts.append(f"### {kind_title}")
            md_parts.append(_md_table(test_headers, tdata["rows"]))
            md_parts.append("")
            md_parts.append(f"![{kind_title}]({tdata['img']})")
            md_parts.append("")
        if sec["gen_img"]:
            md_parts.append(f"### {gen_heading}")
            md_parts.append(f"![{gen_alt}]({sec['gen_img']})")
            md_parts.append("")

    if multi:
        md_parts.append(f"## {comp_heading}")
        md_parts.append(_md_table(comp_headers, comp_rows))
        md_parts.append("")
        for title, rel in comp_imgs:
            md_parts.append(f"### {title}")
            md_parts.append(f"![{title}]({rel})")
            md_parts.append("")

    (report_dir / tx("rapor.md", "report.md")).write_text("\n".join(md_parts), encoding="utf-8")

    # 6. HTML rapor (tek dosya, görseller grafikler/ veya figures/ göreli)
    html_sections: list[str] = [
        f"<h1>{html.escape(report_title)}</h1>",
        tx(
            f"<p class='meta'>Oluşturulma: {html.escape(now_str)} · Çalışma sayısı: {len(loaded_runs)}</p>",
            f"<p class='meta'>Generated: {html.escape(now_str)} · Runs: {len(loaded_runs)}</p>",
        ),
        f"<h2>{html.escape(settings_heading)}</h2>",
        _html_table(ayar_headers, ayar_rows),
    ]
    if learning_imgs:
        html_sections.append(f"<h2>{html.escape(learning_heading)}</h2>")
        for title, rel in learning_imgs:
            html_sections.append(
                f"<div class='card'><h3>{html.escape(title)}</h3>"
                f"<img src='{html.escape(rel)}' alt='{html.escape(title)}'></div>"
            )

    for sec in per_run_sections:
        s = sec["summary"]
        title_label = f"{s['agent']} {s['label']}".strip() or s["run_id"]
        html_sections.append(
            f"<h2>{html.escape(run_heading_prefix)}: {html.escape(title_label)} "
            f"<span class='runid'>({html.escape(s['run_id'])})</span></h2>"
        )
        if not sec["tests"]:
            html_sections.append(f"<p class='empty'>{html.escape(no_test_msg)}</p>")
            continue
        for kind in TEST_KINDS:
            if kind not in sec["tests"]:
                continue
            tdata = sec["tests"][kind]
            kind_title = tx(f"{kind_label(kind)} Testi", f"{kind_label(kind)} Test")
            html_sections.append(f"<h3>{html.escape(kind_title)}</h3>")
            html_sections.append(_html_table(test_headers, tdata["rows"]))
            html_sections.append(
                f"<div class='card'><img src='{html.escape(tdata['img'])}' "
                f"alt='{html.escape(kind_title)}'></div>"
            )
        if sec["gen_img"]:
            html_sections.append(f"<h3>{html.escape(gen_heading)}</h3>")
            html_sections.append(
                f"<div class='card'><img src='{html.escape(sec['gen_img'])}' "
                f"alt='{html.escape(gen_alt)}'></div>"
            )

    if multi:
        html_sections.append(f"<h2>{html.escape(comp_heading)}</h2>")
        html_sections.append(_html_table(comp_headers, comp_rows))
        for title, rel in comp_imgs:
            html_sections.append(
                f"<div class='card'><h3>{html.escape(title)}</h3>"
                f"<img src='{html.escape(rel)}' alt='{html.escape(title)}'></div>"
            )

    lang = get_language()
    html_doc = f"""<!DOCTYPE html>
<html lang="{html.escape(lang)}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(report_title)}</title>
<style>
  body {{ font-family: "Segoe UI", Arial, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1.2rem; color: #1f2328; background: #fafbfc; line-height: 1.5; }}
  h1 {{ border-bottom: 2px solid #d0d7de; padding-bottom: 0.4rem; }}
  h2 {{ margin-top: 2rem; border-bottom: 1px solid #e5e8eb; padding-bottom: 0.25rem; }}
  h3 {{ margin-top: 1.3rem; color: #2b3137; }}
  .meta, .empty {{ color: #57606a; font-size: 0.95rem; }}
  .runid {{ color: #57606a; font-weight: normal; font-size: 0.9rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 0.8rem 0 1.2rem; background: #fff; font-size: 0.9rem; }}
  th, td {{ border: 1px solid #d0d7de; padding: 0.45rem 0.65rem; text-align: left; }}
  th {{ background: #f3f4f6; font-weight: 600; }}
  tr:nth-child(even) td {{ background: #f9fafb; }}
  .card {{ background: #fff; border: 1px solid #d0d7de; border-radius: 6px; padding: 0.8rem; margin: 0.8rem 0 1.4rem; text-align: center; }}
  img {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
{"".join(html_sections)}
</body>
</html>
"""
    html_path = report_dir / tx("rapor.html", "report.html")
    html_path.write_text(html_doc, encoding="utf-8")
    return html_path
