"""Sonuç analizi ve grafikler (SÖZLEŞME: imzalar sabit).

Grafik fonksiyonları matplotlib `Figure` döndürür (pyplot kullanmadan,
`matplotlib.figure.Figure` ile); arayüz bunları gömebilir, kaydetme ayrı.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from . import storage
from .i18n import kind_label, kind_short, method_label, tx
from .settings import TEST_KINDS

BASELINE_COLORS: dict[str, str] = {
    "random": "#7f7f7f",
    "sag_el": "#ff7f0e",
}

AGENT_PALETTE: tuple[str, ...] = (
    "#1f77b4",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
)

_METRIC_META: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    "success": (
        ("Başarı Oranı", "Success Rate"),
        ("Öğrenme Eğrisi — Başarı Oranı", "Learning Curve — Success Rate"),
    ),
    "total_reward": (
        ("Toplam Ödül", "Total Reward"),
        ("Öğrenme Eğrisi — Toplam Ödül", "Learning Curve — Total Reward"),
    ),
    "steps": (
        ("Adım Sayısı", "Steps"),
        ("Öğrenme Eğrisi — Adım Sayısı", "Learning Curve — Steps"),
    ),
}


# ----------------------------------------------------------------- yardımcılar

def _run_label(run: dict[str, Any]) -> str:
    settings = run.get("settings") or {}
    agent_info = settings.get("agent") or {}
    train_cfg = settings.get("train_config") or {}
    agent = agent_info.get("name") or train_cfg.get("agent_name") or Path(run["dir"]).name
    label = str(train_cfg.get("label") or "").strip()
    return f"{agent} {label}".strip()


def _method_color(method: str, agent_idx: int = 0) -> str:
    if method in BASELINE_COLORS:
        return BASELINE_COLORS[method]
    return AGENT_PALETTE[agent_idx % len(AGENT_PALETTE)]


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0 or window <= 1:
        return values.astype(float, copy=True)
    cs = np.cumsum(values, dtype=float)
    out = np.empty_like(cs)
    head = min(window, len(values))
    out[:head] = cs[:head] / np.arange(1, head + 1, dtype=float)
    if len(values) > window:
        out[window:] = (cs[window:] - cs[:-window]) / float(window)
    return out


def _group_train_by_repeat(rows: Sequence[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_repeat: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        rep = int(row.get("repeat", 0))
        by_repeat.setdefault(rep, []).append(row)
    for rep in by_repeat:
        by_repeat[rep].sort(key=lambda r: int(r.get("episode", 0)))
    return dict(sorted(by_repeat.items()))


def _group_test_by_method(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    is_base: dict[str, bool] = {}
    order: list[str] = []
    for row in rows:
        method = str(row["method"])
        if method not in grouped:
            grouped[method] = []
            is_base[method] = bool(row.get("is_baseline"))
            order.append(method)
        grouped[method].append(row)
    ordered = sorted(order, key=lambda m: (1 if is_base[m] else 0, order.index(m)))
    return {m: grouped[m] for m in ordered}


def _compute_method_stats(mrows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_rep: dict[int, list[dict[str, Any]]] = {}
    for r in mrows:
        rep = int(r.get("repeat", 0))
        by_rep.setdefault(rep, []).append(r)

    rep_srs = [
        float(np.mean([1.0 if r.get("success") else 0.0 for r in rrows]))
        for _, rrows in sorted(by_rep.items())
    ]
    basari = float(np.mean([1.0 if r.get("success") else 0.0 for r in mrows]))
    basari_std = float(np.std(rep_srs, ddof=0)) if len(rep_srs) > 1 else 0.0

    succ_rows = [r for r in mrows if r.get("success")]
    adim = float(np.mean([float(r["steps"]) for r in succ_rows])) if succ_rows else None
    eff_vals = [float(r["efficiency"]) for r in succ_rows if r.get("efficiency") is not None]
    verim = float(np.mean(eff_vals)) if eff_vals else None

    odul = float(np.mean([float(r["total_reward"]) for r in mrows]))
    carpma = float(np.mean([float(r["wall_hits"]) for r in mrows]))
    return {
        "basari": basari,
        "basari_std": basari_std,
        "adim": adim,
        "verim": verim,
        "odul": odul,
        "carpma": carpma,
        "n": len(mrows),
    }


def _style_axis(ax: Any) -> None:
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _empty_axis(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#666666")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig: Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def summarize_loaded(run: dict[str, Any]) -> dict[str, Any]:
    """Yüklenmiş çalışma sözlüğünden özet sözlüğü hesaplar (dosyaya yazmaz)."""
    settings = run.get("settings") or {}
    agent_info = settings.get("agent") or {}
    train_cfg = settings.get("train_config") or {}
    run_id = settings.get("run_id") or Path(run["dir"]).name
    agent = agent_info.get("name") or train_cfg.get("agent_name") or ""
    label = str(train_cfg.get("label") or "")

    by_repeat = _group_train_by_repeat(run.get("train") or [])
    tekrar = len(by_repeat)
    bolum = max((len(v) for v in by_repeat.values()), default=0)

    if tekrar > 0:
        rep_succ: list[float] = []
        rep_rew: list[float] = []
        rep_steps: list[float] = []
        for rows in by_repeat.values():
            tail_n = max(1, len(rows) // 10)
            tail = rows[-tail_n:]
            rep_succ.append(float(np.mean([1.0 if r.get("success") else 0.0 for r in tail])))
            rep_rew.append(float(np.mean([float(r["total_reward"]) for r in tail])))
            rep_steps.append(float(np.mean([float(r["steps"]) for r in tail])))
        egitim_summary = {
            "bolum": bolum,
            "tekrar": tekrar,
            "son_basari": float(np.mean(rep_succ)),
            "son_ortalama_odul": float(np.mean(rep_rew)),
            "son_ortalama_adim": float(np.mean(rep_steps)),
        }
    else:
        egitim_summary = {
            "bolum": 0,
            "tekrar": 0,
            "son_basari": None,
            "son_ortalama_odul": None,
            "son_ortalama_adim": None,
        }

    test_summary: dict[str, dict[str, dict[str, Any]]] = {}
    for kind in TEST_KINDS:
        trows = (run.get("test") or {}).get(kind) or []
        if not trows:
            test_summary[kind] = {}
            continue
        grouped = _group_test_by_method(trows)
        test_summary[kind] = {
            method: _compute_method_stats(mrows) for method, mrows in grouped.items()
        }

    return {
        "run_id": run_id,
        "agent": agent,
        "label": label,
        "egitim": egitim_summary,
        "test": test_summary,
    }


# ----------------------------------------------------------------- genel API

def load_run(run_dir: Path) -> dict[str, Any]:
    """{"dir", "settings", "train": [satır...], "test": {kind: [satır...]}} (olmayan dosya -> boş)."""
    run_dir = Path(run_dir)
    settings_path = run_dir / "ayarlar.json"
    settings = storage.read_settings(run_dir) if settings_path.is_file() else {}

    train_path = storage.train_log_path(run_dir)
    train = storage.read_csv(train_path) if train_path.is_file() else []

    test: dict[str, list[dict[str, Any]]] = {}
    for kind in TEST_KINDS:
        tpath = storage.test_csv_path(run_dir, kind)
        test[kind] = storage.read_csv(tpath) if tpath.is_file() else []

    return {
        "dir": run_dir,
        "settings": settings,
        "train": train,
        "test": test,
    }


def summarize(run_dir: Path) -> dict[str, Any]:
    """ozet.json üretir ve döndürür. Ayrıntı: engine/CONTRACT.md "analysis"."""
    run_dir = Path(run_dir)
    summary = summarize_loaded(load_run(run_dir))
    with open(run_dir / "ozet.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def learning_curve(
    runs: Sequence[dict[str, Any]],
    metric: str = "success",
    window: int | None = None,
) -> Figure:
    """metric: "success" | "total_reward" | "steps". Her çalışma bir eğri: tekrarların
    ortalaması (hareketli ortalama) + gölgeli std."""
    if metric not in _METRIC_META:
        raise ValueError(
            tx(
                f"Bilinmeyen metrik: {metric}. Beklenen: {tuple(_METRIC_META)}",
                f"Unknown metric: {metric}. Expected: {tuple(_METRIC_META)}",
            )
        )

    ylabel_pair, title_pair = _METRIC_META[metric]
    ylabel, title = tx(*ylabel_pair), tx(*title_pair)
    fig = Figure(figsize=(8.5, 4.8), dpi=100)
    ax = fig.add_subplot(111)

    plotted = False
    for idx, run in enumerate(runs):
        train_rows = run.get("train") or []
        if not train_rows:
            continue
        by_repeat = _group_train_by_repeat(train_rows)
        max_len = max(len(v) for v in by_repeat.values())
        if max_len == 0:
            continue

        w = max(1, max_len // 50) if window is None else max(1, int(window))
        mat = np.full((len(by_repeat), max_len), np.nan, dtype=float)

        for r_idx, rows in enumerate(by_repeat.values()):
            if metric == "success":
                vals = np.array([1.0 if r.get("success") else 0.0 for r in rows], dtype=float)
            else:
                vals = np.array([float(r[metric]) for r in rows], dtype=float)
            smoothed = _moving_average(vals, w)
            mat[r_idx, : len(smoothed)] = smoothed

        x = np.arange(1, max_len + 1, dtype=float)
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)

        color = AGENT_PALETTE[idx % len(AGENT_PALETTE)]
        label = _run_label(run)
        ax.plot(x, mean, label=label, color=color, linewidth=2.0)

        low = mean - std
        high = mean + std
        if metric == "success":
            low = np.clip(low, 0.0, 1.0)
            high = np.clip(high, 0.0, 1.0)
        ax.fill_between(x, low, high, color=color, alpha=0.18)
        plotted = True

    if plotted:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(tx("Bölüm", "Episode"))
        ax.set_ylabel(ylabel)
        if metric == "success":
            ax.set_ylim(-0.02, 1.05)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
    else:
        ax.set_title(title, fontsize=12, fontweight="bold")
        _empty_axis(ax, tx("Eğitim verisi bulunamadı", "No training data found"))

    fig.tight_layout()
    return fig


def baseline_bars(run: dict[str, Any], kind: str = "normal") -> Figure:
    """Ajan ve baseline'lar: başarı oranı, ort. adım (başarılı), ort. çarpma."""
    fig = Figure(figsize=(12, 4.2), dpi=100)
    axes = fig.subplots(1, 3)
    kind_lbl = kind_label(kind)
    fig.suptitle(
        tx(
            f"{_run_label(run)} — Test Sonuçları ({kind_lbl})",
            f"{_run_label(run)} — Test Results ({kind_lbl})",
        ),
        fontsize=12,
        fontweight="bold",
    )

    trows = (run.get("test") or {}).get(kind) or []
    if not trows:
        for ax in axes:
            _empty_axis(ax, tx("Test verisi yok", "No test data"))
        fig.tight_layout()
        return fig

    grouped = _group_test_by_method(trows)
    methods = list(grouped.keys())
    method_labels = [method_label(m) for m in methods]
    stats = {m: _compute_method_stats(rows) for m, rows in grouped.items()}
    colors = [_method_color(m, agent_idx=0) for m in methods]
    x = np.arange(len(methods))

    # 1. Başarı Oranı
    ax0 = axes[0]
    succ_vals = [stats[m]["basari"] for m in methods]
    succ_stds = [stats[m]["basari_std"] for m in methods]
    bars0 = ax0.bar(x, succ_vals, yerr=succ_stds if any(s > 0 for s in succ_stds) else None,
                    capsize=4, color=colors, width=0.6)
    ax0.set_title(tx("Başarı Oranı", "Success Rate"))
    ax0.set_ylabel(tx("Oran", "Rate"))
    ax0.set_ylim(0.0, 1.12)
    ax0.set_xticks(x)
    ax0.set_xticklabels(method_labels, rotation=15, ha="right")
    _style_axis(ax0)
    for bar, val in zip(bars0, succ_vals):
        pct_txt = tx(f"%{val * 100:.1f}", f"{val * 100:.1f}%")
        ax0.text(bar.get_x() + bar.get_width() / 2, val + 0.02, pct_txt,
                 ha="center", va="bottom", fontsize=9)

    # 2. Ortalama Adım (yalnız başarılı)
    ax1 = axes[1]
    step_raw = [stats[m]["adim"] for m in methods]
    step_vals = [v if v is not None else 0.0 for v in step_raw]
    bars1 = ax1.bar(x, step_vals, color=colors, width=0.6)
    ax1.set_title(tx("Ort. Adım (Başarılı)", "Avg. Steps (Successful)"))
    ax1.set_ylabel(tx("Adım", "Steps"))
    max_step = max(step_vals, default=0.0)
    ax1.set_ylim(0.0, max(1.0, max_step * 1.18))
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_labels, rotation=15, ha="right")
    _style_axis(ax1)
    for bar, raw in zip(bars1, step_raw):
        txt = f"{raw:.1f}" if raw is not None else tx("Yok", "N/A")
        y_pos = (raw if raw is not None else 0.0) + max(0.03, max_step * 0.02)
        ax1.text(bar.get_x() + bar.get_width() / 2, y_pos, txt, ha="center", va="bottom", fontsize=9)

    # 3. Ortalama Çarpma
    ax2 = axes[2]
    hit_vals = [stats[m]["carpma"] for m in methods]
    bars2 = ax2.bar(x, hit_vals, color=colors, width=0.6)
    ax2.set_title(tx("Ort. Duvar Çarpması", "Avg. Wall Hits"))
    ax2.set_ylabel(tx("Çarpma", "Wall Hits"))
    max_hit = max(hit_vals, default=0.0)
    ax2.set_ylim(0.0, max(1.0, max_hit * 1.18))
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_labels, rotation=15, ha="right")
    _style_axis(ax2)
    for bar, val in zip(bars2, hit_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + max(0.03, max_hit * 0.02),
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig


def generalization_plot(run: dict[str, Any]) -> Figure:
    """Aynı çalışmanın normal ve büyük test sonuçları yan yana (ajan + baseline'lar)."""
    fig = Figure(figsize=(10.5, 4.5), dpi=100)
    axes = fig.subplots(1, 2)
    fig.suptitle(
        f"{_run_label(run)} — {tx('Genelleme', 'Generalization')} ({kind_short('normal')} vs {kind_short('buyuk')})",
        fontsize=12,
        fontweight="bold",
    )

    test_data = run.get("test") or {}
    rows_n = test_data.get("normal") or []
    rows_b = test_data.get("buyuk") or []

    if not rows_n and not rows_b:
        for ax in axes:
            _empty_axis(ax, tx("Test verisi yok", "No test data"))
        fig.tight_layout()
        return fig

    grouped_n = _group_test_by_method(rows_n) if rows_n else {}
    grouped_b = _group_test_by_method(rows_b) if rows_b else {}

    methods: list[str] = []
    for m in list(grouped_n.keys()) + list(grouped_b.keys()):
        if m not in methods:
            methods.append(m)
    method_labels = [method_label(m) for m in methods]

    stats_n = {m: _compute_method_stats(grouped_n[m]) for m in grouped_n}
    stats_b = {m: _compute_method_stats(grouped_b[m]) for m in grouped_b}

    x = np.arange(len(methods))
    width = 0.36

    # Sol: Başarı Oranı
    ax0 = axes[0]
    for i, m in enumerate(methods):
        color = _method_color(m, agent_idx=0)
        vn = stats_n[m]["basari"] if m in stats_n else 0.0
        vb = stats_b[m]["basari"] if m in stats_b else 0.0
        if m in stats_n:
            ax0.bar(x[i] - width / 2, vn, width, color=color, edgecolor="#222222", linewidth=0.6)
            ax0.text(x[i] - width / 2, vn + 0.02, tx(f"%{vn * 100:.0f}", f"{vn * 100:.0f}%"),
                     ha="center", va="bottom", fontsize=8)
        if m in stats_b:
            ax0.bar(x[i] + width / 2, vb, width, color=color, alpha=0.65, hatch="//",
                    edgecolor="#222222", linewidth=0.6)
            ax0.text(x[i] + width / 2, vb + 0.02, tx(f"%{vb * 100:.0f}", f"{vb * 100:.0f}%"),
                     ha="center", va="bottom", fontsize=8)

    ax0.set_title(tx("Başarı Oranı", "Success Rate"))
    ax0.set_ylabel(tx("Oran", "Rate"))
    ax0.set_ylim(0.0, 1.15)
    ax0.set_xticks(x)
    ax0.set_xticklabels(method_labels, rotation=15, ha="right")
    _style_axis(ax0)

    # Sağ: Verimlilik (Adım / En Kısa Yol; 1.0 = en kısa yol)
    ax1 = axes[1]
    eff_vals_all: list[float] = []
    for i, m in enumerate(methods):
        color = _method_color(m, agent_idx=0)
        en = stats_n[m]["verim"] if m in stats_n else None
        eb = stats_b[m]["verim"] if m in stats_b else None
        if m in stats_n:
            val = en if en is not None else 0.0
            eff_vals_all.append(val)
            ax1.bar(x[i] - width / 2, val, width, color=color, edgecolor="#222222", linewidth=0.6)
            txt = f"{en:.2f}" if en is not None else tx("Yok", "N/A")
            ax1.text(x[i] - width / 2, val + 0.04, txt, ha="center", va="bottom", fontsize=8)
        if m in stats_b:
            val = eb if eb is not None else 0.0
            eff_vals_all.append(val)
            ax1.bar(x[i] + width / 2, val, width, color=color, alpha=0.65, hatch="//",
                    edgecolor="#222222", linewidth=0.6)
            txt = f"{eb:.2f}" if eb is not None else tx("Yok", "N/A")
            ax1.text(x[i] + width / 2, val + 0.04, txt, ha="center", va="bottom", fontsize=8)

    ax1.axhline(1.0, color="#444444", linestyle=":", linewidth=1.0,
                label=tx("En kısa yol (1.0)", "Shortest path (1.0)"))
    ax1.set_title(tx("Verimlilik (Adım / En Kısa Yol)", "Efficiency (Steps / Shortest Path)"))
    ax1.set_ylabel(tx("Oran (düşük = daha iyi)", "Ratio (lower = better)"))
    max_eff = max(eff_vals_all, default=1.0)
    ax1.set_ylim(0.0, max(1.4, max_eff * 1.2))
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_labels, rotation=15, ha="right")
    _style_axis(ax1)

    legend_handles = [
        Patch(facecolor="#aaaaaa", edgecolor="#222222", label=kind_short("normal")),
        Patch(facecolor="#aaaaaa", alpha=0.65, hatch="//", edgecolor="#222222", label=kind_short("buyuk")),
    ]
    ax0.legend(handles=legend_handles, frameon=False, loc="upper right")
    ax1.legend(handles=legend_handles, frameon=False, loc="upper right")

    fig.tight_layout()
    return fig


def compare_runs(runs: Sequence[dict[str, Any]], kind: str = "normal") -> Figure:
    """Birden çok çalışmanın (varyasyonlar ya da farklı ajanlar) test başarısı ve verimliliği."""
    fig = Figure(figsize=(10.5, 4.6), dpi=100)
    axes = fig.subplots(1, 2)
    kind_lbl = kind_label(kind)
    fig.suptitle(
        tx(f"Çalışma Karşılaştırması ({kind_lbl})", f"Run Comparison ({kind_lbl})"),
        fontsize=12,
        fontweight="bold",
    )

    labels: list[str] = []
    colors: list[str] = []
    succ_vals: list[float] = []
    succ_stds: list[float] = []
    eff_raw: list[float | None] = []
    baseline_ref: dict[str, dict[str, Any]] = {}

    for idx, run in enumerate(runs):
        trows = (run.get("test") or {}).get(kind) or []
        if not trows:
            continue
        grouped = _group_test_by_method(trows)
        # Baseline referanslarını ilk bulduğumuz çalışmadan sakla
        for bname in ("sag_el", "random"):
            if bname not in baseline_ref and bname in grouped:
                baseline_ref[bname] = _compute_method_stats(grouped[bname])

        # Ajan (baseline olmayan) satırları seç
        non_base = [
            m for m, rows in grouped.items() if not any(bool(r.get("is_baseline")) for r in rows)
        ]
        target_method = non_base[0] if non_base else next(iter(grouped))
        st = _compute_method_stats(grouped[target_method])

        labels.append(_run_label(run))
        colors.append(AGENT_PALETTE[idx % len(AGENT_PALETTE)])
        succ_vals.append(st["basari"])
        succ_stds.append(st["basari_std"])
        eff_raw.append(st["verim"])

    if not labels:
        for ax in axes:
            _empty_axis(ax, tx("Karşılaştırılacak test verisi yok", "No test data to compare"))
        fig.tight_layout()
        return fig

    x = np.arange(len(labels))

    # Sol: Başarı Oranı
    ax0 = axes[0]
    bars0 = ax0.bar(x, succ_vals, yerr=succ_stds if any(s > 0 for s in succ_stds) else None,
                    capsize=4, color=colors, width=0.55)
    for bname, bstats in baseline_ref.items():
        ax0.axhline(bstats["basari"], color=_method_color(bname), linestyle="--",
                    linewidth=1.3, label=f"{method_label(bname)} ({bstats['basari']:.2f})")
    ax0.set_title(tx("Test Başarı Oranı", "Test Success Rate"))
    ax0.set_ylabel(tx("Oran", "Rate"))
    ax0.set_ylim(0.0, 1.15)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=18, ha="right")
    _style_axis(ax0)
    if baseline_ref:
        ax0.legend(frameon=False, fontsize=8)
    for bar, val in zip(bars0, succ_vals):
        ax0.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                 tx(f"%{val * 100:.1f}", f"{val * 100:.1f}%"),
                 ha="center", va="bottom", fontsize=9)

    # Sağ: Verimlilik
    ax1 = axes[1]
    eff_vals = [v if v is not None else 0.0 for v in eff_raw]
    bars1 = ax1.bar(x, eff_vals, color=colors, width=0.55)
    for bname, bstats in baseline_ref.items():
        if bstats["verim"] is not None:
            ax1.axhline(bstats["verim"], color=_method_color(bname), linestyle="--",
                        linewidth=1.3, label=f"{method_label(bname)} ({bstats['verim']:.2f})")
    ax1.set_title(tx("Verimlilik (Adım / En Kısa Yol)", "Efficiency (Steps / Shortest Path)"))
    ax1.set_ylabel(tx("Oran (düşük = daha iyi)", "Ratio (lower = better)"))
    ref_max = max((b["verim"] for b in baseline_ref.values() if b["verim"] is not None), default=1.0)
    max_eff = max(max(eff_vals, default=0.0), ref_max, 1.0)
    ax1.set_ylim(0.0, max_eff * 1.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=18, ha="right")
    _style_axis(ax1)
    if any(b["verim"] is not None for b in baseline_ref.values()):
        ax1.legend(frameon=False, fontsize=8)
    for bar, raw in zip(bars1, eff_raw):
        txt = f"{raw:.2f}" if raw is not None else tx("Yok", "N/A")
        y_pos = (raw if raw is not None else 0.0) + max(0.03, max_eff * 0.02)
        ax1.text(bar.get_x() + bar.get_width() / 2, y_pos, txt, ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig


def save_all_plots(run_dir: Path) -> list[Path]:
    """Tek çalışmanın grafiklerini grafikler/*.png olarak kaydeder."""
    run_dir = Path(run_dir)
    run = load_run(run_dir)
    plots_dir = run_dir / "grafikler"
    plots_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for metric, fname in (
        ("success", "ogrenme_basari.png"),
        ("total_reward", "ogrenme_odul.png"),
        ("steps", "ogrenme_adim.png"),
    ):
        fig = learning_curve([run], metric=metric)
        saved.append(_save_figure(fig, plots_dir / fname))

    has_normal = bool(run["test"].get("normal"))
    has_buyuk = bool(run["test"].get("buyuk"))

    if has_normal:
        saved.append(_save_figure(baseline_bars(run, kind="normal"), plots_dir / "test_normal.png"))
    if has_buyuk:
        saved.append(_save_figure(baseline_bars(run, kind="buyuk"), plots_dir / "test_buyuk.png"))
    if has_normal and has_buyuk:
        saved.append(_save_figure(generalization_plot(run), plots_dir / "genelleme.png"))

    return saved
