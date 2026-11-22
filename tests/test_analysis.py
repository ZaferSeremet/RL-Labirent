"""engine.analysis ve engine.report birim testleri (sahte sonuç verileriyle)."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from engine import analysis, i18n, report, storage
from tests.i18n_check import figure_turkish_leftovers


@pytest.fixture(autouse=True)
def _restore_turkish_language():
    i18n.set_language("tr")
    yield
    i18n.set_language("tr")


# ----------------------------------------------------------------- sahte veri yardımcısı

def _make_fake_run(
    root: Path,
    *,
    agent_name: str = "q_ajan",
    label: str = "deney1",
    episodes: int = 200,
    repeats: int = 3,
    include_normal: bool = True,
    include_buyuk: bool = True,
    agent_normal_succ_by_rep: tuple[int, ...] = (8, 9, 7),
    agent_buyuk_succ_by_rep: tuple[int, ...] = (5, 6, 4),
    hparams: dict | None = None,
    reward_overrides: dict | None = None,
) -> Path:
    run_dir = storage.new_run_dir(agent_name, label, root=root)
    hparams = hparams or {"lr": 0.1, "gamma": 0.99}
    reward_overrides = reward_overrides or {"step": -0.01, "wall": -0.05, "exit": 5.0}

    settings = {
        "run_id": run_dir.name,
        "created": "2026-11-22T12:00:00",
        "durum": "tamamlandi",
        "agent": {
            "name": agent_name,
            "class": f"agents.{agent_name}.FakeAgent",
            "obs_mode": "full",
            "is_baseline": False,
        },
        "train_config": {
            "agent_name": agent_name,
            "episodes": episodes,
            "repeats": repeats,
            "label": label,
            "seed": 0,
            "reward_overrides": reward_overrides,
            "env_overrides": {},
            "hparams": hparams,
        },
        "env": {
            "maze_size": 8,
            "view_radius": 1,
            "max_steps": 381,
            "rewards": {**reward_overrides, "new_tile": 0.0, "timeout": 0.0},
        },
        "hyperparameters": hparams,
        "tamamlanan_tekrar": repeats,
    }
    storage.write_settings(run_dir, settings)

    # Eğitim logu: zamanla artan başarı ve ödül, azalan adım
    with storage.CsvAppender(storage.train_log_path(run_dir), storage.TRAIN_FIELDS) as w:
        for r in range(repeats):
            for ep in range(1, episodes + 1):
                prog = ep / episodes
                # Deterministik ama tekrarlar arası hafif farklı başarı deseni
                succ = ((ep + r) % max(1, int(round(1.0 / max(0.15, prog))))) == 0 or prog > 0.85
                if ep > episodes - (episodes // 10):
                    # Son %10'da el ile doğrulaması kolay deterministik değerler
                    succ = (ep % 2 == 0) if r == 0 else True
                steps = 40 - int(20 * prog) + r
                total_reward = (5.0 if succ else -1.0) - 0.01 * steps + 0.1 * r
                w.write({
                    "repeat": r,
                    "episode": ep,
                    "total_reward": total_reward,
                    "steps": steps,
                    "success": succ,
                    "wall_hits": 2 if succ else 8,
                    "visited_tiles": 15,
                    "shortest_path": 12,
                    "episode_seed": ep + r * 1000,
                })

    def _write_test_csv(kind: str, base_seed: int, succ_counts: tuple[int, ...], sp: int) -> None:
        with storage.CsvAppender(storage.test_csv_path(run_dir, kind), storage.TEST_FIELDS) as w:
            # Ajan: tekrar başına 10 seed
            for r in range(repeats):
                n_succ = succ_counts[r % len(succ_counts)]
                for i in range(10):
                    succ = i < n_succ
                    steps = (sp + 4 + r) if succ else (sp * 10)
                    eff = (steps / sp) if succ else None
                    w.write({
                        "method": agent_name,
                        "is_baseline": False,
                        "repeat": r,
                        "seed": base_seed + i,
                        "total_reward": (5.0 - 0.01 * steps) if succ else -2.0,
                        "steps": steps,
                        "success": succ,
                        "wall_hits": 1 if succ else 12,
                        "visited_tiles": sp,
                        "shortest_path": sp,
                        "efficiency": eff,
                    })
            # Baseline'lar: repeat=0, is_baseline=1
            for bname, b_succ in (("random", 1), ("sag_el", 10)):
                for i in range(10):
                    succ = i < b_succ
                    steps = (sp * 3) if succ else (sp * 10)
                    eff = (steps / sp) if succ else None
                    w.write({
                        "method": bname,
                        "is_baseline": True,
                        "repeat": 0,
                        "seed": base_seed + i,
                        "total_reward": (4.0 if succ else -3.0),
                        "steps": steps,
                        "success": succ,
                        "wall_hits": 0 if bname == "sag_el" else 20,
                        "visited_tiles": sp,
                        "shortest_path": sp,
                        "efficiency": eff,
                    })

    if include_normal:
        _write_test_csv("normal", 10000, agent_normal_succ_by_rep, sp=10)
    if include_buyuk:
        _write_test_csv("buyuk", 20000, agent_buyuk_succ_by_rep, sp=20)

    return run_dir


# ----------------------------------------------------------------- testler

def test_load_run_missing_and_present(tmp_path: Path) -> None:
    empty_dir = tmp_path / "bos_calisma"
    empty_dir.mkdir()
    loaded_empty = analysis.load_run(empty_dir)
    assert loaded_empty["dir"] == empty_dir
    assert loaded_empty["settings"] == {}
    assert loaded_empty["train"] == []
    assert loaded_empty["test"] == {"normal": [], "buyuk": []}

    run_dir = _make_fake_run(tmp_path, include_buyuk=False)
    loaded = analysis.load_run(run_dir)
    assert loaded["settings"]["agent"]["name"] == "q_ajan"
    assert len(loaded["train"]) == 200 * 3
    assert len(loaded["test"]["normal"]) == 3 * 10 + 2 * 10
    assert loaded["test"]["buyuk"] == []


def test_summarize_exact_calculations(tmp_path: Path) -> None:
    # Küçük, elle hesaplaması birebir takip edilebilen özel bir çalışma
    run_dir = storage.new_run_dir("test_ajan", "ozet", root=tmp_path)
    storage.write_settings(run_dir, {
        "run_id": run_dir.name,
        "agent": {"name": "test_ajan"},
        "train_config": {"episodes": 20, "repeats": 2, "label": "ozet"},
    })
    # 20 bölüm -> son %10 = 2 bölüm (ep 19, 20)
    with storage.CsvAppender(storage.train_log_path(run_dir), storage.TRAIN_FIELDS) as w:
        for r in (0, 1):
            for ep in range(1, 21):
                # ep 19, 20 için kontrollü değerler; öncekiler farklı
                if ep in (19, 20):
                    succ = (r == 0 and ep == 20) or (r == 1)  # r=0: 0.5, r=1: 1.0 -> ort 0.75
                    rew = 1.0 if r == 0 else 3.0            # r=0: 1.0, r=1: 3.0 -> ort 2.0
                    steps = 10 if r == 0 else 20            # r=0: 10.0, r=1: 20.0 -> ort 15.0
                else:
                    succ = False
                    rew = -10.0
                    steps = 99
                w.write({
                    "repeat": r, "episode": ep, "total_reward": rew, "steps": steps,
                    "success": succ, "wall_hits": 1, "visited_tiles": 5,
                    "shortest_path": 5, "episode_seed": ep,
                })

    # test_normal: repeat 0 -> 1/2 başarılı (steps=10, eff=2.0), repeat 1 -> 2/2 başarılı (steps=14,16; eff=2.8,3.2)
    with storage.CsvAppender(storage.test_csv_path(run_dir, "normal"), storage.TEST_FIELDS) as w:
        rows = [
            {"method": "test_ajan", "is_baseline": False, "repeat": 0, "seed": 1,
             "total_reward": 4.0, "steps": 10, "success": True, "wall_hits": 1,
             "visited_tiles": 5, "shortest_path": 5, "efficiency": 2.0},
            {"method": "test_ajan", "is_baseline": False, "repeat": 0, "seed": 2,
             "total_reward": -2.0, "steps": 100, "success": False, "wall_hits": 5,
             "visited_tiles": 5, "shortest_path": 5, "efficiency": None},
            {"method": "test_ajan", "is_baseline": False, "repeat": 1, "seed": 1,
             "total_reward": 4.0, "steps": 14, "success": True, "wall_hits": 0,
             "visited_tiles": 5, "shortest_path": 5, "efficiency": 2.8},
            {"method": "test_ajan", "is_baseline": False, "repeat": 1, "seed": 2,
             "total_reward": 2.0, "steps": 16, "success": True, "wall_hits": 2,
             "visited_tiles": 5, "shortest_path": 5, "efficiency": 3.2},
            # baseline: tamamen başarısız (adim ve verim None olmalı)
            {"method": "random", "is_baseline": True, "repeat": 0, "seed": 1,
             "total_reward": -3.0, "steps": 100, "success": False, "wall_hits": 10,
             "visited_tiles": 5, "shortest_path": 5, "efficiency": None},
        ]
        for row in rows:
            w.write(row)

    summary = analysis.summarize(run_dir)
    ozet_path = run_dir / "ozet.json"
    assert ozet_path.is_file()
    disk_summary = json.loads(ozet_path.read_text(encoding="utf-8"))
    assert disk_summary == summary

    assert summary["run_id"] == run_dir.name
    assert summary["agent"] == "test_ajan"
    assert summary["label"] == "ozet"

    eg = summary["egitim"]
    assert eg["bolum"] == 20
    assert eg["tekrar"] == 2
    assert eg["son_basari"] == pytest.approx(0.75)
    assert eg["son_ortalama_odul"] == pytest.approx(2.0)
    assert eg["son_ortalama_adim"] == pytest.approx(15.0)

    # test_buyuk yok -> boş sözlük
    assert summary["test"]["buyuk"] == {}

    ajan_test = summary["test"]["normal"]["test_ajan"]
    assert ajan_test["n"] == 4
    assert ajan_test["basari"] == pytest.approx(0.75)
    # Tekrarlar arası std: [0.5, 1.0] -> std = 0.25
    assert ajan_test["basari_std"] == pytest.approx(0.25)
    # Adım yalnız başarılılardan: (10 + 14 + 16) / 3 = 40 / 3
    assert ajan_test["adim"] == pytest.approx(40.0 / 3.0)
    # Verim yalnız başarılılardan: (2.0 + 2.8 + 3.2) / 3 = 8.0 / 3
    assert ajan_test["verim"] == pytest.approx(8.0 / 3.0)
    assert ajan_test["odul"] == pytest.approx((4.0 - 2.0 + 4.0 + 2.0) / 4.0)
    assert ajan_test["carpma"] == pytest.approx((1 + 5 + 0 + 2) / 4.0)

    rnd_test = summary["test"]["normal"]["random"]
    assert rnd_test["basari"] == pytest.approx(0.0)
    assert rnd_test["basari_std"] == pytest.approx(0.0)
    assert rnd_test["adim"] is None
    assert rnd_test["verim"] is None


def test_plots_single_and_multi_runs(tmp_path: Path) -> None:
    r1_dir = _make_fake_run(tmp_path, agent_name="q_ajan", label="baz", episodes=200, repeats=3)
    r2_dir = _make_fake_run(
        tmp_path, agent_name="sarsa", label="hizli", episodes=200, repeats=2,
        agent_normal_succ_by_rep=(6, 7), agent_buyuk_succ_by_rep=(3, 4),
    )
    r1 = analysis.load_run(r1_dir)
    r2 = analysis.load_run(r2_dir)

    # Öğrenme eğrileri (tek ve çoklu çalışma)
    for metric in ("success", "total_reward", "steps"):
        fig1 = analysis.learning_curve([r1], metric=metric)
        assert isinstance(fig1, Figure)
        assert len(fig1.axes[0].lines) == 1
        assert len(fig1.axes[0].collections) >= 1  # std gölgesi

        fig2 = analysis.learning_curve([r1, r2], metric=metric, window=10)
        assert isinstance(fig2, Figure)
        assert len(fig2.axes[0].lines) == 2

    # Baseline çubuk grafiği + sabit renk kontrolü (random gri, sag_el turuncu)
    fig_b = analysis.baseline_bars(r1, kind="normal")
    assert isinstance(fig_b, Figure)
    assert len(fig_b.axes) == 3
    for ax in fig_b.axes:
        assert len(ax.patches) == 3  # ajan, random, sag_el
    labels0 = [t.get_text() for t in fig_b.axes[0].get_xticklabels()]
    idx_rnd = labels0.index("Random")
    idx_sag = labels0.index("Sağ el")
    from matplotlib.colors import to_hex
    assert to_hex(fig_b.axes[0].patches[idx_rnd].get_facecolor()) == "#7f7f7f"
    assert to_hex(fig_b.axes[0].patches[idx_sag].get_facecolor()) == "#ff7f0e"

    # Genelleme grafiği
    fig_g = analysis.generalization_plot(r1)
    assert isinstance(fig_g, Figure)
    assert len(fig_g.axes) == 2
    assert len(fig_g.axes[0].patches) == 6  # 3 yöntem x 2 tür (normal, büyük)
    assert len(fig_g.axes[1].patches) == 6

    # Karşılaştırma grafiği (tek ve iki çalışma)
    fig_c1 = analysis.compare_runs([r1], kind="normal")
    assert isinstance(fig_c1, Figure)
    assert len(fig_c1.axes[0].patches) == 1

    fig_c2 = analysis.compare_runs([r1, r2], kind="normal")
    assert isinstance(fig_c2, Figure)
    assert len(fig_c2.axes[0].patches) == 2
    assert len(fig_c2.axes[1].patches) == 2


def test_save_all_plots_full_and_missing_tests(tmp_path: Path) -> None:
    full_dir = _make_fake_run(tmp_path, agent_name="q_ajan", label="tam")
    saved_full = analysis.save_all_plots(full_dir)
    names_full = {p.name for p in saved_full}
    assert names_full == {
        "ogrenme_basari.png",
        "ogrenme_odul.png",
        "ogrenme_adim.png",
        "test_normal.png",
        "test_buyuk.png",
        "genelleme.png",
    }
    assert all(p.is_file() and p.stat().st_size > 0 for p in saved_full)

    # Test dosyası yoksa ilgili png'ler atlanır
    no_test_dir = _make_fake_run(
        tmp_path, agent_name="q_ajan", label="testsiz",
        include_normal=False, include_buyuk=False,
    )
    saved_no_test = analysis.save_all_plots(no_test_dir)
    names_no_test = {p.name for p in saved_no_test}
    assert names_no_test == {"ogrenme_basari.png", "ogrenme_odul.png", "ogrenme_adim.png"}
    assert not (no_test_dir / "grafikler" / "test_normal.png").exists()
    assert not (no_test_dir / "grafikler" / "test_buyuk.png").exists()
    assert not (no_test_dir / "grafikler" / "genelleme.png").exists()


def test_export_zip_and_import_runs_roundtrip_and_errors(tmp_path: Path) -> None:
    src_root = tmp_path / "src_results"
    dest_root = tmp_path / "dest_results"
    run_dir = _make_fake_run(src_root, agent_name="q_ajan", label="paylas")
    (run_dir / "model_rep0.pkl").write_bytes(b"fake-model-bytes")

    zip_path = tmp_path / "paket.zip"
    out_zip = report.export_run_zip(run_dir, zip_path)
    assert out_zip == zip_path
    assert zip_path.is_file()

    # Zip kök klasörü run_id olmalı
    with zipfile.ZipFile(zip_path, "r") as zf:
        roots = {name.split("/")[0] for name in zf.namelist()}
        assert roots == {run_dir.name}

    # İlk içe aktarma (zip'ten)
    imported1 = report.import_runs([zip_path], root=dest_root)
    assert len(imported1) == 1
    assert imported1[0] == dest_root / run_dir.name
    assert (imported1[0] / "ayarlar.json").read_text(encoding="utf-8") == (run_dir / "ayarlar.json").read_text(encoding="utf-8")
    assert (imported1[0] / "egitim_log.csv").read_text(encoding="utf-8") == (run_dir / "egitim_log.csv").read_text(encoding="utf-8")
    assert (imported1[0] / "model_rep0.pkl").read_bytes() == b"fake-model-bytes"

    # İkinci içe aktarma (klasörden, aynı ad çakışması -> _2)
    imported2 = report.import_runs([run_dir], root=dest_root)
    assert len(imported2) == 1
    assert imported2[0] == dest_root / f"{run_dir.name}_2"
    assert (imported2[0] / "ayarlar.json").is_file()

    # Geçersiz kaynaklar -> ValueError
    with pytest.raises(ValueError):
        report.import_runs([tmp_path / "olmayan_yol"], root=dest_root)

    bad_file = tmp_path / "not_a_zip.txt"
    bad_file.write_text("merhaba", encoding="utf-8")
    with pytest.raises(ValueError):
        report.import_runs([bad_file], root=dest_root)

    empty_folder = tmp_path / "bos_klasor"
    empty_folder.mkdir()
    with pytest.raises(ValueError):
        report.import_runs([empty_folder], root=dest_root)

    bad_zip = tmp_path / "bos.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("rastgele/dosya.txt", "icerik")
    with pytest.raises(ValueError):
        report.import_runs([bad_zip], root=dest_root)


def test_export_report_complete_and_missing_test(tmp_path: Path) -> None:
    r1 = _make_fake_run(tmp_path, agent_name="q_ajan", label="baz", episodes=100, repeats=2)
    r2 = _make_fake_run(
        tmp_path, agent_name="sarsa", label="deney2", episodes=100, repeats=2,
        include_normal=True, include_buyuk=False,
    )
    r_no_test = _make_fake_run(
        tmp_path, agent_name="dqn", label="testsiz", episodes=100, repeats=2,
        include_normal=False, include_buyuk=False,
    )

    # Çoklu rapor (biri eksik testli) — out_dir altında tek rapor_YYYY-MM-DD_HH-MM-SS klasörü
    out_multi = tmp_path / "rapor_coklu"
    html_path = report.export_report([r1, r2, r_no_test], out_multi)
    report_dir = html_path.parent
    assert report_dir.parent == out_multi
    assert re.fullmatch(r"rapor_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_\d+)?", report_dir.name)
    assert list(out_multi.iterdir()) == [report_dir]
    assert html_path == report_dir / "rapor.html"
    assert html_path.is_file()
    assert (report_dir / "rapor.md").is_file()
    assert (report_dir / "grafikler").is_dir()
    assert (report_dir / "tablolar").is_dir()
    assert len(list((report_dir / "tablolar").glob("*.csv"))) >= 2

    html_text = html_path.read_text(encoding="utf-8")
    assert '<html lang="tr">' in html_text
    img_srcs = re.findall(r"<img[^>]+src=['\"]([^'\"]+)['\"]", html_text)
    assert len(img_srcs) >= 4
    for src in img_srcs:
        assert not Path(src).is_absolute()
        assert src.startswith("grafikler/")
        assert (report_dir / src).is_file(), f"Eksik görsel: {src}"

    # Yalnız test verisi hiç olmayan tek çalışmayla da hata vermemeli
    out_notest = tmp_path / "rapor_testsiz"
    html_notest = report.export_report([r_no_test], out_notest)
    notest_dir = html_notest.parent
    assert notest_dir.parent == out_notest
    assert list(out_notest.iterdir()) == [notest_dir]
    assert html_notest.name == "rapor.html"
    assert html_notest.is_file()
    assert (notest_dir / "rapor.md").is_file()
    assert (notest_dir / "tablolar" / "ayarlar.csv").is_file()
    html_nt_text = html_notest.read_text(encoding="utf-8")
    for src in re.findall(r"<img[^>]+src=['\"]([^'\"]+)['\"]", html_nt_text):
        assert (notest_dir / src).is_file()


def test_plots_english_no_turkish_leftovers(tmp_path: Path) -> None:
    r1_dir = _make_fake_run(tmp_path, agent_name="q_agent", label="base", episodes=80, repeats=2)
    r2_dir = _make_fake_run(
        tmp_path, agent_name="sarsa", label="fast", episodes=80, repeats=2,
        agent_normal_succ_by_rep=(6, 7), agent_buyuk_succ_by_rep=(3, 4),
    )
    empty_dir = tmp_path / "empty_run"
    empty_dir.mkdir()
    r1 = analysis.load_run(r1_dir)
    r2 = analysis.load_run(r2_dir)
    r_empty = analysis.load_run(empty_dir)

    i18n.set_language("en")
    figs = [
        *(analysis.learning_curve([r1, r2], metric=m) for m in ("success", "total_reward", "steps")),
        analysis.learning_curve([r_empty], metric="success"),
        analysis.baseline_bars(r1, kind="normal"),
        analysis.baseline_bars(r1, kind="buyuk"),
        analysis.baseline_bars(r_empty, kind="normal"),
        analysis.generalization_plot(r1),
        analysis.generalization_plot(r_empty),
        analysis.compare_runs([r1, r2], kind="normal"),
        analysis.compare_runs([r1, r2], kind="buyuk"),
        analysis.compare_runs([r_empty], kind="normal"),
    ]
    for fig in figs:
        assert figure_turkish_leftovers(fig) == []


def test_export_report_english_and_unique_subfolders(tmp_path: Path) -> None:
    r1 = _make_fake_run(tmp_path, agent_name="q_agent", label="exp1", episodes=60, repeats=2)
    r2 = _make_fake_run(
        tmp_path, agent_name="sarsa", label="exp2", episodes=60, repeats=2,
        agent_normal_succ_by_rep=(6, 7), agent_buyuk_succ_by_rep=(3, 4),
    )

    i18n.set_language("en")
    out_dir = tmp_path / "en_reports"
    html_path = report.export_report([r1, r2], out_dir)
    report_dir = html_path.parent
    assert report_dir.parent == out_dir
    assert re.fullmatch(r"report_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_\d+)?", report_dir.name)
    assert list(out_dir.iterdir()) == [report_dir]

    assert html_path == report_dir / "report.html"
    assert html_path.is_file()
    assert (report_dir / "report.md").is_file()
    assert (report_dir / "figures").is_dir()
    assert (report_dir / "tables").is_dir()
    assert (report_dir / "tables" / "settings.csv").is_file()
    assert (report_dir / "tables" / "comparison.csv").is_file()
    assert (report_dir / "figures" / "learning_success.png").is_file()
    assert (report_dir / "figures" / "comparison_normal.png").is_file()
    assert (report_dir / "figures" / "comparison_large.png").is_file()

    html_text = html_path.read_text(encoding="utf-8")
    assert '<html lang="en">' in html_text
    assert not re.search(r"[çğıöşüÇĞİÖŞÜ]", html_text)
    assert "Right-hand" in html_text

    img_srcs = re.findall(r"<img[^>]+src=['\"]([^'\"]+)['\"]", html_text)
    assert len(img_srcs) >= 4
    for src in img_srcs:
        assert src.startswith("figures/")
        assert (report_dir / src).is_file()

    # İki kez arka arkaya çağrılınca aynı out_dir altında iki ayrı klasör üretir
    html_path_2 = report.export_report([r1], out_dir)
    report_dir_2 = html_path_2.parent
    assert report_dir_2.parent == out_dir
    assert report_dir_2 != report_dir
    assert html_path_2.is_file()
    assert len(list(out_dir.iterdir())) == 2
