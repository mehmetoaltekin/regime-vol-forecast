"""
Dress rehearsal of the full run on synthetic data: every variant, every stage, worker processes, an interruption and a
resume. Slow (minutes), so it is not part of the default run: `uv run pytest -m slow`.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import rvf.study as S

from test_realized import bars
from test_study import small_cfg, write_m1, write_prespec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_study as R  # noqa: E402

pytestmark = pytest.mark.slow

RULES = {"interval_min": 5, "include_session_break": True, "excluded_windows": [[1380, 60]], "merge_sunday": True,
         "short_day_share": 0.5, "day_length": "active", "median_day_length": 264, "ohlc_from_included": True,
         "stale_share": 0.5, "anchor_gap_days": 5}
CLEAN = {"bounce_sd": 20, "bounce_reversal": 0.75, "bounce_lookback_days": 20}
SYMBOLS = (("AAA", "FX", 11), ("BBB", "FX", 12), ("CCC", "Precious metals", 13), ("DDD", "Precious metals", 14))


def synthetic_m1(n_days: int, seed: int) -> pd.DataFrame:
    """Minute bars of a market with calm and stressed days (persistent regimes), 24 hours a day."""
    rng = np.random.default_rng(seed)
    state = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        state[t] = state[t - 1] if rng.random() < (0.97 if state[t - 1] == 0 else 0.92) else 1 - state[t - 1]
    sigma = np.where(state == 0, 0.6, 2.0)[:, None] * rng.uniform(0.8, 1.2, (n_days, 1))
    r = rng.normal(0, 1, (n_days, 1440)) * sigma / np.sqrt(1440)
    close = 100.0 * np.exp(np.cumsum(r.ravel()) / 100)
    return bars(pd.bdate_range("2023-01-02", periods=n_days), np.arange(1440), close)


def make_repo(tmp_path: Path, variants, **cfg_kw):
    repo = tmp_path / "repo"
    (repo / "results").mkdir(parents=True)
    raw = tmp_path / "raw"
    raw.mkdir()
    table = ["file\tbytes\tsha256"]
    for sym, _, seed in SYMBOLS:
        f = raw / f"{sym}_M1_20230102_20231229.csv"
        write_m1(f, synthetic_m1(260, seed))
        table.append(f"{f.name}\t{f.stat().st_size}\t{S.sha256_file(f)}")
    (repo / "results" / "raw_data_sha256.tsv").write_text("\n".join(table) + "\n")
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)
    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    (repo / "tracked.txt").write_text("x")
    write_prespec(repo / "PRESPEC.md", "1.0", "Approved")
    run("add", "tracked.txt", "PRESPEC.md")
    run("commit", "-q", "-m", "x")
    run("tag", "prespec-v1.0")
    instruments = [S.InstrumentSpec(s, "2023-02-01", "2023-12-29", RULES, c, CLEAN) for s, c, _ in SYMBOLS]
    cfg = small_cfg(prespec_version="1.0", instruments=instruments, window=60, refit_every=30, horizons=(1, 5), min_ess=10,
                    ms_starts=1, mcs_reps=100, mcs_block=5, variants=tuple(variants), alt_windows=(40, 80),
                    long_history={"AAA": "2023-01-02", "BBB": "2023-01-02", "CCC": "2023-01-02"}, **cfg_kw)
    cfg_path = repo / "cfg.json"
    cfg_path.write_text(cfg.to_json())
    return repo, raw, cfg, cfg_path


def attach(monkeypatch, repo):
    monkeypatch.setattr(R, "ROOT", repo)
    monkeypatch.setattr(R, "RESULTS", repo / "results")


def test_every_variant_runs_through_all_three_stages_with_worker_processes(tmp_path, monkeypatch):
    repo, raw, cfg, cfg_path = make_repo(tmp_path, S.VARIANTS)
    attach(monkeypatch, repo)
    R.run_full(cfg_path, workers=3, raw_dir=raw)
    res = repo / "results"
    assert (res / "RUN_INFO.md").exists() and not (res / "RUN_STARTED.lock").exists() and (res / "run_progress.log").exists()
    info = (res / "RUN_INFO.md").read_text()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert commit in info and "hmm-regime 0.2.0" in info and "3 worker(s)" in info and "daily__AAA__long.parquet" in info
    for sym, _, _ in SYMBOLS:
        assert f"{sym}_M1_20230102_20231229.csv" in info
    for v in S.VARIANTS:
        symbols = S.variant_symbols(cfg, v)
        assert symbols == (["AAA", "BBB", "CCC"] if v == "long_history" else ["AAA", "BBB", "CCC", "DDD"])
        for s in symbols:
            assert S.task_done(res / "forecasts", v, s)["status"] == "ok", (v, s)
        mean_loss = pd.read_csv(res / "tables" / f"{v}__mean_loss.csv")
        assert set(mean_loss["instrument"]) == set(symbols), v
        assert (res / "figures" / f"{v}__prob_stress.pdf").exists() and (res / "figures" / f"{v}__cum_qlike_diff.pdf").exists()
        assert (res / "tables" / f"{v}__counters.tex").exists() and (res / "tables" / f"{v}__regime_summary.csv").exists()
    assert (res / "tables" / "mse_loss__mean_loss.csv").exists()
    es = pd.read_csv(res / "tables" / "primary__effect_sizes.csv")
    assert {"pooled", "AAA", "BBB", "CCC", "DDD"} == set(es["scope"]) and es["ci_low_pct"].notna().all()
    assert len(pd.read_csv(res / "tables" / "primary__dm_by_instrument.csv").query("family == 'H3' and h == 1")) == 8   # 2 x 4 tests
    assert len(list((repo / "data" / "clean").glob("daily__*.parquet"))) == 7

    # PRESPEC 7: every run that shares the sample has the same evaluation days
    days = lambda v, s: set(pd.read_parquet(res / "forecasts" / f"{v}__{s}__forecasts.parquet").query("h == 1")["target_day"])
    same_days = ["primary", "rv_no_break", "rv_no_rollover", "rv_1min", "parkinson", "garman_klass", "hmm_rolling", "hmm_5bar",
                 "k_bic", "log_har", "hmm_log_rv", "harq"]
    for s in ("AAA", "DDD"):
        ref = days("primary", s)
        assert len(ref) > 100
        for v in same_days:
            assert days(v, s) == ref, (v, s)
        assert days("window_750", s) != ref and days("vintage", s) < ref and days("long_history", "AAA") != days("primary", "AAA")
    # PRESPEC 4 and 16: a target variant changes the target only
    prim = pd.read_parquet(res / "forecasts" / "primary__AAA__forecasts.parquet").query("h == 1 and model == 'm3'").set_index("target_day")
    park = pd.read_parquet(res / "forecasts" / "parkinson__AAA__forecasts.parquet").query("h == 1 and model == 'm3'").set_index("target_day")
    bundle = pd.read_parquet(S.bundle_file(repo / "data" / "clean", "AAA", "main"))
    assert np.allclose(park["target"], bundle["parkinson"].reindex(park.index)) and not np.allclose(park["target"], prim["target"])
    assert np.allclose(prim["target"], bundle["rv"].reindex(prim.index))
    with pytest.raises(SystemExit, match="started before"):
        R.run_full(cfg_path, workers=1, raw_dir=raw)
    with pytest.raises(SystemExit, match="started before"):
        R.run_full(cfg_path, workers=1, resume=True, raw_dir=raw)                          # a finished run cannot be resumed


def run_info_lock_fields(repo):
    return json.loads((repo / "results" / "RUN_STARTED.lock").read_text())


def test_an_interrupted_run_is_resumed_and_gives_the_same_results(tmp_path, monkeypatch):
    variants = ("primary", "harq", "long_history")
    repo, raw, cfg, cfg_path = make_repo(tmp_path / "a", variants)
    attach(monkeypatch, repo)
    R.run_full(cfg_path, workers=1, raw_dir=raw)
    clean_ref = {p.name: p.read_bytes() for p in (repo / "data" / "clean").glob("*.parquet")}

    repo2, raw2, cfg2, cfg_path2 = make_repo(tmp_path / "b", variants)
    attach(monkeypatch, repo2)
    real = S.run_task

    def interrupted(config_path, variant, symbol, clean_dir, out_dir):
        if (variant, symbol) == ("harq", "BBB"):
            raise MemoryError("killed")
        return real(config_path, variant, symbol, clean_dir, out_dir)

    monkeypatch.setattr(S, "run_task", interrupted)
    with pytest.raises(SystemExit, match="1 task.*failed"):
        R.run_full(cfg_path2, workers=1, raw_dir=raw2)
    res2 = repo2 / "results"
    assert (res2 / "RUN_STARTED.lock").exists() and not (res2 / "RUN_INFO.md").exists()
    lock = run_info_lock_fields(repo2)
    assert set(lock) == {"started", "commit", "config_sha256", "prespec_sha256", "bundle_hashes"} and len(lock["bundle_hashes"]) == 7
    assert S.task_done(res2 / "forecasts", "harq", "BBB") is None and S.task_done(res2 / "forecasts", "primary", "DDD") is not None
    with pytest.raises(SystemExit, match="started before"):
        R.run_full(cfg_path2, workers=1, raw_dir=raw2)                                    # not without --resume

    # a changed code base voids the finished tasks: resume is refused
    (repo2 / "tracked.txt").write_text("a new commit")
    subprocess.run(["git", "commit", "-qam", "change"], cwd=repo2, check=True, capture_output=True)
    with pytest.raises(SystemExit, match="not the one the run was started with"):
        R.run_full(cfg_path2, workers=1, resume=True, raw_dir=raw2)
    subprocess.run(["git", "reset", "-q", "--hard", "HEAD~1"], cwd=repo2, check=True)
    # a changed configuration too
    cfg_path2.write_text(cfg2.to_json().replace('"seed": 0', '"seed": 1'))
    with pytest.raises(SystemExit, match="configuration is not the one"):
        R.run_full(cfg_path2, workers=1, resume=True, raw_dir=raw2)
    cfg_path2.write_text(cfg2.to_json())

    ran = []
    monkeypatch.setattr(S, "run_task", lambda *a: (ran.append(a[1:3]), real(*a))[1])
    R.run_full(cfg_path2, workers=1, resume=True, raw_dir=raw2)
    assert ran == [("harq", "BBB")]                                                       # only the unfinished task
    assert (res2 / "RUN_INFO.md").exists() and not (res2 / "RUN_STARTED.lock").exists()
    assert "the run was resumed" in (res2 / "RUN_INFO.md").read_text()
    assert {p.name: p.read_bytes() for p in (repo2 / "data" / "clean").glob("*.parquet")} == clean_ref
    for v in variants:
        for s in S.variant_symbols(cfg, v):
            for name in ("forecasts", "regime", "counters"):
                pd.testing.assert_frame_equal(pd.read_parquet(repo / "results" / "forecasts" / f"{v}__{s}__{name}.parquet"),
                                              pd.read_parquet(res2 / "forecasts" / f"{v}__{s}__{name}.parquet"))
    for f in (repo / "results" / "tables").glob("*.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(f), pd.read_csv(res2 / "tables" / f.name), check_exact=False, rtol=1e-9)


def test_a_data_problem_stops_the_run_before_the_lock_is_written(tmp_path, monkeypatch):
    repo, raw, cfg, cfg_path = make_repo(tmp_path, ("primary",))
    attach(monkeypatch, repo)
    cfg_path.write_text(cfg.to_json().replace('"2023-02-01"', '"2024-02-01"'))             # no bar in the sample period
    with pytest.raises(SystemExit, match="STOPPED in stage 0"):
        R.run_full(cfg_path, workers=1, raw_dir=raw)
    assert not (repo / "results" / "RUN_STARTED.lock").exists() and not list((repo / "results").glob("forecasts/*"))
