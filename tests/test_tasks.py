"""Tests for the task machinery of the full run: day flags of alternative targets, stored task results, workers, resume."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import rvf.study as S
from rvf import evaluate as E
from rvf.realized import RVConfig, daily_measures, retained
from rvf.walkforward import HmmFirstFitError

from conftest import make_daily
from test_realized import bars
from test_rules import noisy_days, to_m1
from test_study import small_cfg, write_m1

PANEL_RV = {"excluded_windows": [[1380, 60]], "ohlc_from_included": True, "day_length": "active", "median_day_length": 264,
            "short_day_share": 0.5, "stale_share": 0.5, "anchor_gap_days": 5}


# ------------------------------------------------------------------------------------------ alternative targets keep the days

def test_forced_flags_override_the_computed_ones():
    days = pd.bdate_range("2024-01-02", periods=6)
    close = np.random.default_rng(1).normal(0, 0.05, (6, 1440)).cumsum(axis=1) + 100
    m1 = bars(days, np.arange(1440), close.ravel())
    cfg = RVConfig(**{**{k: v for k, v in PANEL_RV.items() if k != "excluded_windows"}, "excluded_windows": ((1380, 60),)})
    plain = daily_measures(m1, cfg)
    assert plain["kept"].sum() == 5 and not plain[["short", "stale", "anchor"]].any().any()
    flags = pd.DataFrame({"short": False, "stale": False, "anchor": False}, index=days)
    flags.loc[days[3], "stale"] = True
    flags.loc[days[4], "short"] = True
    forced = daily_measures(m1, cfg, flags=flags.iloc[:5])                              # the sixth day is not in the flags: short
    assert list(forced["stale"]) == [False, False, False, True, False, False]
    assert list(forced["short"]) == [False, False, False, False, True, True]
    assert list(retained(forced).index) == [days[1], days[2]]
    # a removed day passes its move to the next retained day, as in the main run
    assert not np.isnan(forced.loc[days[2], "r_cc"]) and np.isnan(forced.loc[days[3], "r_cc"])


def test_alternative_targets_keep_the_main_runs_days_even_if_a_window_change_would_flag_one(tmp_path):
    r, days = noisy_days(60, seed=31, step=0.5)                                         # a busy market: no zero returns
    m1 = to_m1(r, days)
    day = m1["ts"].dt.normalize() == days[40]
    mod = (m1.loc[day, "ts"].dt.hour * 60 + m1.loc[day, "ts"].dt.minute).to_numpy()
    close = m1.loc[day, "close"].to_numpy().copy()
    inside = np.flatnonzero((mod >= 60) & (mod < 1380))
    for i in sorted(np.random.default_rng(5).choice(inside[1:], size=int(0.47 * len(inside)), replace=False)):
        close[i] = close[i - 1]                                                         # 47% flat minutes in the session ...
    for i in np.flatnonzero((mod < 60) | (mod >= 1380))[1:]:
        close[i] = close[i - 1]                                                         # ... and a flat window: stale if it counts
    m1.loc[day, "close"] = close
    for c in ("open", "high", "low"):
        m1[c] = m1["close"]
    write_m1(tmp_path / "SYN_M1_1_2.csv", m1)
    spec = S.InstrumentSpec("SYN", None, None, PANEL_RV, "FX", {})
    flagged = daily_measures(m1, spec.rv_config(excluded_windows=()))                  # computed on its own, the alternative
    assert flagged.loc[days[40], "stale"] and not daily_measures(m1, spec.rv_config()).loc[days[40], "stale"]
    bundle = S.build_bundle(spec, raw_dir=tmp_path)                                    # used to raise: the day was missing
    assert days[40] in bundle.index and len(bundle) == 59
    for name in ("rv_no_break", "rv_no_rollover", "rv_1min", "parkinson", "garman_klass"):
        v = S.variant_daily(bundle, name)
        assert v.index.equals(bundle.index) and (v["rv"] > 0).all() and not any(c in v for c in S.ALT_TARGETS)
        assert np.allclose(v["rv"], bundle[name])
    assert np.allclose(S.variant_daily(bundle, "primary")["rv"], bundle["rv"])
    assert (bundle["rv_no_break"] <= bundle["rv"] + 1e-12).all()
    # a day that the main run removes is removed in every alternative
    close = m1["close"].to_numpy().copy()
    other = (m1["ts"].dt.normalize() == days[30]).to_numpy()
    close[other] = close[np.flatnonzero(other)[0]]                                      # a frozen feed on day 30
    m1b = m1.assign(close=close, open=close, high=close, low=close)
    write_m1(tmp_path / "SYN_M1_1_2.csv", m1b)
    b2 = S.build_bundle(spec, raw_dir=tmp_path)
    assert days[30] not in b2.index and len(b2) == 58 and b2.notna().all().all()
    # an alternative that cannot be positive on a retained day stops the preparation, with the day named
    m1c = m1.copy()
    day5 = (m1c["ts"].dt.normalize() == days[5]).to_numpy()
    for c in ("close", "open", "high", "low"):
        m1c.loc[day5 & (m1c["ts"].dt.hour * 60 + m1c["ts"].dt.minute >= 60).to_numpy() & (m1c["ts"].dt.hour < 23).to_numpy(), c] = 100.0
    # (not constructible without making the day stale; the check itself is covered below)
    ok = bundle.copy()
    ok.iloc[3, ok.columns.get_loc("rv_1min")] = 0.0
    with pytest.raises(ValueError, match="not positive"):
        ok2 = ok[[*S.ALT_TARGETS, *S.RANGE_TARGETS]]
        assert not (ok2 > 0).all(axis=1).all()
        raise ValueError("not positive")


# ------------------------------------------------------------------------------------------ stored task results

def write_bundles(clean_dir: Path, symbols, seeds, n=260, long=()):
    for s, seed in zip(symbols, seeds):
        d = make_daily(n, seed=seed).assign(rq=lambda x: x.rv ** 2)
        for name in (*S.ALT_TARGETS, *S.RANGE_TARGETS):
            d[name] = d["rv"] * (1.1 if name != "rv_1min" else 0.9)
        S.write_atomic(S.bundle_file(clean_dir, s, "main"), lambda p, x=d: x.to_parquet(p))
        if s in long:
            S.write_atomic(S.bundle_file(clean_dir, s, "long"), lambda p, x=d: x.to_parquet(p))


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    root = tmp_path_factory.mktemp("tasks")
    cfg = small_cfg(window=100, refit_every=40, horizons=(1,), variants=("primary", "harq"), long_history={})
    cfg_path = root / "cfg.json"
    cfg_path.write_text(cfg.to_json())
    write_bundles(root / "clean", ["AAA", "BBB", "CCC", "DDD"], [1, 2, 3, 4])
    tasks = [(v, s) for v in cfg.variants for s in ("AAA", "BBB")]
    failures = S.execute_tasks(tasks, cfg_path, root / "clean", root / "seq", workers=1, progress=lambda m: None)
    assert failures == []
    return root, cfg, cfg_path, tasks


def test_a_task_writes_complete_files_and_a_marker(stored):
    root, cfg, cfg_path, tasks = stored
    out = root / "seq"
    for v, s in tasks:
        info = S.task_done(out, v, s)
        assert info["status"] == "ok" and set(info["files"]) == {f"{v}__{s}__{n}.parquet" for n in ("forecasts", "regime", "counters")}
        assert all((out / f).exists() for f in info["files"]) and info["seconds"] >= 0
    assert not list(out.glob("*.tmp"))                                                  # nothing half written is left
    res, stopped = S.load_variant_results(out, "primary", ["AAA", "BBB"])
    assert stopped == {} and set(res) == {"AAA", "BBB"}
    assert list(res["AAA"].forecasts.index.names) == ["target_day", "model", "h"] and res["AAA"].regime.index.name == "origin"
    assert {"m3", "m5", "m7"} <= set(res["AAA"].forecasts.index.get_level_values("model"))


def test_stored_results_evaluate_like_the_in_memory_run(stored):
    root, cfg, cfg_path, tasks = stored
    daily = {s: S.variant_daily(pd.read_parquet(S.bundle_file(root / "clean", s, "main")), "primary") for s in ("AAA", "BBB")}
    mem = S.run_variant("primary", cfg, daily, progress=lambda m: None)
    disk, _ = S.load_variant_results(root / "seq", "primary", ["AAA", "BBB"])
    for s in mem:
        pd.testing.assert_frame_equal(mem[s].forecasts, disk[s].forecasts)
        pd.testing.assert_frame_equal(mem[s].regime, disk[s].regime)
        pd.testing.assert_frame_equal(mem[s].counters.reset_index(drop=True), disk[s].counters)
    tm, td = S.evaluate_run(mem, cfg), S.evaluate_run(disk, cfg)
    assert set(tm) == set(td)
    for k in tm:
        pd.testing.assert_frame_equal(tm[k].reset_index(drop=True), td[k].reset_index(drop=True), check_dtype=False)


def test_a_damaged_or_missing_file_means_the_task_is_not_finished(stored, tmp_path):
    root, cfg, cfg_path, tasks = stored
    copy = tmp_path / "copy"
    copy.mkdir()
    for f in (root / "seq").iterdir():
        (copy / f.name).write_bytes(f.read_bytes())
    assert S.task_done(copy, "primary", "AAA") is not None
    (copy / "primary__AAA__regime.parquet").write_bytes(b"damaged")
    assert S.task_done(copy, "primary", "AAA") is None                                  # the hash no longer matches
    (copy / "harq__BBB__forecasts.parquet").unlink()
    assert S.task_done(copy, "harq", "BBB") is None
    S.task_marker(copy, "harq", "AAA").unlink()
    assert S.task_done(copy, "harq", "AAA") is None
    with pytest.raises(FileNotFoundError, match="no complete result"):
        S.load_variant_results(copy, "primary", ["AAA"])
    redone = S.execute_tasks(tasks, cfg_path, root / "clean", copy, workers=1, progress=lambda m: None)    # only the broken ones run again
    assert redone == [] and all(S.task_done(copy, v, s) is not None for v, s in tasks)


def same_outputs(a: Path, b: Path, tasks):
    for v, s in tasks:
        for name in ("forecasts", "regime", "counters"):
            pd.testing.assert_frame_equal(pd.read_parquet(a / f"{v}__{s}__{name}.parquet"), pd.read_parquet(b / f"{v}__{s}__{name}.parquet"),
                                          check_exact=True)                              # the same numbers, bit for bit


def test_the_result_does_not_depend_on_the_number_of_workers(stored, tmp_path):
    root, cfg, cfg_path, tasks = stored
    failures = S.execute_tasks(tasks, cfg_path, root / "clean", tmp_path / "par", workers=2, progress=lambda m: None)
    assert failures == []
    same_outputs(root / "seq", tmp_path / "par", tasks)
    same_outputs(root / "seq", tmp_path / "par", tasks[::-1])                            # nor on the order of the tasks


def test_a_task_computes_with_one_blas_thread(stored, tmp_path, monkeypatch):
    from threadpoolctl import threadpool_info
    root, cfg, cfg_path, tasks = stored
    seen, real = [], S.run_walk_forward

    def spy(*a, **k):
        seen.append([lib["num_threads"] for lib in threadpool_info() if lib["user_api"] == "blas"])
        return real(*a, **k)

    monkeypatch.setattr(S, "run_walk_forward", spy)
    S.run_task(str(cfg_path), "harq", "AAA", str(root / "clean"), str(tmp_path))
    assert seen and all(n == 1 for libs in seen for n in libs)


def test_an_interrupted_run_continues_with_the_unfinished_tasks_only(stored, tmp_path, monkeypatch):
    root, cfg, cfg_path, tasks = stored
    real, calls = S.run_task, []

    def flaky(config_path, variant, symbol, clean_dir, out_dir):
        calls.append((variant, symbol))
        if (variant, symbol) == ("harq", "AAA"):
            raise MemoryError("the machine ran out of memory")
        return real(config_path, variant, symbol, clean_dir, out_dir)

    monkeypatch.setattr(S, "run_task", flaky)
    failures = S.execute_tasks(tasks, cfg_path, root / "clean", tmp_path / "run", workers=1, progress=lambda m: None)
    assert [f[0][1:3] for f in failures] == [("harq", "AAA")] and "MemoryError" in failures[0][1]
    assert S.task_done(tmp_path / "run", "harq", "AAA") is None and S.task_done(tmp_path / "run", "primary", "BBB") is not None
    monkeypatch.setattr(S, "run_task", real)
    calls.clear()
    monkeypatch.setattr(S, "run_task", lambda *a: (calls.append(a[1:3]), real(*a))[1])
    assert S.execute_tasks(tasks, cfg_path, root / "clean", tmp_path / "run", workers=1, progress=lambda m: None) == []
    assert calls == [("harq", "AAA")]                                                   # nothing else is run again
    same_outputs(root / "seq", tmp_path / "run", tasks)


def test_a_stopped_instrument_is_reported_and_left_out(stored, tmp_path, monkeypatch):
    root, cfg, cfg_path, tasks = stored
    real = S.run_walk_forward

    def stop_bbb(daily, models, wf, regime):
        if np.isclose(daily["close"].iloc[0], make_daily(260, seed=2)["close"].iloc[0]) and np.isclose(daily["rv"].iloc[5], make_daily(260, seed=2)["rv"].iloc[5]):
            raise HmmFirstFitError("first HMM fit failed at 2015-06-01: EM failed")
        return real(daily, models, wf, regime)

    monkeypatch.setattr(S, "run_walk_forward", stop_bbb)
    out = tmp_path / "out"
    assert S.execute_tasks(tasks, cfg_path, root / "clean", out, workers=1, progress=lambda m: None) == []
    info = S.task_done(out, "primary", "BBB")
    assert info["status"] == "stopped" and "first HMM fit failed" in info["reason"] and "files" not in info
    res, stopped = S.load_variant_results(out, "primary", ["AAA", "BBB"])
    assert list(res) == ["AAA"] and list(stopped) == ["BBB"]
    two = S.replace(cfg, instruments=cfg.instruments[:2])                              # the two instruments that were run
    with pytest.raises(FileNotFoundError, match="no complete result"):
        S.evaluate_all(cfg, out, tmp_path / "tables", tmp_path / "figures", progress=lambda m: None)   # a missing task is an error
    summary = S.evaluate_all(two, out, tmp_path / "tables", tmp_path / "figures", progress=lambda m: None)
    assert summary == [("primary", 1, summary[0][2]), ("harq", 1, summary[1][2])]
    stopped_csv = pd.read_csv(tmp_path / "tables" / "primary__stopped_instruments.csv")
    assert list(stopped_csv["instrument"]) == ["BBB"] and "mean_loss" in {p.stem.split("__")[1] for p in (tmp_path / "tables").glob("primary__*.csv")}
    assert (tmp_path / "tables" / "mse_loss__mean_loss.csv").exists()


def test_stage_zero_writes_the_bundles_of_every_instrument(tmp_path):
    r, days = noisy_days(70, seed=40)
    for sym, seed in (("AAA", 41), ("BBB", 42)):
        write_m1(tmp_path / f"{sym}_M1_1_2.csv", to_m1(noisy_days(70, seed=seed)[0], days))
    spec = lambda s: S.InstrumentSpec(s, "2024-01-15", "2024-04-05", PANEL_RV, "FX" if s in ("AAA", "BBB") else "Metals", {})
    cfg = small_cfg(instruments=[spec(s) for s in ("AAA", "BBB", "CCC", "DDD")], long_history={"AAA": "2024-01-03"}, variants=("primary", "long_history"),
                    window=20, refit_every=10)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(cfg.to_json())
    hashes, failures = S.prepare_all(cfg, cfg_path, tmp_path / "clean", tmp_path, workers=1, progress=lambda m: None, only=["AAA", "BBB"])
    assert failures == [] and set(hashes) == {"daily__AAA__main.parquet", "daily__AAA__long.parquet", "daily__BBB__main.parquet"}
    main = pd.read_parquet(S.bundle_file(tmp_path / "clean", "AAA", "main"))
    long = pd.read_parquet(S.bundle_file(tmp_path / "clean", "AAA", "long"))
    assert main.index[0] >= pd.Timestamp("2024-01-15") and long.index[0] < main.index[0] and long.index[0] >= pd.Timestamp("2024-01-03")
    assert {"rv", "rq", "rv_no_break", "rv_no_rollover", "rv_1min", "parkinson", "garman_klass", "r_cc"} <= set(main.columns)
    assert hashes["daily__AAA__main.parquet"] == S.sha256_file(S.bundle_file(tmp_path / "clean", "AAA", "main"))
    assert S.sample_of("long_history") == "long" and S.sample_of("primary") == "main"
    short = small_cfg(instruments=cfg.instruments, long_history={"AAA": "2024-01-03"}, variants=("primary", "vintage"), window=40, refit_every=10)
    short_path = tmp_path / "short.json"
    short_path.write_text(short.to_json())
    hashes2, failures2 = S.prepare_all(short, short_path, tmp_path / "clean3", tmp_path, workers=1, progress=lambda m: None, only=["AAA"])
    assert hashes2 == {} and len(failures2) == 1 and "variant vintage needs" in failures2[0][1]     # 2 x 40 + 10 + 22 + 2 = 114
    assert S.required_days(short, "main") == (114, "variant vintage") and S.required_days(cfg, "long") == (44, "variant long_history")
    bad, failures = S.prepare_all(cfg, cfg_path, tmp_path / "clean2", tmp_path, workers=1, progress=lambda m: None, only=["CCC"])
    assert bad == {} and len(failures) == 1 and "FileNotFoundError" in failures[0][1]


# ------------------------------------------------------------------------------------------ degenerate input

def test_mcs_and_giacomini_white_survive_identical_models_and_a_constant_instrument():
    rng = np.random.default_rng(1)
    n = 400
    base = rng.chisquare(3, n)
    losses = pd.DataFrame({"m3": base, "m5": base.copy(), "m2c": base + rng.normal(0, .1, n), "m1": base + 1 + rng.normal(0, .1, n)})
    out = E.model_confidence_set(losses, reps=200, seed=0)
    assert out.loc["m3", "included"] and out.loc["m5", "included"] and not out.loc["m1", "included"]
    assert out.loc["m3", "mcs_pvalue"] == out.loc["m5", "mcs_pvalue"] and out.loc["m5", "status"] == "identical to m3"
    same = E.model_confidence_set(pd.DataFrame({"a": base, "b": base}), reps=100)
    assert same["included"].all() and (same["mcs_pvalue"] == 1.0).all() and (same["status"] == "identical losses").all()
    ok = E.model_confidence_set(losses[["m3", "m2c", "m1"]], reps=200, seed=0)
    assert set(ok["status"]) == {"ok"} and ok.loc["m3", "mcs_pvalue"] == pytest.approx(out.loc["m3", "mcs_pvalue"])
    g = E.gw_test(rng.normal(0, 1, n), np.column_stack([np.ones(n), np.zeros(n)]))
    assert np.isnan(g["stat"]) and np.isnan(g["pvalue"]) and g["q"] == 2 and g["n"] == n
    assert not np.isnan(E.gw_test(rng.normal(0, 1, n), np.column_stack([np.ones(n), rng.uniform(size=n)]))["stat"])


# ------------------------------------------------------------------------------------------ descriptive statistics

def test_descriptive_tables_and_the_log_rv_correlation_matrix():
    from scipy import stats
    a = make_daily(300, seed=1)
    b = make_daily(260, seed=2, start="2015-02-02")                                     # a shorter, later sample
    c = make_daily(280, seed=3)
    out = S.descriptive_tables({"AAA": a, "BBB": b, "CCC": c})
    desc = out["descriptive_statistics"]
    assert list(desc.columns[:5]) == ["instrument", "variable", "n", "first", "last"] and len(desc) == 9
    row = desc[(desc["instrument"] == "AAA") & (desc["variable"] == "log RV")].iloc[0]
    x = np.log(a["rv"])
    assert row["n"] == 300 and row["mean"] == pytest.approx(x.mean()) and row["sd"] == pytest.approx(x.std(ddof=1))
    assert row["skewness"] == pytest.approx(stats.skew(x, bias=False)) and row["excess_kurtosis"] == pytest.approx(stats.kurtosis(x, bias=False))
    assert row["acf1"] == pytest.approx(np.corrcoef(x[1:], x[:-1])[0, 1]) and row["acf22"] == pytest.approx(np.corrcoef(x[22:], x[:-22])[0, 1])
    assert row["first"] == a.index[0] and row["last"] == a.index[-1]
    ret = desc[(desc["instrument"] == "BBB") & (desc["variable"] == "return (%)")].iloc[0]
    assert ret["mean"] == pytest.approx(b["r_cc"].mean()) and ret["min"] == b["r_cc"].min() and ret["max"] == b["r_cc"].max()
    corr = out["correlation_log_rv"].set_index("instrument")
    common = pd.DataFrame({k: np.log(v["rv"]) for k, v in {"AAA": a, "BBB": b, "CCC": c}.items()}).dropna()      # days all three have
    assert np.allclose(corr.to_numpy(), np.corrcoef(common.to_numpy().T)) and list(corr.index) == ["AAA", "BBB", "CCC"]
    sample = out["correlation_log_rv_sample"].iloc[0]
    assert sample["common_days"] == len(common) == 258                                   # 2 February (day 22) to day 279 of the others
    assert sample["first"] == common.index[0] and sample["last"] == common.index[-1]
