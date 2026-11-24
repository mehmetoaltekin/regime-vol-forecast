"""
Tests that pin the wiring of the study pipeline (rvf.study): how the configuration reaches the MCS, which loss check 7
uses, what each robustness variant sets up, which models the figures compare, the target of the no-break variant and the
stage-0 minimum-days check. Found by mutation testing; each test fails when the behaviour it names is changed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import rvf.study as S
from rvf import evaluate as E
from rvf.regime import RegimeConfig
from rvf.walkforward import WFResult

from test_realized import brownian
from test_rules import noisy_days, to_m1
from test_study import small_cfg, write_m1
from test_tasks import PANEL_RV

ROOT = Path(__file__).resolve().parents[1]
ALL = ["m0", "m1", "m2", "m2c", "m3", "m4", "m5", "m6", "m7"]
LOG = ["m3_log", "m5_log", "m6_log", "m7_log"]
Q = ["m3_q", "m5_q"]


# ------------------------------------------------------------------------------------------------ synthetic results

def fake_result(seed, models=ALL, horizons=(1, 5), n=60) -> WFResult:
    """Forecasts, regime series and counters of one instrument, built directly (no model is fitted)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-06", periods=n + 1)
    target_days, origins = days[1:], days[:-1]
    rows, counters = [], []
    for h in horizons:
        target = h * 2.0 * rng.chisquare(5, n) / 5
        for j, m in enumerate(models):
            fc = target * np.exp(rng.normal(0.05 * j - 0.1, 0.35, n))
            rows.append(pd.DataFrame({"target_day": target_days, "model": m, "h": h, "origin": origins, "forecast": fc, "raw": fc,
                                      "target": target, "source": m}))
            counters.append(dict(model=m, h=h, refits=3, failures=0, fallbacks=0, insane=0, ess_fallback_refits=0.0,
                                 ess_fallback_regimes=0.0))
    forecasts = pd.concat(rows).set_index(["target_day", "model", "h"]).sort_index()
    regime = pd.DataFrame({"prob_stress": rng.uniform(0, 1, n), "k": 2, "refit": np.arange(n) % 20 == 0, "hmm_converged": True,
                           "hmm_warnings": 0, "hmm_failed": False}, index=pd.Index(origins, name="origin"))
    return WFResult(forecasts, regime, pd.DataFrame(counters))


def hand_losses(res: WFResult, h: int, kind: str) -> pd.DataFrame:
    """Loss of every model on every target day, from the stored target and forecast (a pivot, not rvf.evaluate)."""
    fc = res.forecasts.reset_index()
    fc = fc[fc["h"] == h]
    y = fc.pivot(index="target_day", columns="model", values="target")
    f = fc.pivot(index="target_day", columns="model", values="forecast")
    if kind == "qlike":
        return y / f - np.log(y / f) - 1.0
    return (y - f) ** 2


def spy_mcs():
    """A stand-in for arch's MCS: records its arguments and the loss matrix, and answers with p-values 1 - 0.11 j."""
    class SpyMCS:
        calls = []

        def __init__(self, losses, size, reps=1000, block_size=None, method="r", bootstrap="stationary", *, seed=None):
            self.losses = losses.copy()
            self.args = dict(size=size, reps=reps, block_size=block_size, method=method, bootstrap=bootstrap, seed=seed)
            SpyMCS.calls.append(self)

        def compute(self):
            pass

        @property
        def pvalues(self):
            cols = list(self.losses.columns)[::-1]                       # arch does not return them in column order
            return pd.DataFrame({"Pvalue": [1.0 - 0.11 * list(self.losses.columns).index(c) for c in cols]}, index=cols)

        @property
        def included(self):
            return list(self.losses.columns)[:3]
    return SpyMCS


# ------------------------------------------------------------------------------------------------ the MCS

def check_mcs_calls(spy, results, horizons, size, reps, block, seed):
    assert len(spy.calls) == len(horizons) * len(results)
    k = 0
    for h in horizons:
        for res in results.values():
            call = spy.calls[k]
            k += 1
            a = call.args
            assert a["size"] == size and a["reps"] == reps and a["block_size"] == block and a["seed"] == seed
            assert str(a["method"]).lower() == "r" and a["bootstrap"] == "stationary"      # range statistic, stationary bootstrap
            expect = hand_losses(res, h, "qlike").drop(columns="m0")                           # M1-M7 and M2c, QLIKE, this instrument
            assert list(call.losses.columns) == list(expect.columns) == ["m1", "m2", "m2c", "m3", "m4", "m5", "m6", "m7"]
            assert np.allclose(call.losses.to_numpy(), expect.to_numpy(), rtol=1e-10)


def test_the_mcs_gets_size_replications_block_length_and_seed_from_the_configuration(monkeypatch):
    """PRESPEC 8.4: evaluate_run hands cfg.mcs_size, mcs_reps, mcs_block and seed, the range statistic and the stationary bootstrap to arch's MCS."""
    spy = spy_mcs()
    monkeypatch.setattr(E, "MCS", spy)
    cfg = small_cfg(mcs_size=0.25, mcs_reps=37, mcs_block=7, seed=5)
    results = {"AAA": fake_result(1), "BBB": fake_result(2)}
    tables = S.evaluate_run(results, cfg)
    check_mcs_calls(spy, results, (1, 5), 0.25, 37, 7, 5)
    mcs = tables["mcs"]
    models = ["m1", "m2", "m2c", "m3", "m4", "m5", "m6", "m7"]
    assert list(mcs["h"]) == [1] * 16 + [5] * 16 and list(mcs["instrument"]) == (["AAA"] * 8 + ["BBB"] * 8) * 2
    assert list(mcs["model"]) == models * 4 and set(mcs["status"]) == {"ok"}
    assert np.allclose(mcs["mcs_pvalue"], [1.0 - 0.11 * j for j in range(8)] * 4)         # each model gets its own p-value
    assert list(mcs["included"]) == ([True] * 3 + [False] * 5) * 4
    spy.calls.clear()
    S.evaluate_run(results, cfg, extras=False)
    assert spy.calls == []                                                                  # the check 7 / mse runs do not run an MCS


def test_the_shipped_configuration_reaches_the_mcs_as_prespecified(monkeypatch):
    """PRESPEC 8.4: the shipped configuration runs the MCS at the 90% level with 10,000 replications and an average block of 10."""
    spy = spy_mcs()
    monkeypatch.setattr(E, "MCS", spy)
    cfg = S.StudyConfig.from_json(ROOT / "config" / "study_config.json")
    results = {"AAA": fake_result(3, horizons=(1,)), "BBB": fake_result(4, horizons=(1,))}
    S.evaluate_run(results, cfg, pooled=False, effects=False)
    check_mcs_calls(spy, results, (1,), 0.10, 10_000, 10, cfg.seed)


# ------------------------------------------------------------------------------------------------ stored runs

VARIANT_MODELS = {"primary": ALL, "log_har": LOG, "harq": Q, "window_750": ALL}


@pytest.fixture(scope="module")
def stored_runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("stored")
    cfg = small_cfg(variants=tuple(VARIANT_MODELS), horizons=(1, 5), mcs_reps=20)
    cfg = S.replace(cfg, instruments=cfg.instruments[:2])
    results = {}
    for v, models in VARIANT_MODELS.items():
        for sym, seed in (("AAA", 11), ("BBB", 12)):
            results[v, sym] = fake_result(seed + len(v), models)
            S.save_task(results[v, sym], out, v, sym, 1.0)
    return cfg, out, results


def test_robustness_check_7_is_the_primary_forecasts_under_mse_per_instrument_only(stored_runs, tmp_path, monkeypatch):
    """PRESPEC 10.7: mse_loss tables hold the primary forecasts scored with MSE, per instrument, with no pooled test, MCS or other extras."""
    cfg, out, results = stored_runs
    spy = spy_mcs()
    monkeypatch.setattr(E, "MCS", spy)
    monkeypatch.setattr(S, "make_figures", lambda *a, **k: [])
    S.evaluate_all(cfg, out, tmp_path, tmp_path / "f", progress=lambda m: None)
    assert sorted(p.name for p in tmp_path.glob("mse_loss__*.csv")) == [
        "mse_loss__counters.csv", "mse_loss__dm_by_instrument.csv", "mse_loss__mean_loss.csv", "mse_loss__regime_summary.csv"]
    assert len(spy.calls) == len(VARIANT_MODELS) * 2 * 2                                    # variants x instruments x horizons: none from the mse run
    mse = pd.read_csv(tmp_path / "mse_loss__mean_loss.csv")
    assert (mse["loss"] == "mse").all() and (mse["mean_other"] == mse["mean_loss"]).all()
    assert set(mse["model"]) == set(ALL) and set(mse["h"]) == {1, 5} and set(mse["instrument"]) == {"AAA", "BBB"}   # from the primary run
    for row in mse.itertuples():
        hand = hand_losses(results["primary", row.instrument], row.h, "mse")
        assert row.mean_loss == pytest.approx(hand[row.model].mean(), rel=1e-6) and row.n == len(hand)
    dm = pd.read_csv(tmp_path / "mse_loss__dm_by_instrument.csv")
    for row in dm[(dm["h"] == 1) & (dm["family"] == "H1")].itertuples():
        hand = hand_losses(results["primary", row.instrument], 1, "mse")
        assert row.mean_diff == pytest.approx((hand["m5"] - hand["m3"]).mean(), rel=1e-6)
    # the primary tables themselves are scored with QLIKE and report the MSE next to it
    prim = pd.read_csv(tmp_path / "primary__mean_loss.csv")
    assert (prim["loss"] == "qlike").all()
    for row in prim.itertuples():
        assert row.mean_loss == pytest.approx(hand_losses(results["primary", row.instrument], row.h, "qlike")[row.model].mean(), rel=1e-6)
        assert row.mean_other == pytest.approx(hand_losses(results["primary", row.instrument], row.h, "mse")[row.model].mean(), rel=1e-6)
    assert (tmp_path / "primary__dm_pooled.csv").exists() and (tmp_path / "primary__mcs.csv").exists()
    for v in ("log_har", "harq", "window_750"):                                             # no other variant writes MSE tables
        assert not list(tmp_path.glob(f"{v}__mse*")) and not (tmp_path / f"mse_loss__{v}.csv").exists()


def test_the_figures_compare_the_headline_pair_of_models_of_each_variant(stored_runs, tmp_path, monkeypatch):
    """Each variant's cumulative-loss figure compares the headline pair of its own models: m5 vs m3, in logs for log_har and HARQ for harq."""
    cfg, out, results = stored_runs
    monkeypatch.setattr(E, "MCS", spy_mcs())
    seen = {}
    monkeypatch.setattr(S, "make_figures", lambda res, outdir, prefix, a="m5", b="m3": seen.setdefault(prefix, (a, b)) and [])
    S.evaluate_all(cfg, out, tmp_path, tmp_path / "f", progress=lambda m: None)
    assert seen == {"primary": ("m5", "m3"), "log_har": ("m5_log", "m3_log"), "harq": ("m5_q", "m3_q"), "window_750": ("m5", "m3")}
    for v, (a, b) in seen.items():                                                          # both models exist in the variant's forecasts
        names = set(results[v, "AAA"].forecasts.index.get_level_values("model"))
        assert {a, b} <= names and {a, b} <= {m.name for m in S.variant_setup(v, cfg)["factory"]()}
    assert S.FIGURE_PAIRS == {"log_har": ("m5_log", "m3_log"), "harq": ("m5_q", "m3_q")}


# ------------------------------------------------------------------------------------------------ variant_setup

def names(setup) -> list[str]:
    return [m.name for m in setup["factory"]()]


def test_variant_setup_matches_the_prespec_for_every_variant():
    """PRESPEC 6 and 10: each variant's HMM feature, K candidates, training window, vintage flag, model set and hypothesis families."""
    cfg = small_cfg(window=260, refit_every=17, burn_in=11, horizons=(1, 3), alt_windows=(130, 390), min_ess=7.5, ms_starts=3)
    setup = {v: S.variant_setup(v, cfg) for v in S.VARIANTS}
    default = RegimeConfig()
    assert (default.n_components, default.vol_feature, default.feature_cols, default.sort_by) == (2, "bar", None, "variance:0")
    special = {"hmm_rolling", "hmm_5bar", "k_bic", "hmm_log_rv"}
    for v, s in setup.items():
        if v not in special:
            assert s["regime"] == default, v                                                # K = 2, per-bar log range, sorted by variance
        wf = s["wf"]
        assert (wf.refit_every, wf.horizons, wf.burn_in) == (17, (1, 3), 11), v            # from the configuration, in every variant
        assert wf.window == {"window_750": 130, "window_1500": 390}.get(v, 260), v          # alt_windows[0] and [1]
        assert wf.vintage_training == (v == "vintage"), v
    assert (setup["hmm_rolling"]["regime"].vol_feature, setup["hmm_rolling"]["regime"].vol_window) == ("rolling", 21)
    assert (setup["hmm_5bar"]["regime"].vol_feature, setup["hmm_5bar"]["regime"].vol_window) == ("rolling", 5)
    for v in ("hmm_rolling", "hmm_5bar", "k_bic"):
        assert setup[v]["regime"].feature_cols is None and setup[v]["regime"].sort_by == "variance:0"
    assert setup["hmm_rolling"]["regime"].n_components == setup["hmm_5bar"]["regime"].n_components == 2
    assert setup["k_bic"]["regime"].n_components == (1, 2, 3) and setup["k_bic"]["regime"].vol_feature == "bar"
    log_rv = setup["hmm_log_rv"]["regime"]
    assert (log_rv.feature_cols, log_rv.sort_by, log_rv.n_components) == (("log_rv",), "mean:0", 2)    # states ordered by the mean of log RV
    # models
    base = ["m0", "m1", "m2", "m2c", "m3", "m4", "m5", "m6", "m7"]
    expect = {v: base for v in S.VARIANTS}
    expect.update(hmm_log_rv=base[1:], log_har=LOG, harq=Q, vintage=["m3", "m2c", "m4", "m5", "m6"])
    for v, s in setup.items():
        assert names(s) == expect[v], v
        assert s["factory"]()[0] is not s["factory"]()[0], v                                # new model objects for every run
    # families
    prim = setup["primary"]["families"]
    assert set(prim) == {"H1", "H2", "M4_vs_M2c", "H3"}
    assert all(setup[v]["families"] == prim for v in S.VARIANTS if v not in ("log_har", "harq", "vintage"))
    assert set(setup["vintage"]["families"]) == {"H1", "H2", "M4_vs_M2c"}                    # no MS-HAR in the vintage run
    assert all(setup["vintage"]["families"][k] == prim[k] for k in ("H1", "H2", "M4_vs_M2c"))
    assert set(setup["log_har"]["families"]) == {"H1", "H2", "H3"} and set(setup["harq"]["families"]) == {"HARQ"}
    assert setup["log_har"]["families"]["H1"] == [("m5_log", "m3_log", "less")]
    assert setup["harq"]["families"]["HARQ"] == [("m5_q", "m3_q", "less")]


def test_min_ess_and_ms_starts_of_the_configuration_reach_the_model_objects():
    """PRESPEC 5: the minimum effective sample size of M6 and the random starts of M7 come from the configuration in every variant that has them."""
    cfg = small_cfg(min_ess=7.5, ms_starts=3)
    for v, m6, m7 in (("primary", "m6", "m7"), ("log_har", "m6_log", "m7_log"), ("hmm_log_rv", "m6", "m7"), ("vintage", "m6", None),
                      ("window_750", "m6", "m7"), ("k_bic", "m6", "m7")):
        models = {m.name: m for m in S.variant_setup(v, cfg)["factory"]()}
        assert models[m6].min_ess == 7.5, v
        if m7:
            assert models[m7].n_starts == 3, v
    other = small_cfg(min_ess=55.0, ms_starts=2)
    assert {m.name: m for m in S.variant_setup("vintage", other)["factory"]()}["m6"].min_ess == 55.0


def test_run_variant_runs_the_walk_forward_with_the_variant_setup(monkeypatch):
    """A variant run hands its own walk-forward configuration, regime configuration and model set to the walk-forward, once per instrument."""
    cfg = small_cfg(window=77, refit_every=13, burn_in=9, alt_windows=(50, 99), min_ess=6.0)
    got = []
    monkeypatch.setattr(S, "run_walk_forward", lambda d, models, wf, regime: got.append((len(d), [m.name for m in models], wf, regime)))
    daily = {"AAA": pd.DataFrame(index=range(5)), "BBB": pd.DataFrame(index=range(7))}
    for v in ("hmm_5bar", "window_1500", "vintage", "k_bic"):
        got.clear()
        S.run_variant(v, cfg, daily, progress=lambda m: None)
        setup = S.variant_setup(v, cfg)
        assert [g[0] for g in got] == [5, 7] and all(g[2] == setup["wf"] and g[3] == setup["regime"] for g in got)
        assert all(g[1] == names(setup) for g in got)
    assert got[0][3].n_components == (1, 2, 3) and got[0][2].window == 77
    got.clear()
    S.run_variant("window_1500", cfg, daily, progress=lambda m: None)
    assert got[0][2].window == 99 and got[0][2].refit_every == 13


# ------------------------------------------------------------------------------------------------ the no-break target

def test_rv_no_break_target_is_the_intraday_variance_without_the_session_break_return(tmp_path):
    """PRESPEC 10.1: the no-break target drops the squared session-break return, so on every day it is below the primary RV by exactly that square."""
    m1, days = brownian(60, 1.0, seed=21)
    write_m1(tmp_path / "SYN_M1_1_2.csv", m1)
    spec = S.InstrumentSpec("SYN", rv={"excluded_windows": [[1380, 1440]], "median_day_length": 287.0})
    base = S.prepare_daily(spec, raw_dir=tmp_path)
    nb = S.prepare_daily(spec, "rv_no_break", raw_dir=tmp_path)
    assert len(base) == 59 and nb.index.equals(base.index) and (nb["rv"] < base["rv"]).all()
    # independent from the minute bars: bar HH:MM closes at HH:MM + 1, the 5-minute grid of a day covers minutes 5..1380 (23:00 is excluded)
    close = m1["close"].round(6).to_numpy().reshape(60, 1440)                              # the file keeps six decimals
    grid = np.log(close[:, 4:1380:5])                                                      # prices at 00:05, 00:10, ..., 23:00
    intraday = (100.0 * np.diff(grid, axis=1)) ** 2
    break_ret = 100.0 * ((np.log(close[1:, 1439]) - np.log(close[:-1, 1439])) - (grid[1:, -1] - grid[1:, 0]))
    assert np.allclose(nb["rv"], intraday.sum(axis=1)[1:])
    assert np.allclose(base["rv"] - nb["rv"], break_ret ** 2) and (break_ret ** 2 > 0).all()
    assert np.allclose(base["rv"], intraday.sum(axis=1)[1:] + break_ret ** 2)
    # the break only matters for the target: returns and the other columns are the primary ones
    assert np.allclose(nb["r_cc"], base["r_cc"]) and np.allclose(nb["parkinson"], base["parkinson"])


# ------------------------------------------------------------------------------------------------ stage 0 minimum days

def test_stage_zero_accepts_exactly_the_required_days_and_refuses_one_fewer(tmp_path):
    """A bundle with exactly the required number of retained days is written; with one day fewer stage 0 refuses before the lock."""
    r, days = noisy_days(70, seed=40)
    write_m1(tmp_path / "AAA_M1_1_2.csv", to_m1(r, days))
    panel = [S.InstrumentSpec(s, "2024-01-15", "2024-04-05", PANEL_RV, c, {}) for s, c in (("AAA", "FX"), ("BBB", "FX"), ("CCC", "Metals"), ("DDD", "Metals"))]
    spec = panel[0]
    n = len(S.build_bundle(spec, raw_dir=tmp_path))
    assert n == 59
    for window, need, ok in ((n - 24, n, True), (n - 23, n + 1, False)):                  # need = burn-in 22 + window + 2
        cfg = small_cfg(instruments=panel, variants=("primary",), window=window, refit_every=10, burn_in=22)
        assert S.required_days(cfg, "main") == (need, "variant primary")
        path = tmp_path / f"cfg{window}.json"
        path.write_text(cfg.to_json())
        clean = tmp_path / f"clean{window}"
        if ok:
            hashes = S.prepare_instrument(str(path), "AAA", str(clean), str(tmp_path))
            assert list(hashes) == ["daily__AAA__main.parquet"] and len(pd.read_parquet(S.bundle_file(clean, "AAA", "main"))) == n
        else:
            with pytest.raises(ValueError, match=rf"{n} retained days, but variant primary needs {n + 1}"):
                S.prepare_instrument(str(path), "AAA", str(clean), str(tmp_path))
            assert not clean.exists()                                                       # nothing written for a refused instrument


def test_required_days_of_the_shipped_configuration():
    """PRESPEC 7 and 10.9: 22 burn-in days plus the window plus 2, and for the vintage run two windows plus one refit interval on top."""
    cfg = S.StudyConfig.from_json(ROOT / "config" / "study_config.json")
    assert S.required_days(cfg, "main") == (22 + 2 * 1000 + 21 + 2, "variant vintage")
    assert S.required_days(cfg, "long") == (22 + 1000 + 2, "variant long_history")
    no_vintage = S.replace(cfg, variants=tuple(v for v in cfg.variants if v != "vintage"))
    assert S.required_days(no_vintage, "main") == (22 + 1500 + 2, "variant window_1500")
