"""The HMM return across an anchor gap, the fit diagnostics in the regime table and the per-refit counts of hmm_summary."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import rvf.study as S
from rvf.models import all_models
from rvf.realized import RVConfig, daily_measures, retained
from rvf.regime import RegimeCalibration, RegimeConfig, hmm_features
from rvf.walkforward import WFConfig, WFResult, run_walk_forward

from conftest import make_daily
from test_rules import PANEL, day_bars, series

FAST = RegimeConfig(n_init=2)


# ------------------------------------------------------------------------------------------ the HMM return across an anchor gap

def anchor_plan():
    """Retained days 5, 10 and 19 January 2024 and 26 January 2024; 18 and 25 January are anchors (see test_rules)."""
    d = pd.Timestamp
    full = lambda day, level: day_bars(day, np.arange(60, 1380), np.full(1320, level) + 0.5 * (np.arange(1320) % 2))
    thin = lambda day: day_bars(day, np.array([60, 61]), np.array([200.0, 201.0]))
    plan = [("2024-01-04", full, 100), ("2024-01-05", full, 101), ("2024-01-08", thin, None), ("2024-01-09", thin, None),
            ("2024-01-10", full, 102), ("2024-01-11", thin, None), ("2024-01-12", thin, None), ("2024-01-15", thin, None),
            ("2024-01-18", full, 103), ("2024-01-19", full, 104), ("2024-01-22", thin, None), ("2024-01-23", thin, None),
            ("2024-01-24", thin, None), ("2024-01-25", full, 105), ("2024-01-26", full, 106)]
    return series([fn(d(day), lvl) if lvl is not None else fn(d(day)) for day, fn, lvl in plan])


def test_the_hmm_return_is_measured_from_the_anchor_like_r_cc_and_not_across_the_gap():
    """PRESPEC 4 and 12.6: the return of the day after an anchor starts at the anchor's close, for the HMM as for RV, EWMA and GARCH."""
    out = daily_measures(anchor_plan(), RVConfig(**PANEL))
    daily = retained(out)
    assert list(daily.index) == [pd.Timestamp(x) for x in ("2024-01-05", "2024-01-10", "2024-01-19", "2024-01-26")]
    feats = hmm_features(daily, RegimeConfig(feature_cols=("log_return", "log_range_var")))
    assert list(feats.index) == list(daily.index[1:])                                  # the first row has no previous close
    assert np.allclose(100 * feats["log_return"], daily["r_cc"].iloc[1:])
    across_the_gap = 100 * np.log(daily["close"].loc["2024-01-19"] / daily["close"].loc["2024-01-10"])
    assert abs(across_the_gap - daily["r_cc"].loc["2024-01-19"]) > 0.5                 # the two readings do differ here
    assert 100 * feats.loc["2024-01-19", "log_return"] == pytest.approx(daily["r_cc"].loc["2024-01-19"])
    anchor_return = 100 * np.log(out.loc["2024-01-19", "close"] / out.loc["2024-01-18", "close"])
    assert 100 * feats.loc["2024-01-19", "log_return"] == pytest.approx(anchor_return)


def test_the_hmm_return_equals_the_close_to_close_return_without_an_anchor_and_without_r_cc():
    d = make_daily(80, seed=3)
    f = hmm_features(d)
    expected = np.log(d["close"] / d["close"].shift(1)).iloc[1:]
    assert np.allclose(f["log_return"], expected) and f.index[0] == d.index[1]
    assert np.allclose(hmm_features(d.drop(columns="r_cc"))["log_return"], expected)      # a frame without r_cc: close to close


# ------------------------------------------------------------------------------------------ the diagnostics of one fit

def stub_calibration(means, stress, ll=-10.0, columns=("log_return", "log_range_var")):
    cal = RegimeCalibration(RegimeConfig(feature_cols=columns))
    cal.model = SimpleNamespace(means_=np.array(means, dtype=float), stress_states_=np.array(stress, dtype=bool), log_likelihood_=ll)
    return cal


def test_the_diagnostics_are_occupancy_weighted_mean_differences_in_standard_deviations():
    X = np.array([[0.00, 1.0], [0.02, 3.0], [-0.02, 2.0], [0.01, 6.0], [0.03, 4.0], [0.00, 2.0]])
    filt = np.array([[0.9, 0.1, 0.0], [0.0, 0.5, 0.5], [0.6, 0.2, 0.2], [0.0, 0.0, 1.0], [0.1, 0.1, 0.8], [1.0, 0.0, 0.0]])
    means = [[0.0, 1.0], [0.01, 3.0], [0.04, 5.0]]
    cal = stub_calibration(means, [False, True, True], ll=-123.5)
    out = cal.diagnostics(X, filt)
    occ = filt.mean(axis=0)                                                            # (0.433.., 0.15, 0.416..)
    hot_w, cold_w = occ[1] + occ[2], occ[0]
    expected = {}
    for j, name in enumerate(("log_return", "log_range_var")):
        hot = (occ[1] * means[1][j] + occ[2] * means[2][j]) / hot_w
        cold = means[0][j]
        expected[f"sep_{name}"] = (hot - cold) / X[:, j].std(ddof=1)
    assert out["n_stress"] == 2 and isinstance(out["n_stress"], int)
    assert out["stress_occupancy"] == pytest.approx(hot_w) and out["hmm_loglik"] == -123.5
    assert out["sep_log_return"] == pytest.approx(expected["sep_log_return"]) and out["sep_log_range_var"] == pytest.approx(expected["sep_log_range_var"])
    assert list(out) == ["n_stress", "stress_occupancy", "hmm_loglik", "sep_log_return", "sep_log_range_var"]
    assert out["sep_log_range_var"] > 0 and abs(out["sep_log_range_var"]) > abs(out["sep_log_return"])   # a volatility split
    other = stub_calibration(means, [True, False, False]).diagnostics(X, filt)           # two calm states of unequal weight
    hot_w, cold_w = occ[0], occ[1] + occ[2]
    for j, name in enumerate(("log_return", "log_range_var")):
        cold = (occ[1] * means[1][j] + occ[2] * means[2][j]) / cold_w
        assert other[f"sep_{name}"] == pytest.approx((means[0][j] - cold) / X[:, j].std(ddof=1))
    assert other["n_stress"] == 1 and other["stress_occupancy"] == pytest.approx(hot_w)


def test_the_diagnostics_have_no_separation_when_there_is_no_stress_state_or_no_calm_state():
    X = np.random.default_rng(0).normal(size=(20, 2))
    filt = np.tile([0.5, 0.5], (20, 1))
    for stress in ([False, False], [True, True]):
        out = stub_calibration([[0, 0], [1, 1]], stress).diagnostics(X, filt)
        assert out["n_stress"] == sum(stress) and np.isnan(out["sep_log_return"]) and np.isnan(out["sep_log_range_var"])
    out = stub_calibration([[0, 0], [1, 1]], [False, True]).diagnostics(X, np.tile([1.0, 0.0], (20, 1)))   # the filter never visits the stress state
    assert out["stress_occupancy"] == 0 and np.isnan(out["sep_log_return"])
    assert np.isnan(stub_calibration([[0, 0], [1, 1]], [False, True]).diagnostics(np.zeros((20, 2)), filt)["sep_log_return"])   # a constant feature


def test_the_regime_table_carries_the_diagnostics_of_the_calibration_in_force():
    """Every origin row carries the diagnostics of the fit in force; they change at a refit and are those of a direct fit on the window."""
    d = make_daily(240, seed=4)
    cfg = WFConfig(window=100, refit_every=40, horizons=(1,), burn_in=None)
    res = run_walk_forward(d, [m for m in all_models(min_ess=10, ms_starts=1) if m.name in ("m1", "m3")], cfg, FAST)
    reg = res.regime
    cols = ["n_stress", "stress_occupancy", "hmm_loglik", "sep_log_return", "sep_log_range_var"]
    assert list(reg.columns[-5:]) == cols and reg[cols].notna().all().all()
    block = reg.groupby(reg["refit"].cumsum())[cols]
    assert (block.nunique() == 1).all().all()                                          # constant between refits, new at a refit
    assert reg[cols].drop_duplicates().shape[0] == reg["refit"].sum()
    feats = hmm_features(d, FAST)
    X_all = feats.to_numpy()
    for origin in reg.index[reg["refit"]]:                                              # each refit equals a direct fit on its own window
        t = feats.index.get_loc(origin)
        X = X_all[t - 99: t + 1]
        cal = RegimeCalibration(FAST).fit(X)
        direct = cal.diagnostics(X, cal.filter(X))
        for c in cols:
            assert reg.loc[origin, c] == pytest.approx(direct[c], rel=1e-9), (origin, c)
    first = reg.iloc[0]                                                                # planted volatility regimes: the first fit is a split on the range
    assert first["n_stress"] == 1 and first["sep_log_range_var"] > abs(first["sep_log_return"]) > 0


# ------------------------------------------------------------------------------------------ hmm_summary counts refits

def regime_table(n=100, every=20, not_converged=(), warnings=None, failed=(), sep=None):
    """A regime table of `n` origins with a refit every `every` days; the flags of a refit stay on its origins until the next one."""
    idx = pd.bdate_range("2020-01-01", periods=n, name="origin")
    refit = np.arange(n) % every == 0
    block = np.cumsum(refit) - 1
    g = pd.DataFrame({"prob_stress": 0.3, "k": 2, "refit": refit, "hmm_converged": ~np.isin(block, list(not_converged)),
                      "hmm_warnings": [(warnings or {}).get(int(b), 0) for b in block],
                      "hmm_failed": refit & np.isin(block, list(failed))}, index=idx)
    for name, values in (sep or {}).items():
        g[name] = [values[int(b)] for b in block]
    return g


def summarise(g):
    return S.hmm_summary({"X": WFResult(pd.DataFrame(), g, pd.DataFrame())}).iloc[0]


def test_hmm_summary_counts_refits_and_not_origin_days():
    """One refit that did not converge, with two warnings, is one refit and two warnings, however many origins it served."""
    row = summarise(regime_table(not_converged=[2], warnings={2: 2}))
    assert row["refits"] == 5 and row["origins"] == 100 and row["hmm_not_converged"] == 1 and row["hmm_warnings"] == 2
    row = summarise(regime_table(not_converged=[0, 1, 2, 3, 4], warnings={0: 1, 3: 4}))
    assert row["hmm_not_converged"] == 5 and row["hmm_warnings"] == 5                  # at most one per refit


def test_hmm_summary_does_not_count_a_failed_refit_among_the_fits():
    """A failed refit keeps the previous calibration, whose flags it carries; it is a failure, not a second non-converged fit."""
    g = regime_table(not_converged=[1, 2], warnings={1: 3, 2: 7}, failed=[2])
    row = summarise(g)
    assert row["hmm_failures"] == 1 and row["hmm_not_converged"] == 1 and row["hmm_warnings"] == 3
    assert row["hmm_failure_share"] == pytest.approx(1 / 5)


def test_hmm_summary_reports_the_share_of_refits_that_split_on_the_return():
    sep = {"sep_log_return": [0.1, 2.0, -3.0, 0.5, 1.0], "sep_log_range_var": [1.5, 0.1, 1.0, 0.5, 0.2]}
    row = summarise(regime_table(sep=sep))
    assert row["share_refits_return_split"] == pytest.approx(3 / 5)                    # |return| > |range| in refits 1, 2 and 4 (0.5 is not > 0.5)
    sep["sep_log_range_var"][2] = np.nan                                               # a refit without a separation is left out
    assert summarise(regime_table(sep=sep))["share_refits_return_split"] == pytest.approx(2 / 4)
    assert np.isnan(summarise(regime_table())["share_refits_return_split"])          # no diagnostics in the table
    g = regime_table(sep={"sep_log_return": [0.0] * 5, "sep_volatility": [1.0] * 5})
    assert summarise(g)["share_refits_return_split"] == 0.0
    g = regime_table(sep={"sep_log_rv": [5.0] * 5})
    assert np.isnan(summarise(g)["share_refits_return_split"])                         # a single feature has nothing to compare with
    g = regime_table(failed=[1], sep={"sep_log_return": [2.0] * 5, "sep_log_range_var": [1.0, 9.0, 1.0, 1.0, 1.0]})
    assert summarise(g)["share_refits_return_split"] == 1.0                            # the failed refit carries an old fit and is left out
