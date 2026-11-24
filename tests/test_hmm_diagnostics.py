"""The HMM return across an anchor gap, the fit diagnostics in the regime table and the per-refit counts of hmm_summary."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import rvf.study as S
from rvf.models import all_models
from rvf.realized import RVConfig, daily_measures, retained
from rvf.regime import RegimeCalibration, RegimeConfig, hmm_features
from rvf.walkforward import HmmFirstFitError, WFConfig, WFResult, refit_origins, run_walk_forward

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


def test_hmm_summary_compares_magnitudes_and_only_a_return_with_one_other_feature():
    """The criterion is |return separation| > |range separation|, with either sign; a table without that pair has no share."""
    sep = {"sep_log_return": [0.5] * 5, "sep_log_range_var": [-2.0] * 5}
    assert summarise(regime_table(sep=sep))["share_refits_return_split"] == 0.0        # a volatility split whose range separation is negative
    sep = {"sep_log_return": [-0.5, 0.5, -3.0, 3.0, -0.5], "sep_log_range_var": [-2.0, 2.0, 1.0, -1.0, 0.4]}
    assert summarise(regime_table(sep=sep))["share_refits_return_split"] == pytest.approx(3 / 5)     # refits 2, 3 and 4
    assert np.isnan(summarise(regime_table(sep={"sep_log_return": [1.0] * 5}))["share_refits_return_split"])    # nothing to compare it with
    swapped = regime_table(sep={"sep_volatility": [2.0] * 5, "sep_log_return": [5.0] * 5})
    assert np.isnan(summarise(swapped)["share_refits_return_split"])                   # the return is not the first feature of the HMM


# ------------------------------------------------------------------------------------------ the diagnostics through the engine

DIAG = ["n_stress", "stress_occupancy", "hmm_loglik", "sep_log_return", "sep_log_range_var"]


def m1_only():
    return [m for m in all_models(min_ess=10, ms_starts=1) if m.name == "m1"]


def fail_call(monkeypatch, name, call):
    """The `call`-th call of RegimeCalibration.<name> raises."""
    real, n = getattr(RegimeCalibration, name), {"n": 0}

    def flaky(self, *a, **k):
        n["n"] += 1
        if n["n"] == call:
            raise RuntimeError(f"{name} failed")
        return real(self, *a, **k)

    monkeypatch.setattr(RegimeCalibration, name, flaky)


def test_a_failed_refit_keeps_the_diagnostics_of_the_fit_that_stays_in_force(monkeypatch):
    """PRESPEC 6: the previous parameters are kept, so the row of a failed refit describes the previous fit, and it is not counted as a fit."""
    d = make_daily(240, seed=4)
    cfg = WFConfig(window=100, refit_every=40, horizons=(1,), burn_in=None)
    fail_call(monkeypatch, "fit", 2)
    res = run_walk_forward(d, m1_only(), cfg, FAST)
    reg = res.regime
    refits = list(reg.index[reg["refit"]])
    assert len(refits) == 4 and list(reg.index[reg["hmm_failed"]]) == [refits[1]]
    first = reg[DIAG].iloc[0]
    kept = reg[DIAG][reg.index < refits[2]]                                            # from the first fit to the next successful refit
    assert len(kept) > 40 and (kept == first).all().all()                              # the failed refit's row has the old numbers
    loglik = reg["hmm_loglik"]
    assert loglik.loc[refits[2]] != first["hmm_loglik"] and loglik.loc[refits[3]] != loglik.loc[refits[2]]   # a successful refit changes them
    row = S.hmm_summary({"X": res}).iloc[0]
    assert row["refits"] == 4 and row["hmm_failures"] == 1 and row["hmm_failure_share"] == 0.25
    fitted = reg.loc[[refits[0], refits[2], refits[3]]]
    assert row["share_refits_return_split"] == pytest.approx((fitted["sep_log_return"].abs() > fitted["sep_log_range_var"].abs()).mean())


def test_an_error_in_the_diagnostics_is_a_failed_refit_like_any_other_and_leaves_the_state_whole(monkeypatch):
    """The new fit replaces the old one only when everything about it succeeded; the filter is never half updated."""
    d = make_daily(240, seed=4)
    cfg = WFConfig(window=100, refit_every=40, horizons=(1,), burn_in=None)
    with monkeypatch.context() as m:
        fail_call(m, "fit", 2)
        by_fit = run_walk_forward(d, m1_only(), cfg, FAST)
    with monkeypatch.context() as m:
        fail_call(m, "diagnostics", 2)
        by_diagnostics = run_walk_forward(d, m1_only(), cfg, FAST)
    assert by_fit.regime["hmm_failed"].sum() == 1
    pd.testing.assert_frame_equal(by_diagnostics.regime, by_fit.regime)
    pd.testing.assert_frame_equal(by_diagnostics.forecasts, by_fit.forecasts)
    with monkeypatch.context() as m:
        fail_call(m, "diagnostics", 1)                                                 # at the first refit this is the stop of PRESPEC 6
        with pytest.raises(HmmFirstFitError, match="first HMM fit failed"):
            run_walk_forward(d, m1_only(), cfg, FAST)


def test_k_chosen_by_bic_gives_diagnostics_for_one_two_and_three_states_and_they_survive_the_parquet_round_trip(tmp_path):
    d = make_daily(420, seed=0)
    kb = RegimeConfig(n_components=(1, 2, 3), n_init=2)
    cfg = WFConfig(window=150, refit_every=40, horizons=(1,), burn_in=None)
    res = run_walk_forward(d, m1_only(), cfg, kb)
    reg = res.regime
    rows = reg[reg["refit"]]
    assert set(rows["k"]) == {1, 2, 3} and set(rows["n_stress"]) == {0, 1, 2}          # one stress state at K = 2, one or two at K = 3
    one = rows[rows["k"] == 1]
    assert (one["n_stress"] == 0).all() and (one["stress_occupancy"] == 0).all() and one["hmm_loglik"].notna().all()
    assert one[["sep_log_return", "sep_log_range_var"]].isna().all().all()             # no stress state: nothing to separate
    many = rows[rows["k"] > 1]
    assert (many["n_stress"] >= 1).all() and many[DIAG].notna().all().all() and many["stress_occupancy"].between(0, 1, inclusive="neither").all()
    feats = hmm_features(d, kb)
    X = feats.to_numpy()
    for origin in rows.index:                                                          # each refit equals a direct fit on its own window
        t = feats.index.get_loc(origin)
        cal = RegimeCalibration(kb).fit(X[t - 149: t + 1])
        direct = cal.diagnostics(X[t - 149: t + 1], cal.filter(X[t - 149: t + 1]))
        assert cal.k == rows.loc[origin, "k"]
        for c in DIAG:
            assert reg.loc[origin, c] == pytest.approx(direct[c], rel=1e-9, nan_ok=True), (origin, c)
    stamp = S.run_stamp("a" * 40, "b" * 64, "c" * 64, {})
    S.save_task(res, tmp_path, "k_bic", "AAA", 1.0, stamp)
    back, stopped = S.load_variant_results(tmp_path, "k_bic", ["AAA"], stamp["run_key"])
    pd.testing.assert_frame_equal(back["AAA"].regime, reg)                             # same columns, dtypes (integer counts, NaN separations), index
    row = S.hmm_summary(back).iloc[0]                                                  # K = 1 refits have no separation and are not in the share
    assert row["refits"] == len(rows) and row["share_refits_return_split"] == pytest.approx(
        (many["sep_log_return"].abs() > many["sep_log_range_var"].abs()).mean())


def test_the_diagnostic_columns_follow_the_features_of_the_hmm():
    d = make_daily(330, seed=7)
    cfg = WFConfig(window=100, refit_every=60, horizons=(1,), burn_in=22)
    rolling = run_walk_forward(d, m1_only(), cfg, RegimeConfig(vol_feature="rolling", vol_window=5, n_init=2)).regime
    names = ["n_stress", "stress_occupancy", "hmm_loglik", "sep_log_return", "sep_volatility"]
    assert list(rolling.columns[-5:]) == names and rolling[names].notna().all().all()
    result = run_walk_forward(d, m1_only(), cfg, RegimeConfig(vol_feature="rolling", vol_window=5, n_init=2))
    fitted = result.regime[result.regime["refit"]]
    assert S.hmm_summary({"X": result}).iloc[0]["share_refits_return_split"] == pytest.approx(
        (fitted["sep_log_return"].abs() > fitted["sep_volatility"].abs()).mean())
    result = run_walk_forward(d, m1_only(), cfg, RegimeConfig(feature_cols=("log_rv",), sort_by="mean:0", n_init=2))
    reg = result.regime
    assert list(reg.columns[-4:]) == ["n_stress", "stress_occupancy", "hmm_loglik", "sep_log_rv"] and reg.iloc[:, -4:].notna().all().all()
    assert (reg["sep_log_rv"] > 0).all()                                               # the stress state is the one with the higher log RV
    assert np.isnan(S.hmm_summary({"X": result}).iloc[0]["share_refits_return_split"])  # one feature: no return split to report


def test_the_return_of_an_anchor_day_reaches_the_hmm_through_the_engine_and_every_variant_keeps_it(monkeypatch):
    """The HMM windows of the engine are cut from hmm_features of the frame it is given, so r_cc must be in that frame."""
    d = make_daily(240, seed=4)
    d.iloc[150, d.columns.get_loc("r_cc")] /= 4                                        # a day after an anchor: measured from the anchor's close
    assert abs(100 * np.log(d["close"].iloc[150] / d["close"].iloc[149]) - d["r_cc"].iloc[150]) > 0.1
    windows, real = [], RegimeCalibration.fit
    monkeypatch.setattr(RegimeCalibration, "fit", lambda self, X: (windows.append(X.copy()), real(self, X))[1])
    cfg = WFConfig(window=100, refit_every=40, horizons=(1,), burn_in=22)
    run_walk_forward(d, m1_only(), cfg, FAST)
    X = hmm_features(d, FAST).loc[d.index[22:]].to_numpy()
    origins = list(refit_origins(len(X), cfg))
    assert len(windows) == len(origins) >= 3
    seen = 0
    for w, o in zip(windows, origins):
        assert np.array_equal(w, X[o - 99: o + 1])
        if o - 99 <= 150 - 22 <= o:
            assert w[150 - 22 - (o - 99), 0] == d["r_cc"].iloc[150] / 100
            seen += 1
    assert seen >= 2
    bundle = d.assign(**{n: d["rv"] * 1.1 for n in (*S.ALT_TARGETS, *S.RANGE_TARGETS)})
    for v in S.VARIANTS:
        assert (S.variant_daily(bundle, v)["r_cc"] == bundle["r_cc"]).all(), v
