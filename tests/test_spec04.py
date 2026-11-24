"""Unit tests for the PRESPEC 0.4 additions: burn-in, HMM failure rule, effect sizes, panel validation."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import rvf.study as S
from rvf import evaluate as E
from rvf.regime import RegimeCalibration, RegimeConfig, hmm_features
from rvf.walkforward import HmmFirstFitError, WFConfig, run_walk_forward

from conftest import make_daily
from test_walkforward import LastValue

FAST = RegimeConfig(n_init=2)


class Recorder(LastValue):
    """Records every window the engine hands to the model."""
    name = "rec"

    def __init__(self):
        super().__init__()
        self.windows = []

    def fit(self, win, h):
        self.windows.append(win)
        return super().fit(win, h)


# ------------------------------------------------------------------------------------------ burn-in

def test_burn_in_days_only_supply_lags_and_features():
    daily = make_daily(330, seed=5).assign(rq=lambda d: d.rv ** 2)
    cfg = WFConfig(window=100, refit_every=30, burn_in=22)
    rec = Recorder()
    res = run_walk_forward(daily, [rec], cfg, FAST)
    first = rec.windows[0]
    assert first.data.index[0] == daily.index[22] and len(first.data) == 100         # training rows start after the burn-in
    h1 = res.forecasts.xs("rec", level="model").xs(1, level="h")
    assert h1.index[0] == daily.index[22 + 100] and h1["origin"].iloc[0] == daily.index[22 + 99]
    # the lags of the first training day reach back into the burn-in
    rv = daily["rv"]
    row = first.data.iloc[0]
    assert row["har_d"] == rv.iloc[22] and row["har_w"] == pytest.approx(rv.iloc[18:23].mean())
    assert row["har_m"] == pytest.approx(rv.iloc[1:23].mean())
    assert row["lhar_m"] == pytest.approx(np.log(rv.iloc[1:23]).mean())
    # the burn-in days never enter a training window
    assert all(w.data.index[0] >= daily.index[22] for w in rec.windows)
    # and an HMM feature that needs 21 days of history is available from the first training day
    X = hmm_features(daily, RegimeConfig(vol_feature="rolling", vol_window=21))
    assert daily.index[22] in X.index


def test_every_run_shares_its_evaluation_days_with_burn_in():
    daily = make_daily(330, seed=7).assign(rq=lambda d: d.rv ** 2)
    cfg = WFConfig(window=100, refit_every=30, burn_in=22)
    runs = {name: run_walk_forward(daily, [LastValue()], cfg, rc) for name, rc in (
        ("bar", FAST), ("rolling21", RegimeConfig(vol_feature="rolling", vol_window=21, n_init=2)),
        ("rolling5", RegimeConfig(vol_feature="rolling", vol_window=5, n_init=2)), ("k_bic", RegimeConfig(n_components=(1, 2, 3), n_init=2)),
        ("log_rv", RegimeConfig(feature_cols=("log_rv",), sort_by="mean:0", n_init=2)), ("no_hmm", None))}
    ref = runs["bar"].forecasts
    for name, r in runs.items():
        assert r.forecasts.index.equals(ref.index), name
        assert np.allclose(r.forecasts["target"], ref["target"]) and (r.forecasts["origin"] == ref["origin"]).all(), name
    # without burn-in the HMM feature decides where the sample starts: the runs no longer line up
    legacy = {n: run_walk_forward(daily, [LastValue()], WFConfig(window=100, refit_every=30), rc).forecasts.index[0][0]
              for n, rc in (("bar", FAST), ("rolling21", RegimeConfig(vol_feature="rolling", vol_window=21, n_init=2)))}
    assert legacy["bar"] != legacy["rolling21"]


def test_a_burn_in_that_is_too_short_for_the_hmm_feature_is_an_error():
    daily = make_daily(300, seed=8)
    with pytest.raises(ValueError, match="burn-in is too short"):
        run_walk_forward(daily, [LastValue()], WFConfig(window=100, refit_every=30, burn_in=5),
                         RegimeConfig(vol_feature="rolling", vol_window=21, n_init=2))


def test_the_study_configuration_uses_the_burn_in():
    cfg = S.StudyConfig("t", [])
    assert cfg.burn_in == 22 and S.variant_setup("primary", cfg)["wf"].burn_in == 22
    assert all(S.variant_setup(v, cfg)["wf"].burn_in == 22 for v in S.VARIANTS)


# ------------------------------------------------------------------------------------------ HMM failure rule

class AllWindows(LastValue):
    name = "all"

    def __init__(self):
        super().__init__()
        self.windows = []

    def predict(self, win, h):
        self.windows.append((win.data.index[-1], win.A.copy(), win.filt.copy(), len(win.data)))
        return super().predict(win, h)


def failing_fit(monkeypatch, fail_calls):
    real, calls = RegimeCalibration.fit, {"n": 0}

    def flaky(self, X):
        calls["n"] += 1
        if calls["n"] in fail_calls:
            raise RuntimeError("EM did not converge")
        return real(self, X)

    monkeypatch.setattr(RegimeCalibration, "fit", flaky)


def test_a_failed_refit_keeps_the_previous_parameters_and_the_filter_continues(monkeypatch):
    daily = make_daily(330, seed=6)
    cfg = WFConfig(window=100, refit_every=30, burn_in=22)
    failing_fit(monkeypatch, {2, 3})
    spy = AllWindows()
    res = run_walk_forward(daily, [spy], cfg, FAST)
    data, X = daily.iloc[22:], hmm_features(daily).loc[daily.index[22:]].to_numpy()
    origins = [99 + 30 * k for k in range(len(range(99, len(data) - 1, 30)))]
    reg = res.regime
    assert list(reg.index[reg["refit"]]) == [data.index[o] for o in origins]
    assert list(reg.index[reg["hmm_failed"]]) == [data.index[origins[1]], data.index[origins[2]]]
    assert reg["hmm_failed"].sum() == 2 and reg["refit"].sum() == len(origins)
    # parameters: the first calibration stays through the failed refits, a new one is fitted at the fourth
    by_day = {d: (A, f, n) for d, A, f, n in spy.windows}
    A1 = by_day[data.index[origins[0]]][0]
    for k in (1, 2):
        assert np.array_equal(by_day[data.index[origins[k]]][0], A1)
    assert not np.array_equal(by_day[data.index[origins[3]]][0], A1) or len(origins) < 4
    # the filter keeps its state: the same numbers as a single filter run with the first calibration from its window start
    cal1 = RegimeCalibration(FAST).fit(X[0:100])
    cont = cal1.filter(X[0: origins[2] + 1])
    for k in (1, 2):
        A, f, n = by_day[data.index[origins[k]]]
        assert n == 100 and len(f) == 100                                              # the models' window is the new one
        assert np.allclose(f, cont[origins[k] + 1 - 100: origins[k] + 1], atol=1e-10)
    stress = cal1.stress
    assert reg.loc[data.index[origins[2]], "prob_stress"] == pytest.approx(cont[origins[2], stress].sum(), abs=1e-10)
    assert len(res.forecasts.xs("all", level="model").xs(1, level="h")) == len(data) - 100


def test_a_failure_at_the_first_refit_stops_the_instrument(monkeypatch):
    daily = make_daily(330, seed=6)
    cfg = WFConfig(window=100, refit_every=30, burn_in=22)
    failing_fit(monkeypatch, {1})
    with pytest.raises(HmmFirstFitError, match="first HMM fit failed at"):
        run_walk_forward(daily, [LastValue()], cfg, FAST)
    failing_fit(monkeypatch, set(range(1, 100)))
    scfg = S.StudyConfig("t", [], window=100, refit_every=30, horizons=(1,), ms_starts=1, mcs_reps=50)
    data = {"AAA": make_daily(330, seed=6), "BBB": make_daily(330, seed=9)}
    stopped = {}
    out = S.run_variant("primary", scfg, data, progress=lambda m: None, stopped=stopped)
    assert out == {} and set(stopped) == {"AAA", "BBB"} and "first HMM fit failed" in stopped["AAA"]
    with pytest.raises(HmmFirstFitError):
        S.run_variant("primary", scfg, data, progress=lambda m: None)


def test_one_instrument_can_stop_while_the_others_run(monkeypatch):
    real = RegimeCalibration.fit

    def only_for_bbb(self, X):
        if len(X) and abs(X[0, 0] - bbb_first) < 1e-12:
            raise RuntimeError("fails for BBB only")
        return real(self, X)

    aaa, bbb = make_daily(330, seed=6), make_daily(330, seed=9)
    bbb_first = hmm_features(bbb).loc[bbb.index[22]].iloc[0]
    monkeypatch.setattr(RegimeCalibration, "fit", only_for_bbb)
    scfg = S.StudyConfig("t", [], window=100, refit_every=30, horizons=(1,), ms_starts=1, mcs_reps=50)
    stopped = {}
    out = S.run_variant("harq", scfg, {"AAA": aaa.assign(rq=aaa.rv ** 2), "BBB": bbb.assign(rq=bbb.rv ** 2)},
                        progress=lambda m: None, stopped=stopped)
    assert list(out) == ["AAA"] and list(stopped) == ["BBB"]


def test_hmm_failures_are_counted_per_instrument_and_flagged_above_ten_percent():
    def result(refits, failures):
        idx = pd.bdate_range("2020-01-01", periods=refits * 3)
        reg = pd.DataFrame({"prob_stress": 0.3, "k": 2, "refit": [i % 3 == 0 for i in range(len(idx))], "hmm_converged": True,
                            "hmm_warnings": 0, "hmm_failed": False}, index=idx)
        reg.loc[reg.index[reg["refit"]][:failures], "hmm_failed"] = True
        return SimpleNamespace(regime=reg)
    out = S.hmm_summary({"none": result(20, 0), "exactly10": result(20, 2), "above10": result(20, 3), "all": result(10, 10)}
                        ).set_index("instrument")
    assert list(out["refits"]) == [20, 20, 20, 10] and list(out["hmm_failures"]) == [0, 2, 3, 10]
    assert list(out["flag_failures_over_10pct"]) == [False, False, True, True]         # 10% is not more than 10%
    assert out.loc["above10", "hmm_failure_share"] == pytest.approx(0.15)


# ------------------------------------------------------------------------------------------ effect sizes

def test_effect_size_by_hand():
    ref, mod = np.array([3.0, 4.0, 5.0, 6.0]), np.array([2.0, 2.0, 2.0, 2.0])
    out = E.effect_size(ref, mod, h=1)
    d = np.array([1.0, 2.0, 3.0, 4.0])
    f = (4 + 1 - 2 + 0) / 4                                                            # HLN factor for n = 4, h = 1
    se = np.sqrt(E.long_run_variance(d, 5) / 4) / np.sqrt(f)
    crit = stats.t(3).ppf(0.975)
    assert out["mean_d"] == pytest.approx(2.5) and out["ref_mean_loss"] == pytest.approx(4.5) and out["n"] == 4 and out["lag"] == 5
    assert out["rel_improvement_pct"] == pytest.approx(100 * 2.5 / 4.5) and out["se_d"] == pytest.approx(se)
    assert out["ci_low_pct"] == pytest.approx(100 * (2.5 - crit * se) / 4.5) and out["ci_high_pct"] == pytest.approx(100 * (2.5 + crit * se) / 4.5)
    # sign: a model with the lower loss has a positive improvement; the reverse is negative
    assert E.effect_size(mod, ref)["rel_improvement_pct"] < 0
    # without a variance the interval collapses to the point estimate
    const = E.effect_size([2.0, 4.0, 2.0, 4.0, 2.0, 4.0], [1.5, 3.5, 1.5, 3.5, 1.5, 3.5])
    assert const["rel_improvement_pct"] == pytest.approx(100 * 0.5 / 3.0) and const["ci_low_pct"] == pytest.approx(const["ci_high_pct"])
    assert np.isnan(E.effect_size([1.0, 2.0], [1.0, 1.0])["mean_d"])                   # too few days
    assert E.effect_size([1.0, np.nan, 3.0, 4.0, 5.0], [0.5, 2.0, 2.0, 3.0, 4.0])["n"] == 4


def test_the_interval_excludes_zero_exactly_when_the_two_sided_test_rejects():
    rng = np.random.default_rng(11)
    for h in (1, 5):
        agree = []
        for _ in range(300):
            ref = rng.chisquare(3, 250) + 1.0
            mod = ref - rng.normal(0.05, 0.6, 250)
            mod = np.maximum(mod, 0.05)
            e = E.effect_size(ref, mod, h=h)
            p = E.dm_test(mod - ref, h=h, alternative="two-sided")["pvalue"]
            agree.append((e["ci_low_pct"] > 0 or e["ci_high_pct"] < 0) == (p < 0.05))
        assert all(agree), h
    # pooled version: normal critical values and no HLN factor
    ref = pd.DataFrame(rng.chisquare(3, (250, 3)) + 1.0)
    mod = (ref - rng.normal(0.05, 0.6, (250, 3))).clip(lower=0.05)
    e = E.pooled_effect_size(ref, mod)
    p = E.pooled_dm(mod - ref)["pvalue"]
    assert (e["ci_low_pct"] > 0 or e["ci_high_pct"] < 0) == (p < 0.05) and e["instruments_per_day"] == 3


def test_pooled_effect_size_averages_over_the_instruments_available_each_day():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(12)
    ref = pd.DataFrame(rng.uniform(1, 3, (300, 3)), index=idx, columns=list("abc"))
    mod = ref - rng.normal(0.1, 0.3, (300, 3))
    mod = mod.clip(lower=0.1)
    ref.iloc[:100, 2] = np.nan                                                         # instrument c starts later
    mod.iloc[:100, 2] = np.nan
    e = E.pooled_effect_size(ref, mod, h=1)
    ref_bar, d_bar = ref.mean(axis=1), (ref - mod).mean(axis=1)
    assert e["n"] == 300 and e["mean_d"] == pytest.approx(d_bar.mean()) and e["ref_mean_loss"] == pytest.approx(ref_bar.mean())
    assert e["rel_improvement_pct"] == pytest.approx(100 * d_bar.mean() / ref_bar.mean())
    se = np.sqrt(E.long_run_variance(d_bar.to_numpy(), 5) / 300)
    assert e["ci_high_pct"] == pytest.approx(100 * (d_bar.mean() + stats.norm.ppf(0.975) * se) / ref_bar.mean())
    assert e["instruments_per_day"] == pytest.approx((100 * 2 + 200 * 3) / 300)


def test_effect_size_tables_use_the_reference_of_each_comparison():
    cfg = S.StudyConfig("t", [], window=100, refit_every=30, horizons=(1, 5), ms_starts=1, mcs_reps=50, mcs_block=5)
    res = S.run_variant("primary", cfg, {"AAA": make_daily(330, seed=1), "BBB": make_daily(330, seed=2)}, progress=lambda m: None)
    t = S.evaluate_run(res, cfg)
    es = t["effect_sizes"]
    assert set(es["scope"]) == {"AAA", "BBB", "pooled"} and set(es["family"]) == {"H1", "H2", "M4_vs_M2c", "H3"}
    ref = es.drop_duplicates(["family", "model"]).set_index(["family", "model"])["reference"]
    assert ref[("H1", "m5")] == "m3" and ref[("H2", "m6")] == "m3" and ref[("M4_vs_M2c", "m4")] == "m2c"
    assert ref[("H3", "m5")] == "m7" and ref[("H3", "m6")] == "m7"
    losses = E.loss_table(res["AAA"].forecasts, 1)
    row = es[(es["scope"] == "AAA") & (es["family"] == "H1") & (es["h"] == 1)].iloc[0]
    direct = E.effect_size(losses["m3"], losses["m5"], h=1)
    assert row["mean_d"] == pytest.approx(direct["mean_d"]) and row["ci_low_pct"] == pytest.approx(direct["ci_low_pct"])
    assert row["rel_improvement_pct"] == pytest.approx(100 * (losses["m3"] - losses["m5"]).mean() / losses["m3"].mean())
    # the same comparison is reported as a test with the opposite sign: model minus reference
    dm = t["dm_by_instrument"]
    dm_row = dm[(dm["instrument"] == "AAA") & (dm["family"] == "H1") & (dm["h"] == 1)].iloc[0]
    assert dm_row["mean_diff"] == pytest.approx(-row["mean_d"])
    pooled = es[(es["scope"] == "pooled") & (es["family"] == "H1") & (es["h"] == 1)].iloc[0]
    both = E.pooled_effect_size(pd.DataFrame({s: E.loss_table(r.forecasts, 1)["m3"] for s, r in res.items()}),
                                pd.DataFrame({s: E.loss_table(r.forecasts, 1)["m5"] for s, r in res.items()}), h=1)
    assert pooled["rel_improvement_pct"] == pytest.approx(both["rel_improvement_pct"])
    assert "effect_sizes" not in S.evaluate_run(res, cfg, loss="mse", pooled=False, extras=False)      # section 8.6 is for QLIKE


# ------------------------------------------------------------------------------------------ panel

def test_the_panel_is_validated():
    spec = lambda s, c: S.InstrumentSpec(s, asset_class=c)
    good = [spec("XAUUSD", "Precious metals"), spec("XAGUSD", "Precious metals"), spec("EURUSD", "FX"), spec("USDJPY", "FX")]
    S.StudyConfig("t", good).validate_panel()
    for bad, message in (([*good, spec("US500", "Equity indices")], "equity indices are not part"),
                         ([spec("DE40", "Equity indices"), *good], "equity indices"),
                         (good[:3], "at least four instruments"),
                         ([spec("A", "FX"), spec("B", "FX"), spec("C", "FX"), spec("D", "FX")], "two asset classes"),
                         ([*good, spec("EURUSD", "FX")], "listed twice")):
        with pytest.raises(ValueError, match=message):
            S.StudyConfig("t", bad).validate_panel()
    with pytest.raises(ValueError, match="outside the panel"):
        S.StudyConfig("t", good, long_history={"GBPUSD": "2000-01-03"}).validate_panel()


def test_the_shipped_configuration_is_the_prespec_panel():
    cfg = S.StudyConfig.from_json(S.Path(__file__).resolve().parents[1] / "config" / "study_config.json")
    assert [i.symbol for i in cfg.instruments] == ["XAUUSD", "XAGUSD", "EURUSD", "USDJPY", "GBPUSD"]
    assert {i.asset_class for i in cfg.instruments} == {"FX", "Precious metals"}
    assert all(i.start == "2008-11-10" and i.end == "2026-10-31" for i in cfg.instruments)
    rv = cfg.instruments[0].rv_config()
    assert rv.excluded_windows == ((1380, 60),) and rv.ohlc_from_included and rv.interval_min == 5
    assert (rv.day_length, rv.median_day_length, rv.short_day_share, rv.stale_share, rv.anchor_gap_days) == ("active", 264, 0.5, 0.5, 5)
    clean = cfg.instruments[0].clean_config()
    assert (clean.bounce_sd, clean.bounce_reversal, clean.bounce_lookback_days) == (20, 0.75, 20)
    assert all(i.rv == cfg.instruments[0].rv and i.clean == cfg.instruments[0].clean for i in cfg.instruments)
    assert (cfg.window, cfg.refit_every, cfg.burn_in, cfg.horizons) == (1000, 21, 22, (1, 5, 22))
    assert (cfg.min_ess, cfg.ms_starts, cfg.mcs_reps, cfg.mcs_block, cfg.mcs_size) == (100.0, 10, 10_000, 10, 0.10)
    assert cfg.long_history == {"EURUSD": "2000-01-03", "USDJPY": "2000-01-03", "GBPUSD": "2000-01-03", "XAUUSD": "2004-06-14"}
    assert len(cfg.variants) == 16 and cfg.prespec_version == "0.4"
