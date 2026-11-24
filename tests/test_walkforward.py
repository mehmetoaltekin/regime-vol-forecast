"""Unit tests for the regime series and the walk-forward engine."""
import numpy as np
import pandas as pd
import pytest
from hmm_regime import prepare_hmm_features

from rvf.regime import RegimeCalibration, RegimeConfig, ahead, hmm_features, prob_stress
from rvf.walkforward import WFConfig, Window, cumulative_target, refit_origins, run_walk_forward

FAST = RegimeConfig(n_init=2)
WF = WFConfig(window=150, refit_every=40)


class LastValue:
    """Toy model: h times the last target value. Records what it was shown."""
    name, fallback = "last", None

    def __init__(self):
        self.seen, self.fits = [], []

    def fit(self, win, h):
        self.fits.append((win.data.index[0], win.data.index[-1], len(win.data)))
        return True

    def predict(self, win, h):
        self.seen.append((win.data.index[-1], len(win.data), len(win.filt) if win.filt is not None else None))
        return h * win.y[-1]


class UsesRegime:
    name, fallback = "regime", None

    def fit(self, win, h):
        self.level = np.array([win.y[win.p <= 0.5].mean(), win.y[win.p > 0.5].mean()])
        return True

    def predict(self, win, h):
        return h * float(self.level @ [1 - win.pi(1)[-1][win.stress].sum(), win.pi(1)[-1][win.stress].sum()])


def test_cumulative_target():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(cumulative_target(y, 1), [2, 3, 4, 5, np.nan], equal_nan=True)
    assert np.allclose(cumulative_target(y, 3), [9, 12, np.nan, np.nan, np.nan], equal_nan=True)
    assert np.isnan(cumulative_target(y, 5)).all()


def test_features_match_the_library(daily):
    lib = daily.rename(columns={"close": "Close", "high": "High", "low": "Low"})
    for cfg, kw in ((RegimeConfig(), dict(vol_feature="bar")),
                    (RegimeConfig(vol_feature="rolling", vol_window=5), dict(vol_feature="rolling", vol_window=5))):
        ref, X = prepare_hmm_features(lib, **kw)
        mine = hmm_features(daily, cfg)
        assert mine.index.equals(ref.index)
        assert np.allclose(mine.to_numpy(), X)
    custom = hmm_features(daily, RegimeConfig(feature_cols=("log_return", "log_rv")))
    assert list(custom.columns) == ["log_return", "log_rv"] and np.allclose(custom["log_rv"], np.log(daily["rv"].iloc[1:]))


def test_features_and_filter_do_not_look_ahead(daily):
    changed = daily.copy()
    cut = 300
    changed.iloc[cut:, changed.columns.get_indexer(["open", "high", "low", "close"])] *= 1.3
    changed.iloc[cut:, changed.columns.get_loc("rv")] *= 4.0
    for cfg in (RegimeConfig(), RegimeConfig(vol_feature="rolling"), RegimeConfig(feature_cols=("log_return", "log_rv"))):
        a, b = hmm_features(daily, cfg), hmm_features(changed, cfg)
        assert a.loc[: daily.index[cut - 1]].equals(b.loc[: daily.index[cut - 1]])
    X, Xc = hmm_features(daily).to_numpy(), hmm_features(changed).to_numpy()
    cal = RegimeCalibration(FAST).fit(X[:200])
    k = cut - 1                                                              # feature rows start one day late
    assert np.array_equal(cal.filter(X)[:k], cal.filter(Xc)[:k])
    assert not np.allclose(cal.filter(X)[k:], cal.filter(Xc)[k:])


def test_filter_can_be_continued_one_day_at_a_time(daily):
    X = hmm_features(daily).to_numpy()
    cal = RegimeCalibration(FAST).fit(X[:200])
    full = cal.filter(X[:260])
    step = cal.filter(X[:200])
    for t in range(200, 260):
        step = np.vstack([step, cal.filter(X[t: t + 1], prior=step[-1])])
    assert np.allclose(full, step, atol=1e-12)
    assert np.allclose(full.sum(axis=1), 1.0)
    assert np.allclose(ahead(full, cal.A, 3), full @ cal.A @ cal.A @ cal.A)
    assert np.allclose(prob_stress(full, cal.stress), full[:, cal.stress].sum(axis=1))
    assert cal.state_var.shape == (2,) and cal.state_var[1] > cal.state_var[0]          # state 0 is the calm state
    assert RegimeCalibration(RegimeConfig(feature_cols=("log_rv", "log_return"), n_init=1)).fit(X[:200]).state_var is None


def test_regime_detects_the_planted_states(daily):
    X = hmm_features(daily).to_numpy()
    cal = RegimeCalibration(FAST).fit(X)
    p = prob_stress(cal.filter(X), cal.stress)
    truth = daily["state"].to_numpy()[1:]
    assert ((p > 0.5) == (truth == 1)).mean() > 0.85


def test_calendar_and_what_models_are_shown(daily):
    m = LastValue()
    res = run_walk_forward(daily, [m], WFConfig(window=150, refit_every=40, horizons=(1, 5)), FAST)
    data = daily.iloc[1:]                                                    # the first day has no return feature
    n = len(data)
    origins = refit_origins(n, WF)
    assert origins[0] == 149 and (np.diff(origins) == 40).all()
    fc = res.forecasts.xs("last", level="model")
    h1 = fc.xs(1, level="h")
    assert h1.index[0] == data.index[150]                                    # first forecast: day 151 of the sample
    assert h1.index[-1] == data.index[-1] and len(h1) == n - 150
    assert (h1["origin"].to_numpy() == data.index[149:-1]).all()
    assert np.allclose(h1["target"], data["rv"].iloc[150:])
    h5 = fc.xs(5, level="h")
    assert h5["target"].iloc[0] == pytest.approx(data["rv"].iloc[150:155].sum()) and h5["target"].iloc[-4:].isna().all()
    # training windows: always 150 days, ending at the refit origin
    assert [f[2] for f in m.fits] == [150] * (2 * len(origins))
    assert [f[1] for f in m.fits[::2]] == list(data.index[origins])
    # every forecast call ends at its origin and has one regime row per data row
    assert all(rows == filt for _, rows, filt in m.seen)
    assert [s[0] for s in m.seen[::2]] == list(data.index[149:-1])
    assert max(rows for _, rows, _ in m.seen) == 150 + 39
    assert res.regime["refit"].sum() == len(origins) and (res.regime["k"] == 2).all()
    assert res.counters.set_index(["model", "h"]).loc[("last", 1), "refits"] == len(origins)


def test_forecasts_do_not_depend_on_later_data(daily):
    cut = 330
    changed = daily.copy()
    changed.iloc[cut:, changed.columns.get_indexer(["open", "high", "low", "close"])] *= 1.2
    changed.iloc[cut:, changed.columns.get_loc("rv")] *= 5.0
    a = run_walk_forward(daily, [LastValue(), UsesRegime()], WF, FAST)
    b = run_walk_forward(changed, [LastValue(), UsesRegime()], WF, FAST)
    last_clean_origin = daily.index[cut - 1]
    fa = a.forecasts[a.forecasts["origin"] <= last_clean_origin]
    fb = b.forecasts[b.forecasts["origin"] <= last_clean_origin]
    assert len(fa) > 300
    pd.testing.assert_frame_equal(fa[["forecast", "raw", "origin"]], fb[["forecast", "raw", "origin"]])
    pd.testing.assert_frame_equal(a.regime.loc[:last_clean_origin], b.regime.loc[:last_clean_origin])
    later = a.forecasts["origin"] > daily.index[cut + 45]                    # after the next refit everything differs
    assert not np.allclose(a.forecasts.loc[later, "forecast"], b.forecasts.loc[later, "forecast"])


def test_vintage_is_the_probability_computed_on_the_day(daily):
    res = run_walk_forward(daily, [LastValue()], WF, FAST)
    X = hmm_features(daily).to_numpy()
    data = daily.iloc[1:]
    # second refit at row 189: calibrate on rows 40..189 and filter; the vintage of row 200 comes from that calibration
    cal = RegimeCalibration(FAST).fit(X[40:190])
    filt = cal.filter(X[40:201])
    assert res.regime.loc[data.index[200], "prob_stress"] == pytest.approx(prob_stress(filt, cal.stress)[-1], abs=1e-10)
    assert np.allclose(res.details["vintage_filt"][data.index[200]], filt[-1], atol=1e-10)


def test_insanity_filter_and_fallback_are_counted(daily):
    class Wild(LastValue):
        name = "wild"

        def predict(self, win, h):
            return 1e9 if win.data.index[-1].day % 2 else -1.0

    class NeverFits(LastValue):
        name, fallback = "never", "last"

        def fit(self, win, h):
            return False

        def predict(self, win, h):
            return None

    res = run_walk_forward(daily, [LastValue(), Wild(), NeverFits()], WF, None)        # no HMM needed
    c = res.counters.set_index("model")
    n_fc = len(res.forecasts.xs("wild", level="model"))
    assert c.loc["wild", "insane"] == n_fc and c.loc["never", "fallbacks"] == n_fc
    assert c.loc["never", "failures"] == c.loc["never", "refits"] > 0
    wild = res.forecasts.xs("wild", level="model")
    never, last = res.forecasts.xs("never", level="model"), res.forecasts.xs("last", level="model")
    assert (never["source"] == "last").all() and np.allclose(never["forecast"], last["forecast"])
    # replaced by the mean target of the training window of the last refit
    first = daily["rv"].iloc[:150]
    assert wild["forecast"].iloc[0] == pytest.approx(first.mean()) and (wild["raw"].iloc[0] in (1e9, -1.0))
    assert (wild["forecast"] <= daily["rv"].max()).all() and (wild["forecast"] > 0).all()
    off = run_walk_forward(daily, [Wild()], WFConfig(window=150, refit_every=40, insanity_filter=False), None)
    assert off.counters["insane"].sum() == 0 and set(off.forecasts["forecast"]) == {1e9, -1.0}


def test_too_little_data_is_an_error(daily):
    with pytest.raises(ValueError):
        run_walk_forward(daily.iloc[:100], [LastValue()], WF, None)
