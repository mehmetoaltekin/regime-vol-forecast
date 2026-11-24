"""
Tests for behaviours of the walk-forward engine and the regime series that no other test pins (review 2B, lane
walkforward): the insanity filter, the vintage weights, K selection by BIC, the unit of the state variances, the
meaning of the stress state, and the burn-in path of the engine. Every expected value is derived by hand, from the
library or from the planted data, never with the engine itself.
"""
import numpy as np
import pandas as pd
import pytest
from hmm_regime import MarketRegimeHMM

import rvf.models as M
import rvf.study as S
from rvf.regime import RegimeCalibration, RegimeConfig, hmm_features, prob_stress
from rvf.walkforward import WFConfig, refit_origins, run_walk_forward

from conftest import make_daily, perturb_after

FAST = RegimeConfig(n_init=2)
KW = dict(n_init=2, random_state=42, sort_by="variance:0")           # what RegimeCalibration passes to the library


class Scripted:
    """Model that returns a prepared forecast for every (origin day, horizon)."""
    name, fallback = "scripted", None

    def __init__(self, schedule):
        self.schedule = schedule

    def fit(self, win, h):
        return True

    def predict(self, win, h):
        return self.schedule[(win.data.index[-1], h)]


class Spy:
    """Model that records every window it is shown for estimation and for forecasting."""
    name, fallback = "spy", None

    def __init__(self):
        self.fits, self.preds = [], []

    @staticmethod
    def snap(win):
        return dict(index=win.data.index, vp=win.vp.copy(), vpi=win.vpi.copy(), A=win.A.copy())

    def fit(self, win, h):
        self.fits.append(self.snap(win))
        return True

    def predict(self, win, h):
        self.preds.append(self.snap(win))
        return 1.0


class LastY:
    """Model that forecasts the last target value; keeps a copy of every training window."""
    name, fallback = "last", None

    def __init__(self):
        self.trained_on = []

    def fit(self, win, h):
        self.trained_on.append(win.data.copy())
        return True

    def predict(self, win, h):
        return win.y[-1]


def window_sums(w, h):
    """h-day sums inside a window, written out day by day."""
    return np.array([sum(w[i: i + h]) for i in range(len(w) - h + 1)], dtype=float)


# ------------------------------------------------------------------------------------------------ insanity filter

def test_insanity_filter_bounds_come_from_the_latest_refit_and_the_h_day_sums():
    """PRESPEC 7: a forecast outside [min, max] of the h-day sums of the latest training window is replaced by their mean."""
    window, every, n, horizons = 20, 10, 75, (1, 5)
    y = [float(2 + (i * 7) % 5 + i // 4) for i in range(n)]                    # integer values with a trend: every window differs
    daily = pd.DataFrame({"rv": y}, index=pd.bdate_range("2020-01-01", periods=n))
    cfg = WFConfig(window=window, refit_every=every, horizons=horizons)
    origins = list(range(window - 1, n - 1))
    refits = list(refit_origins(n, cfg))
    assert refits == list(range(window - 1, n - 1, every)) and len(refits) == 6

    schedule, expect, bounds_seen = {}, {}, {}
    for t in origins:
        last_refit = max(r for r in refits if r <= t)
        for h in horizons:
            sums = window_sums(y[last_refit - window + 1: last_refit + 1], h)
            lo, hi, mean = sums.min(), sums.max(), sums.mean()
            bounds_seen.setdefault(h, set()).add((lo, hi))
            f = [lo, hi, 1.2 * hi, lo - 0.5, mean, hi + 0.5][(t + h) % 6]        # on a bound, just outside, far outside, inside
            schedule[(daily.index[t], h)] = f
            expect[(daily.index[t + 1], h)] = (f, f if lo <= f <= hi else mean)
    assert all(len(b) > 3 for b in bounds_seen.values())                          # the bounds do change from refit to refit

    res = run_walk_forward(daily, [Scripted(schedule)], cfg, None)
    fc = res.forecasts.xs("scripted", level="model")
    assert len(fc) == len(origins) * len(horizons)
    for (day, h), (raw, final) in expect.items():
        row = fc.loc[(day, h)]
        assert row["raw"] == raw and row["forecast"] == pytest.approx(final, rel=1e-12), (day, h)
    counts = res.counters.set_index(["model", "h"])["insane"]
    for h in horizons:
        replaced = sum(1 for (day, hh), (raw, final) in expect.items() if hh == h and final != raw)
        assert 5 < replaced < len(origins)
        assert counts.loc[("scripted", h)] == replaced                              # one count per replaced forecast, whatever h is


# ------------------------------------------------------------------------------------------------ vintage weights

def run_vintage(n, window, every, seed=0):
    daily = make_daily(n, seed=seed)
    spy = Spy()
    res = run_walk_forward(daily, [spy], WFConfig(window=window, refit_every=every, vintage_training=True), FAST)
    return daily, spy, res


def test_vintage_one_step_ahead_weights_are_the_filter_of_the_day_times_the_calibration_then_in_force():
    """PRESPEC 7 (check 9): vpi of each row is the filtered vector of that day times A of the calibration in force on it."""
    window, every = 60, 29
    daily, spy, res = run_vintage(220, window, every)
    X = hmm_features(daily).to_numpy()
    data = daily.iloc[1:]
    refits = list(refit_origins(len(data), WFConfig(window=window, refit_every=every)))
    A_of = {r: RegimeCalibration(FAST).fit(X[r - window + 1: r + 1]).A for r in refits}

    def A_in_force(t):
        return A_of[max(r for r in refits if r <= t)]

    assert len(spy.preds) > 40 and len(spy.fits) == 3
    pos = {d: i for i, d in enumerate(data.index)}
    for snap in spy.fits + spy.preds:
        for j, day in enumerate(snap["index"]):
            filt = res.details["vintage_filt"][day]
            assert np.allclose(snap["vpi"][j], filt @ A_in_force(pos[day]), atol=1e-12, rtol=0)
            assert snap["vp"][j] == pytest.approx(res.regime.loc[day, "prob_stress"], abs=1e-12)
    A1 = A_of[refits[0]]
    assert not np.allclose(A1, np.eye(2), atol=0.01) and not np.allclose(A1 @ A1, A1, atol=1e-3)   # the transition matters


@pytest.mark.parametrize("window, every, first_origin", [(60, 29, 146), (60, 59, 118)])
def test_vintage_forecasts_start_at_the_first_origin_from_2_window_minus_2_and_never_see_a_missing_vintage(window, every,
                                                                                                          first_origin):
    """PRESPEC 7 (check 9): every row of a training window needs its own-day probability, so origin 2w-3 is too early, 2w-2 is not."""
    daily, spy, res = run_vintage(220, window, every)
    data = daily.iloc[1:]
    origins = list(refit_origins(len(data), WFConfig(window=window, refit_every=every)))
    assert 2 * window - 3 in origins or 2 * window - 2 in origins                  # the boundary is hit by this calendar
    fc = res.forecasts
    assert fc["origin"].min() == data.index[first_origin]
    assert [s["index"][-1] for s in spy.preds][0] == data.index[first_origin]
    for snap in spy.fits + spy.preds:
        assert not np.isnan(snap["vp"]).any() and not np.isnan(snap["vpi"]).any()
        assert len(snap["index"]) <= window + every
    assert spy.fits[0]["index"][-1] == data.index[first_origin]


# ------------------------------------------------------------------------------------------------ K by BIC

def bic_by_hand(X, k):
    """-2 log-likelihood + parameters * log(n), from the library's fit of K states."""
    m = MarketRegimeHMM(n_components=k, **KW).fit(X)
    return -2 * m.log_likelihood_ + m.n_parameters_ * np.log(len(X)), m.aic_


def test_k_is_selected_by_bic_not_aic():
    """PRESPEC 10 (check 3): on one-regime Gaussian data BIC gives K = 1 where AIC would give 3."""
    X = np.random.default_rng(13).normal(0, 1, (300, 2))
    bic = [bic_by_hand(X, k)[0] for k in (1, 2, 3)]
    aic = [bic_by_hand(X, k)[1] for k in (1, 2, 3)]
    assert int(np.argmin(bic)) + 1 == 1 and int(np.argmin(aic)) + 1 == 3             # the data tell the two criteria apart
    cal = RegimeCalibration(RegimeConfig(n_components=(1, 2, 3), n_init=2)).fit(X)
    assert cal.k == 1 and cal.A.shape == (1, 1) and cal.model.n_components == 1


def test_k_is_two_on_two_planted_regimes():
    """PRESPEC 10 (check 3): with two planted Markov regimes of different variance and mean the selected K is 2."""
    rng = np.random.default_rng(3)
    s = np.zeros(400, int)
    for t in range(1, 400):
        s[t] = s[t - 1] if rng.random() < 0.97 else 1 - s[t - 1]
    X = np.column_stack([rng.normal(0, np.where(s == 0, 0.5, 2.0)), rng.normal(np.where(s == 0, -1.0, 1.0), 0.7)])
    assert int(np.argmin([bic_by_hand(X, k)[0] for k in (1, 2, 3)])) + 1 == 2
    cal = RegimeCalibration(RegimeConfig(n_components=(1, 2, 3), n_init=2)).fit(X)
    assert cal.k == 2
    assert ((prob_stress(cal.filter(X), cal.stress) > 0.5) == (s == 1)).mean() > 0.9


def test_the_reported_k_is_the_k_of_the_calibration_in_force(daily):
    """PRESPEC 10 (check 3): regime['k'] holds, for every origin, the BIC choice of the latest refit's window."""
    wf = WFConfig(window=150, refit_every=40)
    res = run_walk_forward(daily, [LastY()], wf, RegimeConfig(n_components=(1, 2, 3), n_init=2))
    X = hmm_features(daily).to_numpy()
    refits = list(refit_origins(len(X), wf))
    k_of = {r: 1 + int(np.argmin([bic_by_hand(X[r - 149: r + 1], k)[0] for k in (1, 2, 3)])) for r in refits}
    assert set(k_of.values()) == {1, 2, 3}                                          # K really changes between refits
    data = daily.iloc[1:]
    expected = [k_of[max(r for r in refits if r <= t)] for t in range(149, len(data) - 1)]
    assert list(res.regime["k"]) == expected


# ------------------------------------------------------------------------------------------------ state variance unit

def test_state_variance_is_in_percent_squared(daily):
    """PRESPEC 4, 6: state_var is the return variance of each state in percent squared, as r_cc is measured."""
    X = hmm_features(daily).to_numpy()
    cal = RegimeCalibration(FAST).fit(X)
    r, state = daily["r_cc"].to_numpy()[1:], daily["state"].to_numpy()[1:]
    truth = [r[state == 0].var(), r[state == 1].var()]                              # about 0.49 and 4.84 by construction
    p = prob_stress(cal.filter(X), cal.stress)
    classified = [r[p <= 0.5].var(), r[p > 0.5].var()]
    assert np.allclose(cal.state_var, truth, rtol=0.2) and np.allclose(cal.state_var, classified, rtol=0.2)
    assert 0.3 < cal.state_var[0] < 0.8 and 3.5 < cal.state_var[1] < 6.5


# ------------------------------------------------------------------------------------------------ stress state

def make_ratio_daily(n, seed, ratio=4.0, low=0.6):
    """Two persistent volatility regimes whose daily volatility differs by `ratio`; open/high/low/close, r_cc, rv, state."""
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        state[t] = state[t - 1] if rng.random() < (0.98 if state[t - 1] == 0 else 0.96) else 1 - state[t - 1]
    sigma = low * ratio ** state
    r = rng.normal(0, sigma)
    close = 100 * np.exp(np.cumsum(r) / 100)
    open_ = np.concatenate([[100.0], close[:-1]])
    wick = np.abs(rng.normal(0, sigma)) / 100
    rv = sigma ** 2 * rng.chisquare(40, n) / 40
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * np.exp(wick),
                         "low": np.minimum(open_, close) * np.exp(-wick), "close": close, "r_cc": r, "rv": rv, "state": state},
                        index=pd.bdate_range("2015-01-01", periods=n, name="tday"))


@pytest.mark.parametrize("variant, column", [("primary", 1), ("hmm_log_rv", 0)])
def test_the_stress_state_is_the_high_volatility_state_with_the_full_number_of_starts(variant, column):
    """PRESPEC 6: with the shipped n_init the stress state has the higher volatility feature and tracks the true regime."""
    cfg = S.StudyConfig(prespec_version="test", instruments=[])
    regime = S.variant_setup(variant, cfg)["regime"]
    assert regime.n_init == 10                                                       # the shipped number of starts
    d = make_ratio_daily(900, seed=4)
    feats = hmm_features(d, regime)
    X = feats.to_numpy()
    truth = d["state"].reindex(feats.index).to_numpy()
    for start in (0, 150, 300, 450, 600):
        w = slice(start, start + 250)
        cal = RegimeCalibration(regime).fit(X[w])
        means = cal.model.means_[:, column]
        assert list(cal.stress) == [False, True] and int(np.argmax(means)) == 1       # stress = the state with the larger feature
        p = prob_stress(cal.filter(X[w]), cal.stress)
        assert np.corrcoef(p, truth[w])[0, 1] > 0.6
        assert X[w][p > 0.5, column].mean() > X[w][p <= 0.5, column].mean()          # in the raw feature, not only in means_
        assert truth[w].min() == 0 and truth[w].max() == 1


# ------------------------------------------------------------------------------------------------ burn-in path

def test_forecasts_and_regime_do_not_depend_on_later_data_with_a_burn_in():
    """PRESPEC 7: the no-look-ahead property also holds on the burn-in path that the full run takes."""
    daily = make_daily(340, seed=7)
    cfg = WFConfig(window=200, refit_every=60, horizons=(1, 5), burn_in=22)
    cut = 282                                                  # daily row 281 = data row 259 is a refit origin
    changed = perturb_after(daily, cut)
    models = lambda: M.all_models(min_ess=20, ms_starts=1)
    a = run_walk_forward(daily, models(), cfg, FAST)
    b = run_walk_forward(changed, models(), cfg, FAST)
    last_clean = daily.index[cut - 1]
    assert a.regime.loc[last_clean, "refit"] and a.regime.index[0] == daily.index[221]
    fa, fb = (x.forecasts[x.forecasts["origin"] <= last_clean] for x in (a, b))
    assert len(fa) == 9 * 2 * 61
    pd.testing.assert_frame_equal(fa[["forecast", "raw", "source"]], fb[["forecast", "raw", "source"]])
    pd.testing.assert_frame_equal(a.regime.loc[:last_clean], b.regime.loc[:last_clean])
    after = a.forecasts["origin"] > last_clean
    for name in ("m0", "m1", "m2", "m2c", "m3", "m4", "m5", "m6", "m7"):
        x = a.forecasts[after].xs(name, level="model")["raw"]
        y = b.forecasts[after].xs(name, level="model")["raw"]
        assert not np.allclose(x, y), name
    short = run_walk_forward(daily.iloc[:cut], models(), cfg, FAST)
    pd.testing.assert_frame_equal(short.forecasts[["forecast", "raw", "source"]],
                                  fa.loc[short.forecasts.index, ["forecast", "raw", "source"]])
    pd.testing.assert_frame_equal(short.regime, a.regime.loc[short.regime.index])


def test_a_burn_in_only_supplies_lags_and_features(daily):
    """PRESPEC 7: the first training day starts after the burn-in but its HAR averages and HMM features use the burn-in days."""
    model = LastY()
    res = run_walk_forward(daily, [model], WFConfig(window=150, refit_every=60, burn_in=30), FAST)
    first = model.trained_on[0]
    assert first.index[0] == daily.index[30] and len(first) == 150
    rv = daily["rv"].to_numpy()
    assert first["har_m"].iloc[0] == pytest.approx(rv[9:31].mean(), rel=1e-12)       # 22 days ending on the first training day
    assert first["har_w"].iloc[0] == pytest.approx(rv[26:31].mean(), rel=1e-12)
    assert first["lhar_d"].iloc[0] == pytest.approx(np.log(rv[30]), rel=1e-12)
    assert res.regime.index[0] == daily.index[179]                                   # the first origin ends the first window
    # a burn-in too short for the rolling-volatility feature is refused
    with pytest.raises(ValueError, match="burn-in is too short"):
        run_walk_forward(daily, [LastY()], WFConfig(window=150, refit_every=60, burn_in=5),
                         RegimeConfig(vol_feature="rolling", vol_window=21, n_init=1))
