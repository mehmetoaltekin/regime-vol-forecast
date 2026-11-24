"""
Tests that close gaps found by mutation testing in rvf.models (M1-M7 and M2c, PRESPEC 5 and 16).

Every expected value is derived here with plain numpy loops or numpy least squares, not with the code under test.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import rvf.models as M

from conftest import make_daily
from test_models import window


# ---------------------------------------------------------------------------------------------------------------
# independent reference computations
# ---------------------------------------------------------------------------------------------------------------

def _win(d: pd.DataFrame, filt=None, A=None, stress=None, n_train=None):
    return window(d["rv"].to_numpy(), d["r_cc"].to_numpy(), filt, A, stress, None, n_train)


def _probs(d: pd.DataFrame) -> np.ndarray:
    """Smoothed regime indicator as stand-in filtered probabilities, shape (n, 2), strictly inside (0, 1)."""
    p = pd.Series(d["state"].to_numpy(dtype=float)).rolling(5, min_periods=1).mean().to_numpy() * 0.8 + 0.1
    return np.column_stack([1 - p, p])


def _sum_target(y: np.ndarray, h: int) -> np.ndarray:
    """out[s] = y[s+1] + ... + y[s+h], NaN where the sum runs past the end."""
    out = np.full(len(y), np.nan)
    for s in range(len(y) - h):
        out[s] = y[s + 1: s + h + 1].sum()
    return out


def _har_regressors(y: np.ndarray, log: bool = False) -> np.ndarray:
    """[1, daily, 5-day mean, 22-day mean] per row, NaN before the 22nd row (log: averages of the daily logs)."""
    v = np.log(y) if log else y
    X = np.full((len(y), 4), np.nan)
    for s in range(21, len(y)):
        X[s] = [1.0, v[s], v[s - 4: s + 1].mean(), v[s - 21: s + 1].mean()]
    return X


def _lstsq(X, t, w=None):
    ok = np.isfinite(X).all(axis=1) & np.isfinite(t)
    X, t = X[ok], t[ok]
    if w is not None:
        sw = np.sqrt(w[ok])
        X, t = X * sw[:, None], t * sw
    return np.linalg.lstsq(X, t, rcond=None)[0]


def _garch_sum_path(r: np.ndarray, params, h: int) -> np.ndarray:
    """For every row s: the GARCH forecast made at s of the variance summed over days s+1..s+h (loops)."""
    mu, om, al, be, s0 = params
    out, s2 = [], s0
    for x in r:
        s2 = om + al * (x - mu) ** 2 + be * s2                  # variance of the next day
        f, total = s2, s2
        for _ in range(h - 1):
            f = om + (al + be) * f
            total += f
        out.append(total)
    return np.array(out)


# ---------------------------------------------------------------------------------------------------------------
# EWMA and GARCH start variances
# ---------------------------------------------------------------------------------------------------------------

def test_ewma_starts_at_the_sample_variance_of_the_return_window():
    """M1 starts the recursion at the ddof=1 sample variance of the training window's returns (PRESPEC 5); short window so the start still matters."""
    r = np.random.default_rng(10).normal(0, 1.5, 12)
    y = np.random.default_rng(11).uniform(0.5, 3.0, 12)                    # the rv column must not be what is used
    win = window(y, r)
    m = M.Ewma()
    assert m.fit(win, 1) and m.fit(win, 3)
    s0 = ((r - r.mean()) ** 2).sum() / (len(r) - 1)
    assert m.est[1] == pytest.approx(s0, rel=1e-12)
    assert abs(m.est[1] - np.var(y, ddof=1)) > 0.1
    s = s0
    for x in r:
        s = 0.94 * s + 0.06 * x ** 2
    assert 0.94 ** 12 * s0 > 0.4 * s                                       # the start is a large part of the forecast
    assert m.predict(win, 1) == pytest.approx(s, rel=1e-12)
    assert m.predict(win, 3) == pytest.approx(3 * s, rel=1e-12)


def test_garch_start_variance_is_the_population_variance_of_the_demeaned_returns():
    """M2 starts its variance recursion at var(r - mu) with ddof 0 (an implementation choice behind PRESPEC 5); path[0] is checked by hand."""
    r = make_daily(400, seed=3)["r_cc"].to_numpy()
    win = window(np.ones(len(r)), r)
    g = M.Garch()
    assert g.fit(win, 1)
    mu, om, al, be, s0 = g.est[1]
    e = r - mu
    assert s0 == pytest.approx(((e - e.mean()) ** 2).mean(), rel=1e-12)
    assert s0 > 0.5 and abs(s0 - om / (1 - al - be)) > 1e-3 * s0             # neither zero nor the unconditional variance
    path = g.path(win, 1)
    assert path[0] == pytest.approx(om + al * e[0] ** 2 + be * s0, rel=1e-12)
    assert np.allclose(path, _garch_sum_path(r, (mu, om, al, be, s0), 1), rtol=1e-12)


# ---------------------------------------------------------------------------------------------------------------
# GARCH acceptance conditions
# ---------------------------------------------------------------------------------------------------------------

def _fake_arch(params, flag):
    return lambda *a, **k: SimpleNamespace(fit=lambda **kw: SimpleNamespace(params=np.array(params, dtype=float),
                                                                            convergence_flag=flag))


@pytest.mark.parametrize("params, flag, accepted", [
    ([0.05, 0.1, 0.08, 0.9], 0, True),
    ([0.05, 0.1, 0.08, 0.9], 1, False),                                    # optimiser did not converge
    ([0.05, 0.0, 0.08, 0.9], 0, False),                                    # omega = 0 is not allowed
    ([0.05, -0.1, 0.08, 0.9], 0, False),                                   # negative omega
    ([0.05, 0.1, np.nan, 0.9], 0, False),
])
def test_garch_accepts_only_converged_finite_fits_with_positive_omega(monkeypatch, params, flag, accepted):
    """A GARCH fit counts as an estimate only if converged, finite and omega > 0; otherwise it is a failure (PRESPEC 5, estimation failures)."""
    r = np.random.default_rng(5).normal(0, 1, 50)
    win = window(np.ones(50), r)
    monkeypatch.setattr(M, "arch_model", _fake_arch(params, flag))
    g = M.Garch()
    assert bool(g.fit(win, 1)) is accepted
    if accepted:
        mu = params[0]
        assert g.est[1] == (mu, params[1], params[2], params[3], pytest.approx(np.var(r - mu), rel=1e-12))
    else:
        assert 1 not in g.est and g.predict(win, 1) is None


# ---------------------------------------------------------------------------------------------------------------
# M2c / M4 keep the GARCH parameters and the coefficients together when GARCH fails
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("with_regime", [False, True])
def test_calibrated_garch_failure_keeps_garch_parameters_and_coefficients_together(monkeypatch, with_regime):
    """After a failed GARCH refit M2c/M4 keep both the old GARCH parameters and the old coefficients unchanged and forecast with them (PRESPEC 16)."""
    d = make_daily(360, seed=5)
    filt, A, stress = _probs(d), np.eye(2), np.array([False, True])
    win1, win2 = _win(d.iloc[:200], filt[:200], A, stress), _win(d.iloc[60:260], filt[60:260], A, stress)
    m = M.CalibratedGarch(with_regime=with_regime)
    assert m.fit(win1, 1)
    est1, garch1 = m.est[1].copy(), m.garch.est[1]
    assert len(est1) == (3 if with_regime else 2)

    def refuse(*a, **k):
        raise RuntimeError("no convergence")

    monkeypatch.setattr(M, "arch_model", refuse)
    assert m.fit(win2, 1) is False
    assert np.array_equal(m.est[1], est1) and m.garch.est[1] == garch1            # bit-identical, nothing re-estimated
    cols = [np.ones(200), _garch_sum_path(d["r_cc"].to_numpy()[60:260], garch1, 1)]
    if with_regime:
        cols.append(filt[60:260, 1])
    X2 = np.column_stack(cols)
    assert m.predict(win2, 1) == pytest.approx(X2[-1] @ est1, rel=1e-12)
    redone = _lstsq(X2, _sum_target(d["rv"].to_numpy()[60:260], 1))                # what a re-estimation would give
    assert np.abs(redone - est1).max() > 1e-3

    fresh = M.CalibratedGarch(with_regime=with_regime)                           # no previous refit: nothing to carry
    assert fresh.fit(win2, 1) is False and 1 not in fresh.est and fresh.predict(win2, 1) is None


# ---------------------------------------------------------------------------------------------------------------
# direct regressions for h > 1
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("with_regime", [False, True])
def test_calibrated_garch_h5_regresses_the_5_day_sum_on_the_summed_garch_forecast(with_regime):
    """M2c/M4 for h = 5 regress the 5-day RV sum on the 5-day summed GARCH forecast (PRESPEC 16)."""
    d = make_daily(300, seed=6)
    filt, A, stress = _probs(d), np.eye(2), np.array([False, True])
    win = _win(d, filt, A, stress)
    m = M.CalibratedGarch(with_regime=with_regime)
    assert m.fit(win, 5)
    g5 = _garch_sum_path(d["r_cc"].to_numpy(), m.garch.est[5], 5)
    X = np.column_stack([np.ones(300), g5] + ([filt[:, 1]] if with_regime else []))
    t5 = _sum_target(d["rv"].to_numpy(), 5)
    assert np.allclose(m.est[5], _lstsq(X, t5), rtol=1e-9)
    assert m.predict(win, 5) == pytest.approx(X[-1] @ m.est[5], rel=1e-12)
    wrong = np.column_stack([np.ones(300), _garch_sum_path(d["r_cc"].to_numpy(), m.garch.est[5], 1)]
                            + ([filt[:, 1]] if with_regime else []))
    assert np.abs(_lstsq(wrong, t5) - m.est[5]).max() > 1e-3                      # a one-day GARCH regressor would differ
    assert np.abs(_lstsq(X, _sum_target(d["rv"].to_numpy(), 1)) - m.est[5]).max() > 1e-3


def test_log_har_h5_regresses_the_log_of_the_5_day_sum_without_a_horizon_shift():
    """M3 in logs for h = 5 regresses log(5-day sum) on the log regressors, with no division by h and the residual variance in the correction (PRESPEC 16, log-HAR)."""
    d = make_daily(300, seed=7)
    y = d["rv"].to_numpy()
    win = _win(d)
    m = M.Har(log=True)
    assert m.fit(win, 5)
    X, t = _har_regressors(y, log=True), np.log(_sum_target(y, 5))
    ref = _lstsq(X, t)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(t)
    resid = t[ok] - X[ok] @ ref
    beta, s2 = m.est[5]
    assert np.allclose(beta, ref, rtol=1e-9) and s2 == pytest.approx(resid @ resid / (ok.sum() - 4), rel=1e-9)
    assert m.predict(win, 5) == pytest.approx(np.exp(X[-1] @ ref + 0.5 * s2), rel=1e-9)


def _m6_reference(d, filt, A, h, log):
    """Per-regime weighted least squares of the h-day target with the one-step-ahead probabilities as weights."""
    y = d["rv"].to_numpy()
    X = _har_regressors(y, log=log)
    t = _sum_target(y, h)
    t = np.log(t) if log else t
    W = filt @ A
    ok = np.isfinite(X).all(axis=1) & np.isfinite(t)
    betas, s2 = [], []
    for k in (0, 1):
        b = _lstsq(X, t, W[:, k])
        betas.append(b)
        e = t[ok] - X[ok] @ b
        s2.append((W[ok, k] * e ** 2).sum() / W[ok, k].sum())
    return X, np.array(betas), np.array(s2), W


@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("h", [1, 5])
def test_regime_weighted_har_with_asymmetric_transition_matrix(log, h):
    """M6 weights with and mixes by the one-step-ahead probability filt @ A at every horizon, and its target is the h-day sum (PRESPEC 5, M6 details)."""
    d = make_daily(300, seed=8)
    st, rng = d["state"].to_numpy(), np.random.default_rng(8)                           # regime-specific HAR dynamics
    y = np.empty(300)
    y[0] = 1.0
    for t in range(299):
        y[t + 1] = (0.2, 2.0)[st[t + 1]] + (0.3, 0.1)[st[t + 1]] * y[t] + rng.uniform(0, 0.2)
    d = d.assign(rv=y)
    p = pd.Series(st.astype(float)).rolling(3, min_periods=1).mean().to_numpy() * 0.92 + 0.04
    filt, A = np.column_stack([1 - p, p]), np.array([[0.9, 0.1], [0.3, 0.7]])
    win = _win(d, filt, A, np.array([False, True]))
    m = M.RegimeWeightedHar(min_ess=20, log=log)
    assert m.fit(win, h)
    X, betas, s2, W = _m6_reference(d, filt, A, h, log)
    est_b = m.est[h][0] if log else m.est[h]
    assert np.allclose(est_b, betas, rtol=1e-9) and np.abs(betas[0] - betas[1]).max() > 0.02
    pi = filt[-1] @ A
    if log:
        assert np.allclose(m.est[h][1], s2, rtol=1e-9)
        per_regime = np.exp(betas @ X[-1] + 0.5 * s2)
    else:
        per_regime = betas @ X[-1]
    expected = pi @ per_regime
    assert m.predict(win, h) == pytest.approx(expected, rel=1e-9)
    assert abs(filt[-1] @ per_regime - expected) > 1e-3 * abs(expected)                  # not the filtered probability
    if h > 1:
        assert abs(filt[-1] @ np.linalg.matrix_power(A, h) @ per_regime - expected) > 1e-3 * abs(expected)


class _Recorder:
    """Stand-in for MarkovRegression.fit: records start_params and returns scripted (llf, converged) results."""

    def __init__(self, script):
        self.script, self.starts = list(script), []

    def __call__(self, start_params=None, **kw):
        self.starts.append(np.array(start_params, dtype=float))
        llf, converged = self.script[len(self.starts) - 1]
        return SimpleNamespace(llf=llf, params=np.array(start_params, dtype=float), mle_retvals={"converged": converged})


@pytest.mark.parametrize("log, h", [(False, 5), (True, 5), (False, 1)])
def test_ms_har_fit_pairs_for_horizon_h(monkeypatch, log, h):
    """M7 fits on (h-day target, HAR regressors) pairs of the rows whose target is known, divided by the window's mean RV in levels and by nothing in logs (PRESPEC 16)."""
    d = make_daily(130, seed=9)
    y = d["rv"].to_numpy()
    seen, real = [], M.MsHar._model

    def spy(yy, XX):
        seen.append((np.array(yy), np.array(XX)))
        return real(yy, XX)

    monkeypatch.setattr(M.MsHar, "_model", staticmethod(spy))
    monkeypatch.setattr(M.MarkovRegression, "fit", _Recorder([(1.0, True)]))
    m = M.MsHar(n_starts=1, log=log)
    assert m.fit(_win(d), h)
    scale = 1.0 if log else y.mean()
    X = _har_regressors(y, log=log)[:, 1:]
    t = _sum_target(y, h)
    t = np.log(t) if log else t
    ok = np.isfinite(X).all(axis=1) & np.isfinite(t)
    assert ok.sum() == 130 - 21 - h
    (yy, XX), = seen
    assert np.allclose(yy, t[ok] / scale, rtol=1e-12) and np.allclose(XX, X[ok] / scale, rtol=1e-12)
    assert m.est[h][1] == pytest.approx(scale, rel=1e-12)


# ---------------------------------------------------------------------------------------------------------------
# M6 effective sample size
# ---------------------------------------------------------------------------------------------------------------

def _ess_window(weights1: np.ndarray):
    """Window with A = identity, so the weights are the given probability of regime 1 (and its complement)."""
    d = make_daily(300, seed=11)
    filt = np.column_stack([1 - weights1, weights1])
    return d, filt, _win(d, filt, np.eye(2), np.array([False, True]))


def _block(rows) -> np.ndarray:
    w = np.zeros(300)
    w[rows] = 1.0
    return w


def test_regime_weighted_har_ess_threshold_is_strict_and_uses_the_exact_ess():
    """A regime with ESS exactly 80 (80 rows of weight 1) keeps its own HAR at min_ess 79 and 80 and falls back to the pooled HAR at 81 (PRESPEC 5, M6 details: ESS < threshold)."""
    d, filt, win = _ess_window(_block(range(100, 180)))
    y = d["rv"].to_numpy()
    X, t = _har_regressors(y), _sum_target(y, 1)
    own1 = _lstsq(X[100:180], t[100:180])
    own0 = _lstsq(np.delete(X, np.s_[100:180], 0), np.delete(t, np.s_[100:180]))
    pooled = _lstsq(X, t)
    assert np.abs(own1 - pooled).max() > 1e-3
    for min_ess, fallback in ((79.0, False), (80.0, False), (81.0, True)):
        m = M.RegimeWeightedHar(min_ess)
        assert m.fit(win, 1)
        assert np.allclose(m.est[1][0], own0, rtol=1e-8)
        assert np.allclose(m.est[1][1], pooled if fallback else own1, rtol=1e-8), min_ess
        assert m.extra_counts[(1, "ess_fallback_refits")] == int(fallback)
        assert m.extra_counts[(1, "ess_fallback_regimes")] == int(fallback)


def test_regime_weighted_har_ess_is_the_squared_sum_of_weights_over_the_sum_of_squared_weights():
    """PRESPEC 5, M6 details: ESS_k = (sum pi)^2 / sum pi^2. 100 rows of weight 0.5 have ESS 100, not their weight sum 50."""
    w = np.zeros(300)
    w[100:200] = 0.5
    d, filt, win = _ess_window(w)
    y = d["rv"].to_numpy()
    X, t = _har_regressors(y), _sum_target(y, 1)
    own1 = _lstsq(X[100:200], t[100:200])                                      # a constant weight does not change the weighted fit
    pooled = _lstsq(X, t)
    assert np.abs(own1 - pooled).max() > 1e-3
    for min_ess, fallback in ((99.0, False), (100.0, False), (101.0, True)):
        m = M.RegimeWeightedHar(min_ess)
        assert m.fit(win, 1)
        assert np.allclose(m.est[1][1], pooled if fallback else own1, rtol=1e-8), min_ess
        assert m.extra_counts[(1, "ess_fallback_regimes")] == int(fallback)


def test_regime_weighted_har_regime_without_weight_falls_back_to_pooled():
    """A regime whose weights are all zero has ESS 0 and uses the pooled HAR, and the fallback is counted (PRESPEC 5, M6 details)."""
    d, filt, win = _ess_window(np.zeros(300))
    y = d["rv"].to_numpy()
    pooled = _lstsq(_har_regressors(y), _sum_target(y, 1))
    m = M.RegimeWeightedHar(min_ess=10.0)
    assert m.fit(win, 1)
    assert np.allclose(m.est[1][1], pooled, rtol=1e-8) and np.allclose(m.est[1][0], pooled, rtol=1e-8)
    assert m.extra_counts[(1, "ess_fallback_refits")] == 1 and m.extra_counts[(1, "ess_fallback_regimes")] == 1


def test_regime_weighted_har_ess_counts_only_rows_with_a_valid_pair():
    """The ESS of a regime counts only rows with a finite (x_t, RV_{t+1}) pair, as implemented: 60 usable rows plus 22 unusable ones give ESS 60, not 82."""
    w = _block(list(range(0, 21)) + list(range(100, 160)) + [299])             # rows 0-20 have no HAR month, row 299 no target
    d, filt, win = _ess_window(w)
    y = d["rv"].to_numpy()
    X, t = _har_regressors(y), _sum_target(y, 1)
    pooled, own1 = _lstsq(X, t), _lstsq(X[100:160], t[100:160])
    low = M.RegimeWeightedHar(min_ess=50.0)
    high = M.RegimeWeightedHar(min_ess=70.0)
    assert low.fit(win, 1) and high.fit(win, 1)
    assert np.allclose(low.est[1][1], own1, rtol=1e-8) and low.extra_counts[(1, "ess_fallback_refits")] == 0
    assert np.allclose(high.est[1][1], pooled, rtol=1e-8) and high.extra_counts[(1, "ess_fallback_regimes")] == 1


def test_regime_weighted_har_fallback_counts_accumulate_over_refits_and_horizons():
    """The ESS fallback counts (refits with a fallback, regimes that fell back) add up over refits and are kept per horizon (PRESPEC 5, M6 details: number of such refits is reported)."""
    d, filt, win = _ess_window(_block(range(100, 180)))
    m = M.RegimeWeightedHar(min_ess=79.0)
    seen = []
    for h, min_ess in ((1, 79.0), (1, 81.0), (1, 1e9), (5, 81.0), (1, 79.0)):
        m.min_ess = min_ess
        assert m.fit(win, h)
        seen.append({k: v for k, v in m.extra_counts.items()})
    c = lambda i, h, key: seen[i].get((h, "ess_fallback_" + key), 0)
    assert [(c(i, 1, "refits"), c(i, 1, "regimes")) for i in range(5)] == [(0, 0), (1, 1), (2, 3), (2, 3), (2, 3)]
    assert (c(3, 5, "refits"), c(3, 5, "regimes")) == (1, 1) and (c(4, 5, "refits"), c(4, 5, "regimes")) == (1, 1)


# ---------------------------------------------------------------------------------------------------------------
# M7 fit selection
# ---------------------------------------------------------------------------------------------------------------

def test_ms_har_fit_tries_n_starts_plus_previous_estimate_and_keeps_the_highest_likelihood(monkeypatch):
    """Each M7 refit runs n_starts random starts plus the previous estimate for that horizon and keeps the highest likelihood (PRESPEC 5, M7 details)."""
    d = make_daily(120, seed=12)
    win = _win(d)
    rec = _Recorder([(10.0, True), (30.0, True), (20.0, True),                       # first fit: the second start wins
                     (1.0, True), (2.0, True), (3.0, True), (99.0, True),            # second fit: the previous estimate wins
                     (1.0, True), (2.0, True), (3.0, True)])                         # other horizon: no previous estimate
    monkeypatch.setattr(M.MarkovRegression, "fit", rec)
    m = M.MsHar(n_starts=3, seed=3)
    assert m.fit(win, 1)
    assert len(rec.starts) == 3 and not any(np.allclose(rec.starts[i], rec.starts[j]) for i in range(3) for j in range(i))
    assert np.array_equal(m.est[1][0], rec.starts[1])
    first_best = m.est[1][0].copy()
    assert m.fit(win, 1)
    assert len(rec.starts) == 7
    assert np.array_equal(rec.starts[6], first_best)                                 # previous estimate is the extra start
    assert np.array_equal(m.est[1][0], first_best)
    assert m.fit(win, 5)
    assert len(rec.starts) == 10                                                     # est[1] is not a start for h = 5
    assert np.array_equal(m.est[5][0], rec.starts[9])                                # highest likelihood of the three


def test_ms_har_fit_rejects_non_converged_and_non_finite_fits(monkeypatch):
    """An M7 fit that did not converge, or has a non-finite likelihood, is never kept even if its likelihood is highest; if none qualifies the refit fails and the old estimate stays (PRESPEC 5, estimation failures)."""
    d = make_daily(120, seed=13)
    win = _win(d)

    def use(script):
        rec = _Recorder(script)
        monkeypatch.setattr(M.MarkovRegression, "fit", rec)
        return rec

    rec = use([(50.0, False), (np.nan, True), (np.inf, True), (40.0, False)])
    m = M.MsHar(n_starts=4, seed=1)
    assert m.fit(win, 1) is False and 1 not in m.est and len(rec.starts) == 4         # nothing admissible: the refit fails
    rec = use([(50.0, False), (np.nan, True), (3.0, True), (2.0, True)])
    assert m.fit(win, 1) and np.array_equal(m.est[1][0], rec.starts[2])               # best of the admissible fits
    before = m.est[1]
    rec = use([(50.0, False), (40.0, False), (np.nan, True), (np.inf, True), (30.0, False)])   # four starts and the previous one
    assert m.fit(win, 1) is False and m.est[1] is before and len(rec.starts) == 5


# ---------------------------------------------------------------------------------------------------------------
# M7 predict
# ---------------------------------------------------------------------------------------------------------------

MS_NAMES = ["p[0->0]", "p[1->0]", "const[0]", "const[1]", "x1[0]", "x1[1]", "x2[0]", "x2[1]", "x3[0]", "x3[1]",
            "sigma2[0]", "sigma2[1]"]


def _ms_params(h: int, log: bool) -> np.ndarray:
    """Hand-set parameters in the order of MarkovRegression's names: regime 0 calm, regime 1 stress, asymmetric P."""
    levels = (-0.35, 1.1) if log else (1.0 * h, 4.3 * h)
    sig = (0.05, 0.4) if log else (0.1 * h, 1.0 * h)
    return np.array([0.9, 0.4, levels[0], levels[1], 0.10, 0.30, 0.20, 0.10, 0.05, 0.40, sig[0], sig[1]])


def _ms_forecast(y_full: np.ndarray, filter_rows: int, params, scale: float, h: int, log: bool, steps=None) -> float:
    """Own implementation of M7's forecast: Hamilton filter over the pairs inside y_full[:filter_rows], P^h, regime means."""
    v = np.log(y_full) if log else y_full
    ys, xs = [], []
    for s in range(21, filter_rows - h):
        total = y_full[s + 1: s + h + 1].sum()
        ys.append(np.log(total) if log else total)
        xs.append([v[s], v[s - 4: s + 1].mean(), v[s - 21: s + 1].mean()])
    ys, xs = np.array(ys) / scale, np.array(xs) / scale
    a, b = params[0], params[1]
    P = np.array([[a, b], [1 - a, 1 - b]])                                           # P[i, j] = P(regime i now | regime j before)
    pred = np.array([b, 1 - a]) / (1 - a + b)                                        # steady state, as statsmodels
    for t in range(len(ys)):
        dens = np.empty(2)
        for k in (0, 1):
            mean = params[2 + k] + sum(params[4 + 2 * j + k] * xs[t, j] for j in range(3))
            dens[k] = np.exp(-0.5 * (ys[t] - mean) ** 2 / params[10 + k]) / np.sqrt(2 * np.pi * params[10 + k])
        filt = pred * dens / (pred @ dens)
        pred = P @ filt
    prob = filt
    for _ in range(h if steps is None else steps):
        prob = P @ prob
    x = np.array([v[-1], v[-5:].mean(), v[-22:].mean()]) / scale
    f = 0.0
    for k in (0, 1):
        mean = params[2 + k] + sum(params[4 + 2 * j + k] * x[j] for j in range(3))
        f += prob[k] * (np.exp(mean + 0.5 * params[10 + k]) if log else mean)
    return f if log else f * scale


def _two_level_window(n_calm=120, n_high=25, seed=14, scale_up=1.0):
    """Calm rows (train window) followed by a high-variance stretch; the window is longer than its training part."""
    rng = np.random.default_rng(seed)
    y = np.concatenate([rng.uniform(0.5, 0.9, n_calm), rng.uniform(2.5, 3.5, n_high)]) * scale_up
    return y, window(y, np.zeros(len(y)), n_train=n_calm)


def test_ms_har_parameter_layout_matches_the_hand_set_parameters():
    """Guards the helper above: statsmodels' MarkovRegression orders its parameters as MS_NAMES."""
    y, _ = _two_level_window()
    assert M.MsHar._model(y[1:30], np.random.default_rng(0).normal(size=(29, 3))).param_names == MS_NAMES


def test_ms_har_predict_filters_over_the_whole_window_and_rescales():
    """M7's forecast runs the filter over every pair known at the origin, also beyond the training rows, and multiplies back by the training window's mean RV (PRESPEC 5 M7, 16)."""
    y, win = _two_level_window()
    n_train = win.n_train
    scale = float(np.mean(win.train.y))
    params = _ms_params(1, log=False)
    m = M.MsHar()
    m.est[1] = (params, scale, 1)
    f = m.predict(win, 1)
    assert f == pytest.approx(_ms_forecast(y, len(y), params, scale, 1, False), rel=1e-7)
    stale = _ms_forecast(y, n_train, params, scale, 1, False)                          # filter over the training rows only
    assert abs(f - stale) > 0.2 * f
    _, big_win = _two_level_window(scale_up=100.0)
    big = M.MsHar()
    big.est[1] = (params, 100.0 * scale, 1)
    assert big.predict(big_win, 1) == pytest.approx(100.0 * f, rel=1e-7)


@pytest.mark.parametrize("log", [False, True])
def test_ms_har_predict_propagates_the_filtered_probability_h_steps(log):
    """For h = 5 M7 mixes the regime regressions with P^5 times the last filtered probability (PRESPEC 16, M7 for h > 1), in levels and logs."""
    y, win = _two_level_window()
    params = _ms_params(5, log)
    scale = 1.0 if log else float(np.mean(win.train.y))
    m = M.MsHar(log=log)
    m.est[5] = (params, scale, 1)
    f = m.predict(win, 5)
    assert f == pytest.approx(_ms_forecast(y, len(y), params, scale, 5, log), rel=1e-7)
    one_step = _ms_forecast(y, len(y), params, scale, 5, log, steps=1)
    assert abs(one_step - f) > 0.02 * f                                                # one step of propagation would differ
