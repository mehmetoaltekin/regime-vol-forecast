"""Tests for the robustness variants: log-HAR, HARQ and training on real-time vintage probabilities."""
import numpy as np
import pandas as pd
import pytest

import rvf.models as M
from rvf.regime import RegimeConfig
from rvf.walkforward import WFConfig, Window, cumulative_target, har_columns, refit_origins, run_walk_forward

from conftest import make_daily, perturb_after

FAST = RegimeConfig(n_init=2)
WF = WFConfig(window=150, refit_every=40)


def window(y, rq=None, filt=None, A=None, stress=None) -> Window:
    y = np.asarray(y, dtype=float)
    data = pd.DataFrame({"rv": y, "r_cc": np.zeros(len(y))}, index=pd.bdate_range("2020-01-01", periods=len(y)))
    if rq is not None:
        data["rq"] = rq
    data = data.join(har_columns(data["rv"]))
    return Window(data, len(y), filt, A, stress)


def log_har_series(n=500, c=0.05, bd=0.4, bw=0.3, bm=0.2, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    ly = list(rng.normal(0, 0.3, 22))
    for _ in range(n - 22):
        ly.append(c + bd * ly[-1] + bw * np.mean(ly[-5:]) + bm * np.mean(ly[-22:]) + rng.normal(0, noise))
    return np.exp(np.array(ly))


def test_har_columns_hold_log_averages():
    y = pd.Series(np.arange(1.0, 31.0))
    c = har_columns(y)
    ly = np.log(y)
    assert np.allclose(c["lhar_d"], ly) and np.allclose(c["lhar_w"].iloc[4:], ly.rolling(5).mean().iloc[4:])
    assert np.allclose(c["lhar_m"].iloc[21:], ly.rolling(22).mean().iloc[21:])
    assert np.isnan(har_columns(pd.Series([1.0, 0.0, 2.0]))["lhar_d"].iloc[1])        # a zero has no log


def test_log_har_recovers_planted_coefficients_and_applies_the_correction():
    y = log_har_series()
    win = window(y)
    m = M.Har(log=True)
    assert m.name == "m3_log" and m.fit(win, 1)
    beta, s2 = m.est[1]
    assert np.allclose(beta, [0.05, 0.4, 0.3, 0.2], atol=1e-8) and s2 == pytest.approx(0.0, abs=1e-18)
    x = M.har_x(win, log=True)[-1]
    assert m.predict(win, 1) == pytest.approx(np.exp(x @ beta))
    noisy = window(log_har_series(noise=0.4, seed=1))
    m2 = M.Har(log=True)
    m2.fit(noisy, 1)
    beta, s2 = m2.est[1]
    X, t = M.har_x(noisy, log=True), M.log_target(noisy, 1)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(t)
    ref = np.linalg.lstsq(X[ok], t[ok], rcond=None)[0]
    resid = t[ok] - X[ok] @ ref
    assert np.allclose(beta, ref) and s2 == pytest.approx(resid @ resid / (ok.sum() - 4))
    assert m2.predict(noisy, 1) == pytest.approx(np.exp(M.har_x(noisy, log=True)[-1] @ ref + 0.5 * s2))
    assert s2 == pytest.approx(0.16, rel=0.25)
    m2.fit(noisy, 5)                                                                   # log of the 5-day sum
    assert np.isfinite(m2.predict(noisy, 5)) and m2.predict(noisy, 5) > 0


def test_log_x_variant_uses_the_vintage_probability_only_when_fitting():
    n = 500
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, n)
    y = np.exp(0.1 * np.cumsum(rng.normal(0, 0.2, n)) * 0 + 0.5 * p + rng.normal(0, 0.05, n))
    win = window(y, filt=np.column_stack([1 - p, p]), A=np.eye(2), stress=np.array([False, True]))
    m = M.Har(with_regime=True, log=True)
    assert m.name == "m5_log"
    m.fit(win, 1)
    assert len(m.est[1][0]) == 5 and m.est[1][1] >= 0


def test_harq_recovers_planted_coefficients():
    rng = np.random.default_rng(3)
    n = 600
    y = list(rng.uniform(0.5, 1.5, 22))
    rq = rng.uniform(0.2, 3.0, n)
    for t in range(21, n - 1):
        y.append(0.05 + (0.3 + 0.1 * np.sqrt(rq[t])) * y[t] + 0.3 * np.mean(y[t - 4: t + 1]) + 0.25 * np.mean(y[t - 21: t + 1]) )
    y = np.array(y)
    win = window(y, rq=rq)
    m = M.Har(q=True)
    assert m.name == "m3_q" and m.fit(win, 1)
    # columns: const, daily, weekly, monthly, sqrt(RQ) * daily
    assert np.allclose(m.est[1], [0.05, 0.3, 0.3, 0.25, 0.1], atol=1e-8)
    x = M.har_x(win, q=True)[-1]
    assert x[4] == pytest.approx(np.sqrt(rq[-1]) * y[-1])
    assert m.predict(win, 1) == pytest.approx(x @ m.est[1])
    with pytest.raises(ValueError):
        M.Har(log=True, q=True)
    with pytest.raises(KeyError):                                                       # HARQ needs the quarticity column
        M.Har(q=True).fit(window(y), 1)
    p = rng.uniform(size=n)
    wx = window(y, rq=rq, filt=np.column_stack([1 - p, p]), A=np.eye(2), stress=np.array([False, True]))
    mx = M.Har(with_regime=True, q=True)
    mx.fit(wx, 1)
    assert mx.name == "m5_q" and len(mx.est[1]) == 6


def test_regime_weighted_log_har():
    rng = np.random.default_rng(4)
    n = 600
    state = (np.arange(n) // 50) % 2
    ly = np.empty(n)
    ly[0] = 0.0
    for t in range(n - 1):
        k = state[t + 1]
        ly[t + 1] = (0.0, 1.0)[k] + (0.5, 0.2)[k] * ly[t] + rng.normal(0, (0.1, 0.5)[k])
    y = np.exp(ly)
    nxt = np.r_[state[1:], state[-1]]
    filt = np.column_stack([1 - nxt, nxt]) * 0.998 + 0.001
    win = window(y, filt=filt, A=np.eye(2), stress=np.array([False, True]))
    m = M.RegimeWeightedHar(min_ess=50, log=True)
    assert m.name == "m6_log" and m.fit(win, 1)
    betas, s2 = m.est[1]
    assert betas.shape == (2, 4) and s2[1] > 5 * s2[0]                                # the noisy regime has the larger variance
    X, t = M.har_x(win, log=True), M.log_target(win, 1)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(t)
    for k in (0, 1):
        w = filt[:, k]
        assert np.allclose(betas[k], M.ols(X, t, w))
        e = t[ok] - X[ok] @ betas[k]
        assert s2[k] == pytest.approx((w[ok] * e ** 2).sum() / w[ok].sum())
    pi = win.pi(1)[-1]
    assert m.predict(win, 1) == pytest.approx(pi @ np.exp(betas @ X[-1] + 0.5 * s2))
    strict = M.RegimeWeightedHar(min_ess=1e9, log=True)
    strict.fit(win, 1)
    pooled = M.Har(log=True)
    pooled.fit(win, 1)
    assert np.allclose(strict.est[1][0], [pooled.est[1][0]] * 2) and np.allclose(strict.est[1][1], pooled.est[1][1])
    assert strict.predict(win, 1) == pytest.approx(pooled.predict(win, 1))


def test_ms_har_log_forecast_applies_the_regime_variance_correction():
    rng = np.random.default_rng(5)
    n = 500
    state = (np.arange(n) // 60) % 2
    ly = np.empty(n)
    ly[0] = 0.0
    for t in range(n - 1):
        k = state[t + 1]
        ly[t + 1] = (0.0, 0.8)[k] + (0.5, 0.3)[k] * ly[t] + rng.normal(0, (0.1, 0.6)[k])
    win = window(np.exp(ly))
    m = M.MsHar(n_starts=3, seed=1, log=True)
    assert m.name == "m7_log" and m.fallback == "m3_log" and m.fit(win, 1)
    params = m.est[1][0]
    names = M.MsHar._model(*m._pairs(win, 1, 1.0)).param_names
    f = m.predict(win, 1)
    assert np.isfinite(f) and f > 0
    y, X = m._pairs(win, 1, 1.0)
    res = M.MsHar._model(y, X).filter(params)
    prob = np.asarray(res.regime_transition)[:, :, 0] @ np.asarray(res.filtered_marginal_probabilities)[-1]
    x = M.har_x(win, log=True)[-1, 1:]
    manual = sum(prob[k] * np.exp(params[names.index(f"const[{k}]")]
                                  + np.array([params[names.index(f"x{j}[{k}]")] for j in (1, 2, 3)]) @ x
                                  + 0.5 * params[names.index(f"sigma2[{k}]")]) for k in (0, 1))
    assert f == pytest.approx(manual)


def test_robustness_model_sets_run_and_do_not_look_ahead():
    daily = make_daily(340, seed=11).assign(rq=lambda d: d.rv ** 2)
    cfg = WFConfig(window=200, refit_every=60, horizons=(1, 5))
    cut = 261
    changed = perturb_after(daily, cut)
    changed.iloc[cut:, changed.columns.get_loc("rq")] = changed["rv"].iloc[cut:] ** 2
    base = M.log_models(min_ess=20, ms_starts=1) + M.harq_models()
    a = run_walk_forward(daily, base, cfg, FAST)
    b = run_walk_forward(changed, M.log_models(min_ess=20, ms_starts=1) + M.harq_models(), cfg, FAST)
    assert set(a.forecasts.index.get_level_values("model")) == {"m3_log", "m5_log", "m6_log", "m7_log", "m3_q", "m5_q"}
    assert np.isfinite(a.forecasts["forecast"]).all() and (a.forecasts["forecast"] > 0).all()
    last_clean = daily.index[cut - 1]
    fa, fb = a.forecasts[a.forecasts["origin"] <= last_clean], b.forecasts[b.forecasts["origin"] <= last_clean]
    pd.testing.assert_frame_equal(fa[["forecast", "raw", "source"]], fb[["forecast", "raw", "source"]])
    after = a.forecasts["origin"] > last_clean
    for name in ("m3_log", "m5_log", "m6_log", "m7_log", "m3_q", "m5_q"):
        assert not np.allclose(a.forecasts[after].xs(name, level="model")["raw"], b.forecasts[after].xs(name, level="model")["raw"]), name


class Spy:
    name, fallback = "spy", None

    def __init__(self):
        self.fits = []

    def fit(self, win, h):
        self.fits.append(win)
        return True

    def predict(self, win, h):
        return float(win.y[-1])


def test_vintage_training_calendar_and_what_models_are_shown():
    daily = make_daily(420)
    vcfg = WFConfig(window=150, refit_every=40, vintage_training=True)
    spy = Spy()
    res = run_walk_forward(daily, [spy, M.Har(with_regime=True)], vcfg, FAST)
    data = daily.iloc[1:]
    origins = refit_origins(len(data), vcfg)
    first = int(origins[origins >= 2 * 150 - 2][0])
    h1 = res.forecasts.xs("spy", level="model").xs(1, level="h")
    assert h1["origin"].iloc[0] == data.index[first] and h1.index[0] == data.index[first + 1]
    assert [w.data.index[-1] for w in spy.fits] == [data.index[o] for o in origins if o >= first]
    assert res.regime.index[0] == data.index[149]                                       # the HMM runs from the first window on
    for w in spy.fits:                                                                  # the vintage is what was computed on each day
        assert w.vp is not None and np.isfinite(w.vp).all() and np.isfinite(w.vpi).all()
        assert np.allclose(w.vp, res.regime["prob_stress"].reindex(w.data.index).to_numpy())
        assert np.allclose(w.vpi.sum(axis=1), 1.0)
    w = spy.fits[1]
    assert not np.allclose(w.p, w.vp)                                                   # the refit window's own filter differs
    assert w.vp[-1] == pytest.approx(w.p[-1])                                           # but the last row is the same day's value
    plain = run_walk_forward(daily, [Spy()], WF, FAST)
    assert plain.forecasts.index.get_level_values("target_day")[0] < h1.index[0]        # fewer forecasts with vintage training


def test_models_use_the_vintage_probability_only_for_estimation():
    daily = make_daily(420)
    spy = Spy()
    run_walk_forward(daily, [spy], WFConfig(window=150, refit_every=40, vintage_training=True), FAST)
    win = spy.fits[0]
    y = cumulative_target(win.y, 1)
    for with_vintage in (True, False):
        w = win if with_vintage else Window(win.data, win.n_train, win.filt, win.A, win.stress, win.state_var, win.target)
        m5, m4, m6 = M.Har(with_regime=True), M.CalibratedGarch(with_regime=True), M.RegimeWeightedHar(min_ess=5)
        for m in (m5, m4, m6):
            m.fit(w, 1)
        pfit = win.vp if with_vintage else w.p
        X = np.column_stack([M.har_x(w), pfit])
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        assert np.allclose(m5.est[1], np.linalg.lstsq(X[ok], y[ok], rcond=None)[0])
        g = m4.garch.path(w, 1)
        X4 = np.column_stack([np.ones(len(y)), g, pfit])
        ok4 = np.isfinite(X4).all(axis=1) & np.isfinite(y)
        assert np.allclose(m4.est[1], np.linalg.lstsq(X4[ok4], y[ok4], rcond=None)[0])
        W = win.vpi if with_vintage else w.pi(1)
        Xh = M.har_x(w)
        okh = np.isfinite(Xh).all(axis=1) & np.isfinite(y)
        ess = W[okh, 1].sum() ** 2 / (W[okh, 1] ** 2).sum()
        assert ess >= 5                                                                  # the stress regime did not fall back
        assert np.allclose(m6.est[1][1], M.ols(Xh, y, W[:, 1]))
        # forecasts use the origin's probability, which is the same in both set-ups
        assert m5.predict(w, 1) == pytest.approx(float(np.r_[M.har_x(w)[-1], w.p[-1]] @ m5.est[1]))
    a, b = M.Har(with_regime=True), M.Har(with_regime=True)
    a.fit(win, 1)
    b.fit(Window(win.data, win.n_train, win.filt, win.A, win.stress, win.state_var, win.target), 1)
    assert not np.allclose(a.est[1], b.est[1])


def test_vintage_runs_do_not_look_ahead_and_need_a_fixed_k():
    daily = make_daily(420, seed=13)
    cfg = WFConfig(window=150, refit_every=40, vintage_training=True)
    cut = 351
    changed = perturb_after(daily, cut)
    a = run_walk_forward(daily, [M.Har(with_regime=True), M.RegimeWeightedHar(20), M.CalibratedGarch(with_regime=True)], cfg, FAST)
    b = run_walk_forward(changed, [M.Har(with_regime=True), M.RegimeWeightedHar(20), M.CalibratedGarch(with_regime=True)], cfg, FAST)
    last_clean = daily.index[cut - 1]
    fa, fb = a.forecasts[a.forecasts["origin"] <= last_clean], b.forecasts[b.forecasts["origin"] <= last_clean]
    assert len(fa) > 30
    pd.testing.assert_frame_equal(fa[["forecast", "raw", "source"]], fb[["forecast", "raw", "source"]])
    with pytest.raises(ValueError, match="fixed number"):
        run_walk_forward(daily, [M.Har()], cfg, RegimeConfig(n_components=(2, 3), n_init=1))
    with pytest.raises(ValueError):
        run_walk_forward(daily, [M.Har()], cfg, None)
    with pytest.raises(ValueError, match="enough"):
        run_walk_forward(make_daily(300), [M.Har()], cfg, FAST)
