"""Unit tests for the forecast models."""
import numpy as np
import pandas as pd
import pytest

import rvf.models as M
from rvf.regime import RegimeConfig
from rvf.walkforward import WFConfig, Window, cumulative_target, har_columns, run_walk_forward

from conftest import make_daily, perturb_after


def window(y, r=None, filt=None, A=None, stress=None, state_var=None, n_train=None) -> Window:
    y = np.asarray(y, dtype=float)
    idx = pd.bdate_range("2020-01-01", periods=len(y))
    data = pd.DataFrame({"rv": y, "r_cc": np.zeros(len(y)) if r is None else r}, index=idx)
    data = data.join(har_columns(data["rv"]))
    return Window(data, n_train or len(y), filt, A, stress, state_var)


def har_series(n=400, c=0.1, bd=0.4, bw=0.3, bm=0.2, seed=0):
    """A series that follows y[t+1] = c + bd*y[t] + bw*mean5 + bm*mean22 exactly after a random start."""
    y = list(np.random.default_rng(seed).uniform(0.5, 1.5, 22))
    for _ in range(n - 22):
        y.append(c + bd * y[-1] + bw * np.mean(y[-5:]) + bm * np.mean(y[-22:]))
    return np.array(y)


def test_ewma_matches_the_recursion():
    r = np.random.default_rng(0).normal(0, 1, 300)
    win = window(np.ones(300), r)
    m = M.Ewma()
    assert m.fit(win, 1) and m.fit(win, 5)
    s = np.var(r, ddof=1)
    for x in r:
        s = 0.94 * s + 0.06 * x ** 2
    assert m.predict(win, 1) == pytest.approx(s)
    assert m.predict(win, 5) == pytest.approx(5 * s)


def test_garch_path_and_multi_step_forecast():
    rng = np.random.default_rng(1)
    omega, alpha, beta, n = 0.05, 0.08, 0.90, 3000
    s2, r = omega / (1 - alpha - beta), []
    for _ in range(n):
        r.append(np.sqrt(s2) * rng.normal())
        s2 = omega + alpha * r[-1] ** 2 + beta * s2
    win = window(np.ones(n), np.array(r))
    g = M.Garch()
    assert g.fit(win, 1) and g.fit(win, 3)
    mu, om, al, be, s0 = g.est[1]
    assert al == pytest.approx(alpha, abs=0.03) and be == pytest.approx(beta, abs=0.04) and abs(mu) < 0.05
    v, path = s0, []
    for x in r:
        v = om + al * (x - mu) ** 2 + be * v
        path.append(v)
    assert np.allclose(g.path(win, 1), path)
    mu, om, al, be, s0 = g.est[3]
    v = s0
    for x in r:
        v = om + al * (x - mu) ** 2 + be * v
    f2 = om + (al + be) * v
    f3 = om + (al + be) * f2
    assert g.predict(win, 3) == pytest.approx(v + f2 + f3)
    assert M.Garch().predict(win, 1) is None                                           # no estimate yet


def test_har_recovers_planted_coefficients():
    y = har_series()
    win = window(y)
    m = M.Har()
    m.fit(win, 1)
    assert np.allclose(m.est[1], [0.1, 0.4, 0.3, 0.2], atol=1e-8)
    nxt = 0.1 + 0.4 * y[-1] + 0.3 * y[-5:].mean() + 0.2 * y[-22:].mean()
    assert m.predict(win, 1) == pytest.approx(nxt)
    m.fit(win, 5)                                                                      # direct regression on the 5-day sum
    X, t5 = M.har_x(win), cumulative_target(y, 5)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(t5)
    assert np.allclose(m.est[5], np.linalg.lstsq(X[ok], t5[ok], rcond=None)[0])
    assert ok.sum() == len(y) - 21 - 5                                                 # no target beyond the window


def test_har_x_and_calibrated_garch_use_prob_stress():
    rng = np.random.default_rng(2)
    n = 500
    p = rng.uniform(0, 1, n)
    filt = np.column_stack([1 - p, p])
    stress, A = np.array([False, True]), np.eye(2)
    y = np.empty(n)
    y[0] = 1.0
    for t in range(n - 1):
        y[t + 1] = 0.2 + 0.5 * y[t] + 2.0 * p[t]                                        # p of day t moves day t+1
    win = window(y, rng.normal(0, 1, n), filt, A, stress)
    m5 = M.Har(with_regime=True)
    m5.fit(win, 1)
    assert m5.name == "m5" and m5.est[1][1] == pytest.approx(0.5, abs=1e-6) and m5.est[1][4] == pytest.approx(2.0, abs=1e-6)
    assert m5.predict(win, 1) == pytest.approx(0.2 + 0.5 * y[-1] + 2.0 * p[-1])
    m4, m2c = M.CalibratedGarch(with_regime=True), M.CalibratedGarch()
    assert m4.fit(win, 1) and m2c.fit(win, 1) and (m4.name, m2c.name) == ("m4", "m2c")
    g = m4.garch.path(win, 1)
    assert np.allclose(m4.est[1], np.linalg.lstsq(np.column_stack([np.ones(n), g, p])[:-1], y[1:], rcond=None)[0])
    assert m4.predict(win, 1) == pytest.approx(m4.est[1] @ [1, g[-1], p[-1]])
    assert len(m2c.est[1]) == 2


def test_regime_weighted_har_recovers_regime_coefficients_and_falls_back():
    rng = np.random.default_rng(3)
    n = 600
    state = (np.arange(n) // 50) % 2                                                    # regime of the day being forecast
    x = rng.uniform(0.5, 2.0, n)
    y = np.empty(n)
    y[0] = x[0]
    beta = np.array([[0.1, 0.3], [1.0, 0.9]])                                           # intercept and daily slope per regime
    for t in range(n - 1):
        y[t + 1] = beta[state[t + 1], 0] + beta[state[t + 1], 1] * y[t] if t % 7 else x[t]
    # filtered probabilities chosen so that the one-step-ahead probability is (almost) the next day's regime
    nxt = np.r_[state[1:], state[-1]]
    filt = np.column_stack([1 - nxt, nxt]) * 0.998 + 0.001
    win = window(y, None, filt, np.eye(2), np.array([False, True]))
    X, t1 = M.har_x(win), cumulative_target(y, 1)
    m = M.RegimeWeightedHar(min_ess=50)
    m.fit(win, 1)
    for k in (0, 1):
        ok = np.isfinite(X).all(axis=1) & np.isfinite(t1)
        w = filt[:, k]
        ref = np.linalg.lstsq(X[ok] * np.sqrt(w[ok])[:, None], t1[ok] * np.sqrt(w[ok]), rcond=None)[0]
        assert np.allclose(m.est[1][k], ref)
    assert not np.allclose(m.est[1][0], m.est[1][1], atol=0.05)
    assert m.predict(win, 1) == pytest.approx(filt[-1] @ (m.est[1] @ X[-1]))
    assert m.extra_counts[(1, "ess_fallback_refits")] == 0

    strict = M.RegimeWeightedHar(min_ess=1e9)                                           # every regime falls back to the pooled HAR
    strict.fit(win, 1)
    pooled = M.Har()
    pooled.fit(win, 1)
    assert np.allclose(strict.est[1], [pooled.est[1], pooled.est[1]])
    assert strict.predict(win, 1) == pytest.approx(pooled.predict(win, 1))
    assert strict.extra_counts[(1, "ess_fallback_refits")] == 1 and strict.extra_counts[(1, "ess_fallback_regimes")] == 2


def test_pure_hmm_forecast_formula():
    A = np.array([[0.9, 0.1], [0.3, 0.7]])
    filt = np.array([[0.5, 0.5], [0.2, 0.8]])
    win = window([1.0, 1.0], None, filt, A, np.array([False, True]), np.array([0.5, 4.0]))
    m = M.PureHmm()
    assert m.fit(win, 1)
    p1 = filt[-1] @ A
    assert m.predict(win, 1) == pytest.approx(p1 @ [0.5, 4.0])
    assert m.predict(win, 2) == pytest.approx(p1 @ [0.5, 4.0] + p1 @ A @ [0.5, 4.0])
    no_var = window([1.0, 1.0], None, filt, A, np.array([False, True]), None)
    assert not m.fit(no_var, 1) and m.predict(no_var, 1) is None


def test_ms_har_fits_two_regimes_and_forecasts_from_filtered_probabilities():
    rng = np.random.default_rng(4)
    n = 500
    state = (np.arange(n) // 60) % 2
    y = np.empty(n)
    y[0] = 1.0
    for t in range(n - 1):
        k = state[t + 1]
        y[t + 1] = abs((0.3, 2.0)[k] + (0.2, 0.5)[k] * y[t] + rng.normal(0, (0.05, 0.6)[k]))
    win = window(y)
    m = M.MsHar(n_starts=3, seed=1)
    assert m.fit(win, 1)
    params, scale, stress = m.est[1]
    names = M.MsHar._model(*m._pairs(win, 1, scale)).param_names
    sig = [params[names.index(f"sigma2[{k}]")] for k in (0, 1)]
    assert stress == int(np.argmax(sig)) and max(sig) / min(sig) > 10                   # the noisy regime is "stress"
    f = m.predict(win, 1)
    assert np.isfinite(f) and y.min() * 0.5 < f < y.max() * 1.5
    # the forecast uses only pairs whose target is known: changing nothing but the last regressor row changes it,
    # and a model without an estimate gives no forecast
    assert M.MsHar().predict(win, 1) is None
    again = M.MsHar(n_starts=3, seed=1)
    again.fit(win, 1)
    assert again.predict(win, 1) == pytest.approx(f)                                    # reproducible


def test_failed_estimation_keeps_the_previous_estimate_or_falls_back(monkeypatch):
    daily = make_daily(330)
    cfg = WFConfig(window=200, refit_every=60)
    good = run_walk_forward(daily, [M.Ewma(), M.Har(), M.Garch(), M.CalibratedGarch(), M.MsHar(1)], cfg, None)
    calls = {"n": 0}
    real = M.arch_model

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] > 2:                                    # the first refit (m2 and m2c) works, later ones fail
            raise RuntimeError("no convergence")
        return real(*a, **k)

    monkeypatch.setattr(M, "arch_model", flaky)
    res = run_walk_forward(daily, [M.Ewma(), M.Har(), M.Garch(), M.CalibratedGarch()], cfg, None)
    c = res.counters.set_index("model")
    assert c.loc["m2", "refits"] == 3 and c.loc["m2", "failures"] == 2 and c.loc["m2", "fallbacks"] == 0
    assert c.loc["m2c", "failures"] == 2 and c.loc["m2c", "fallbacks"] == 0
    first = slice(None, 60)                                   # until the second refit the two runs agree
    for name in ("m2", "m2c"):
        a, b = good.forecasts.xs(name, level="model")["raw"], res.forecasts.xs(name, level="model")["raw"]
        assert np.allclose(a.iloc[first], b.iloc[first]) and not np.allclose(a.iloc[60:], b.iloc[60:])

    monkeypatch.setattr(M, "arch_model", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("never")))
    monkeypatch.setattr(M.MarkovRegression, "fit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("never")))
    res = run_walk_forward(daily, [M.Ewma(), M.Har(), M.Garch(), M.CalibratedGarch(), M.MsHar(1)], cfg, None)
    fc = res.forecasts
    assert (fc.xs("m2", level="model")["source"] == "m1").all()
    assert (fc.xs("m2c", level="model")["source"] == "m3").all() and (fc.xs("m7", level="model")["source"] == "m3").all()
    assert np.allclose(fc.xs("m2", level="model")["forecast"], fc.xs("m1", level="model")["forecast"])
    assert np.allclose(fc.xs("m7", level="model")["forecast"], fc.xs("m3", level="model")["forecast"])
    c = res.counters.set_index("model")
    assert c.loc["m7", "failures"] == 3 and c.loc["m7", "fallbacks"] == len(fc.xs("m7", level="model"))


def test_no_model_looks_ahead():
    daily = make_daily(340, seed=7)
    cfg = WFConfig(window=200, refit_every=60, horizons=(1, 5))
    regime = RegimeConfig(n_init=2)
    cut = 261                                                # daily row 260 is a refit origin (the first row has no feature)
    changed = perturb_after(daily, cut)                      # every input changes, highs and lows independently
    a = run_walk_forward(daily, M.all_models(min_ess=20, ms_starts=1), cfg, regime)
    b = run_walk_forward(changed, M.all_models(min_ess=20, ms_starts=1), cfg, regime)
    last_clean = daily.index[cut - 1]
    assert a.regime.loc[last_clean, "refit"]
    fa = a.forecasts[a.forecasts["origin"] <= last_clean]
    fb = b.forecasts[b.forecasts["origin"] <= last_clean]
    assert set(fa.index.get_level_values("model")) == {"m0", "m1", "m2", "m2c", "m3", "m4", "m5", "m6", "m7"}
    assert len(fa) == 9 * 2 * 61
    pd.testing.assert_frame_equal(fa[["forecast", "raw", "source"]], fb[["forecast", "raw", "source"]])
    after = a.forecasts["origin"] > last_clean
    for name in ("m0", "m1", "m2", "m2c", "m3", "m4", "m5", "m6", "m7"):
        x = a.forecasts[after].xs(name, level="model")["raw"]
        y = b.forecasts[after].xs(name, level="model")["raw"]
        assert not np.allclose(x, y), name                   # the perturbation does reach every model afterwards
    # a sample that ends at the cut gives the same forecasts for the origins it contains
    short = run_walk_forward(daily.iloc[:cut], M.all_models(min_ess=20, ms_starts=1), cfg, regime)
    pd.testing.assert_frame_equal(short.forecasts[["forecast", "raw", "source"]],
                                  fa.loc[short.forecasts.index, ["forecast", "raw", "source"]])
