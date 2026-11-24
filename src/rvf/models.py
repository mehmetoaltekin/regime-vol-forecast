"""
Forecast models behind one interface (see rvf.walkforward.Model).

Every model forecasts the summed variance of days t+1 .. t+h from a `Window` that ends at day t. HAR-type models and
the calibrated GARCH models use a direct regression of the h-day sum on regressors of day t; EWMA, GARCH and the
pure HMM forecast iterate their own recursion. All estimates are kept per horizon.

  m0   pure HMM          sum over regimes of P(s_{t+j} = k | F_t) * sigma2_k                      reference only
  m1   EWMA              lambda = 0.94, started at the sample variance of the training window
  m2   GARCH(1,1)        constant mean, normal QMLE (arch)
  m2c  calibrated GARCH  RV on the GARCH forecast, OLS
  m3   HAR-RV            OLS in levels on the daily value and its 5- and 22-day averages
  m4   GARCH-X           m2c plus prob_stress
  m5   HAR-X             m3 plus prob_stress
  m6   regime-weighted   one HAR per regime by weighted least squares; weights and forecast mix are the
       HAR               one-step-ahead regime probabilities; a regime with too small an effective sample uses m3
  m7   MS-HAR            two-regime Markov-switching regression (statsmodels), filtered probabilities only
"""
from __future__ import annotations

import warnings

import numpy as np
from arch import arch_model
from scipy.signal import lfilter
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from rvf.walkforward import Window, cumulative_target

HAR = ["har_d", "har_w", "har_m"]


def ols(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Least squares on the rows where X and y are finite; w are observation weights."""
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    if w is not None:
        sw = np.sqrt(w[ok])
        X, y = X * sw[:, None], y * sw
    return np.linalg.lstsq(X, y, rcond=None)[0]


def har_x(win: Window, extra: np.ndarray | None = None) -> np.ndarray:
    """Regressor matrix [1, daily, weekly, monthly (, extra)] for every row of the window."""
    cols = [np.ones(len(win.data)), *(win.data[c].to_numpy(dtype=float) for c in HAR)]
    if extra is not None:
        cols.append(extra)
    return np.column_stack(cols)


class _Base:
    name = ""
    fallback: str | None = None

    def __init__(self):
        self.est: dict = {}                    # estimates per horizon
        self.extra_counts: dict = {}


class Ewma(_Base):
    name = "m1"

    def __init__(self, lam: float = 0.94):
        super().__init__()
        self.lam = lam

    def fit(self, win: Window, h: int) -> bool:
        self.est[h] = float(np.var(win.data["r_cc"].to_numpy(dtype=float), ddof=1))
        return True

    def predict(self, win: Window, h: int) -> float | None:
        r2 = win.data["r_cc"].to_numpy(dtype=float) ** 2
        n = len(r2)
        decay = self.lam ** np.arange(n - 1, -1, -1)
        return h * float(self.lam ** n * self.est[h] + (1 - self.lam) * decay @ r2)


class Garch(_Base):
    """GARCH(1,1) with constant mean. est[h] = (mu, omega, alpha, beta, start variance)."""
    name, fallback = "m2", "m1"

    def fit(self, win: Window, h: int) -> bool:
        r = win.data["r_cc"].to_numpy(dtype=float)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(
                    disp="off", show_warning=False)
            mu, omega, alpha, beta = (float(v) for v in res.params)
            ok = res.convergence_flag == 0 and np.isfinite([mu, omega, alpha, beta]).all() and omega > 0
        except Exception:
            ok = False
        if ok:
            self.est[h] = (mu, omega, alpha, beta, float(np.var(r - mu)))
        return ok

    def path(self, win: Window, h: int) -> np.ndarray | None:
        """For every row s: the forecast, made at s, of the summed variance of days s+1 .. s+h."""
        if h not in self.est:
            return None
        mu, omega, alpha, beta, s0 = self.est[h]
        e2 = (win.data["r_cc"].to_numpy(dtype=float) - mu) ** 2
        step = lfilter([1.0], [1.0, -beta], omega + alpha * e2, zi=[beta * s0])[0]     # sigma2_{s+1 | s}
        total, nxt = step.copy(), step
        for _ in range(h - 1):
            nxt = omega + (alpha + beta) * nxt
            total = total + nxt
        return total

    def predict(self, win: Window, h: int) -> float | None:
        p = self.path(win, h)
        return None if p is None else float(p[-1])


class CalibratedGarch(_Base):
    """RV regressed on the GARCH forecast (m2c), optionally with prob_stress (m4)."""
    fallback = "m3"

    def __init__(self, with_regime: bool = False):
        super().__init__()
        self.with_regime = with_regime
        self.name = "m4" if with_regime else "m2c"
        self.garch = Garch()

    def _x(self, win: Window, h: int) -> np.ndarray:
        cols = [np.ones(len(win.data)), self.garch.path(win, h)]
        if self.with_regime:
            cols.append(win.p)
        return np.column_stack(cols)

    def fit(self, win: Window, h: int) -> bool:
        if not self.garch.fit(win, h):
            return False                           # the previous refit's GARCH parameters and coefficients stay
        self.est[h] = ols(self._x(win, h), cumulative_target(win.y, h))
        return True

    def predict(self, win: Window, h: int) -> float | None:
        if h not in self.est:
            return None
        return float(self._x(win, h)[-1] @ self.est[h])


class Har(_Base):
    """HAR-RV (m3), optionally with prob_stress (m5)."""

    def __init__(self, with_regime: bool = False):
        super().__init__()
        self.with_regime = with_regime
        self.name = "m5" if with_regime else "m3"

    def _x(self, win: Window) -> np.ndarray:
        return har_x(win, win.p if self.with_regime else None)

    def fit(self, win: Window, h: int) -> bool:
        self.est[h] = ols(self._x(win), cumulative_target(win.y, h))
        return True

    def predict(self, win: Window, h: int) -> float | None:
        return float(self._x(win)[-1] @ self.est[h])


class RegimeWeightedHar(_Base):
    """One HAR per regime (m6). est[h] has one coefficient row per regime."""
    name = "m6"

    def __init__(self, min_ess: float = 100.0):
        super().__init__()
        self.min_ess = min_ess

    def fit(self, win: Window, h: int) -> bool:
        X, y, W = har_x(win), cumulative_target(win.y, h), win.pi(1)
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        pooled = ols(X, y)
        betas, fell_back = [], 0
        for k in range(W.shape[1]):
            w = W[ok, k]
            ess = w.sum() ** 2 / (w ** 2).sum() if w.sum() > 0 else 0.0
            if ess < self.min_ess:
                betas.append(pooled)
                fell_back += 1
            else:
                betas.append(ols(X, y, W[:, k]))
        self.est[h] = np.array(betas)
        self.extra_counts[(h, "ess_fallback_refits")] = self.extra_counts.get((h, "ess_fallback_refits"), 0) + (fell_back > 0)
        self.extra_counts[(h, "ess_fallback_regimes")] = self.extra_counts.get((h, "ess_fallback_regimes"), 0) + fell_back
        return True

    def predict(self, win: Window, h: int) -> float | None:
        return float(win.pi(1)[-1] @ (self.est[h] @ har_x(win)[-1]))


class MsHar(_Base):
    """
    Markov-switching HAR (m7): switching intercept, slopes and variance, two regimes. Each refit runs `n_starts`
    random starting values plus the previous refit's estimate and keeps the converged fit with the highest
    likelihood. The forecast mixes the regime regressions with predicted probabilities built from filtered
    probabilities only. Target and regressors are divided by the training window's mean variance for the optimiser.
    est[h] = (parameters, scale, stress regime).
    """
    name, fallback = "m7", "m3"

    def __init__(self, n_starts: int = 10, seed: int = 0, maxiter: int = 200):
        super().__init__()
        self.n_starts, self.seed, self.maxiter = n_starts, seed, maxiter
        self.n_fits = 0

    @staticmethod
    def _model(y: np.ndarray, X: np.ndarray) -> MarkovRegression:
        return MarkovRegression(y, k_regimes=2, trend="c", exog=X, switching_trend=True, switching_exog=True,
                                switching_variance=True)

    def _pairs(self, win: Window, h: int, scale: float):
        X, y = har_x(win)[:, 1:] / scale, cumulative_target(win.y, h) / scale
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        return y[ok], X[ok]

    def fit(self, win: Window, h: int) -> bool:
        scale = float(np.nanmean(win.y))
        y, X = self._pairs(win, h, scale)
        mod = self._model(y, X)
        rng = np.random.default_rng([self.seed, h, self.n_fits])
        self.n_fits += 1
        base = np.asarray(mod.start_params, dtype=float)
        names = mod.param_names
        starts = []
        for _ in range(self.n_starts):
            s = base.copy()
            for i, nm in enumerate(names):
                if nm.startswith("p["):
                    s[i] = rng.uniform(0.02, 0.4) if nm in ("p[1->0]",) else rng.uniform(0.6, 0.98)
                elif nm.startswith("sigma2"):
                    s[i] = base[i] * np.exp(rng.normal(0, 0.7))
                else:
                    s[i] = base[i] + rng.normal(0, 0.3)
            starts.append(s)
        if h in self.est:
            starts.append(self.est[h][0])
        best = None
        for s in starts:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = mod.fit(start_params=s, disp=False, maxiter=self.maxiter)
                good = bool(res.mle_retvals.get("converged", True)) and np.isfinite(res.llf) and np.isfinite(res.params).all()
            except Exception:
                good = False
            if good and (best is None or res.llf > best.llf):
                best = res
        if best is None:
            return False
        sig = [best.params[names.index(f"sigma2[{k}]")] for k in (0, 1)]
        self.est[h] = (np.asarray(best.params, dtype=float), scale, int(np.argmax(sig)))
        return True

    def predict(self, win: Window, h: int) -> float | None:
        if h not in self.est:
            return None
        params, scale, _ = self.est[h]
        y, X = self._pairs(win, h, scale)            # pairs whose h-day target is known at the origin
        mod = self._model(y, X)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = mod.filter(params)
            filt = np.asarray(res.filtered_marginal_probabilities)[-1]
            P = np.asarray(res.regime_transition)[:, :, 0]              # P[i, j] = P(regime i now | regime j before)
        except Exception:
            return None
        prob = np.linalg.matrix_power(P, h) @ filt                      # regime of the pair whose regressors are day t
        x = har_x(win)[-1, 1:] / scale
        names = mod.param_names
        f = 0.0
        for k in (0, 1):
            const = params[names.index(f"const[{k}]")]
            beta = np.array([params[names.index(f"x{j}[{k}]")] for j in (1, 2, 3)])
            f += prob[k] * (const + beta @ x)
        return float(f * scale)


class PureHmm(_Base):
    """Reference forecast from the HMM alone (m0)."""
    name = "m0"

    def fit(self, win: Window, h: int) -> bool:
        return win.state_var is not None

    def predict(self, win: Window, h: int) -> float | None:
        if win.state_var is None:
            return None
        prob, total = win.filt[-1], 0.0
        for _ in range(h):
            prob = prob @ win.A
            total += float(prob @ win.state_var)
        return total


def all_models(min_ess: float = 100.0, ms_starts: int = 10) -> list:
    """M0-M7 and M2c in the pre-specified set-up."""
    return [PureHmm(), Ewma(), Garch(), CalibratedGarch(), Har(), CalibratedGarch(with_regime=True), Har(with_regime=True),
            RegimeWeightedHar(min_ess), MsHar(ms_starts)]
