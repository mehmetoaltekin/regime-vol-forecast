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

Robustness variants (names with a suffix):
  m3_log, m5_log, m6_log, m7_log   the same models on log variance: log of the h-day sum on log regressors (the
                                   weekly and monthly terms average the daily logs); forecasts are converted back
                                   with exp(fit + s2 / 2), s2 the residual variance of the training window (per
                                   regime for m6 and m7)
  m3_q, m5_q                       HARQ and HARQ-X (Bollerslev, Patton and Quaedvlieg 2016): the daily term is
                                   multiplied by (b_d + b_q * sqrt(RQ_t))

With vintage training (WFConfig.vintage_training) m4, m5 and m6 are estimated on the regime probabilities as they
were computed on each training day; forecasts always use the probability of the origin.
"""
from __future__ import annotations

import warnings

import numpy as np
from arch import arch_model
from scipy.signal import lfilter
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from rvf.walkforward import Window, cumulative_target

HAR = ["har_d", "har_w", "har_m"]
LOG_HAR = ["lhar_d", "lhar_w", "lhar_m"]


def ols(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Least squares on the rows where X and y are finite; w are observation weights."""
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    if w is not None:
        sw = np.sqrt(w[ok])
        X, y = X * sw[:, None], y * sw
    return np.linalg.lstsq(X, y, rcond=None)[0]


def har_x(win: Window, extra: np.ndarray | None = None, log: bool = False, q: bool = False) -> np.ndarray:
    """
    Regressor matrix [1, daily, weekly, monthly (, sqrt(RQ) * daily) (, extra)] for every row of the window.
    log=True uses the log regressors; q=True adds the HARQ term (levels only).
    """
    cols = [np.ones(len(win.data)), *(win.data[c].to_numpy(dtype=float) for c in (LOG_HAR if log else HAR))]
    if q:
        if log:
            raise ValueError("HARQ is a levels model")
        cols.append(np.sqrt(win.data["rq"].to_numpy(dtype=float)) * win.data["har_d"].to_numpy(dtype=float))
    if extra is not None:
        cols.append(extra)
    return np.column_stack(cols)


def log_target(win: Window, h: int) -> np.ndarray:
    """Log of the h-day variance sum; NaN where the sum is not positive or runs past the window."""
    t = cumulative_target(win.y, h)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log(np.where(t > 0, t, np.nan))


def resid_var(X: np.ndarray, y: np.ndarray, beta: np.ndarray, w: np.ndarray | None = None) -> float:
    """Residual variance of a (weighted) regression on the rows where X and y are finite."""
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    e = y[ok] - X[ok] @ beta
    if w is None:
        return float(e @ e / max(ok.sum() - X.shape[1], 1))
    return float((w[ok] * e ** 2).sum() / w[ok].sum())


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

    def _x(self, win: Window, h: int, fit: bool = False) -> np.ndarray:
        cols = [np.ones(len(win.data)), self.garch.path(win, h)]
        if self.with_regime:
            cols.append(win.p_fit if fit else win.p)
        return np.column_stack(cols)

    def fit(self, win: Window, h: int) -> bool:
        if not self.garch.fit(win, h):
            return False                           # the previous refit's GARCH parameters and coefficients stay
        self.est[h] = ols(self._x(win, h, fit=True), cumulative_target(win.y, h))
        return True

    def predict(self, win: Window, h: int) -> float | None:
        if h not in self.est:
            return None
        return float(self._x(win, h)[-1] @ self.est[h])


class Har(_Base):
    """
    HAR-RV (m3), with prob_stress (m5), in logs (suffix _log) or as HARQ (suffix _q). est[h] is the coefficient
    vector for levels models and (coefficients, residual variance) for log models.
    """

    def __init__(self, with_regime: bool = False, log: bool = False, q: bool = False):
        super().__init__()
        if log and q:
            raise ValueError("HARQ is a levels model")
        self.with_regime, self.log, self.q = with_regime, log, q
        self.name = ("m5" if with_regime else "m3") + ("_log" if log else "") + ("_q" if q else "")

    def _x(self, win: Window, fit: bool = False) -> np.ndarray:
        extra = (win.p_fit if fit else win.p) if self.with_regime else None
        return har_x(win, extra, log=self.log, q=self.q)

    def fit(self, win: Window, h: int) -> bool:
        X = self._x(win, fit=True)
        if self.log:
            y = log_target(win, h)
            beta = ols(X, y)
            self.est[h] = (beta, resid_var(X, y, beta))
        else:
            self.est[h] = ols(X, cumulative_target(win.y, h))
        return True

    def predict(self, win: Window, h: int) -> float | None:
        x = self._x(win)[-1]
        if self.log:
            beta, s2 = self.est[h]
            return float(np.exp(x @ beta + 0.5 * s2))
        return float(x @ self.est[h])


class RegimeWeightedHar(_Base):
    """
    One HAR per regime (m6), in levels or logs (m6_log). est[h] has one coefficient row per regime for levels, and
    (coefficient rows, residual variance per regime) for logs.
    """

    def __init__(self, min_ess: float = 100.0, log: bool = False):
        super().__init__()
        self.min_ess, self.log = min_ess, log
        self.name = "m6_log" if log else "m6"

    def fit(self, win: Window, h: int) -> bool:
        X = har_x(win, log=self.log)
        y = log_target(win, h) if self.log else cumulative_target(win.y, h)
        W = win.pi_fit()
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        pooled = ols(X, y)
        pooled_s2 = resid_var(X, y, pooled) if self.log else None
        betas, s2s, fell_back = [], [], 0
        for k in range(W.shape[1]):
            w = W[ok, k]
            ess = w.sum() ** 2 / (w ** 2).sum() if w.sum() > 0 else 0.0
            if ess < self.min_ess:
                betas.append(pooled)
                s2s.append(pooled_s2)
                fell_back += 1
            else:
                b = ols(X, y, W[:, k])
                betas.append(b)
                s2s.append(resid_var(X, y, b, W[:, k]) if self.log else None)
        self.est[h] = (np.array(betas), np.array(s2s)) if self.log else np.array(betas)
        self.extra_counts[(h, "ess_fallback_refits")] = self.extra_counts.get((h, "ess_fallback_refits"), 0) + (fell_back > 0)
        self.extra_counts[(h, "ess_fallback_regimes")] = self.extra_counts.get((h, "ess_fallback_regimes"), 0) + fell_back
        return True

    def predict(self, win: Window, h: int) -> float | None:
        x, pi = har_x(win, log=self.log)[-1], win.pi(1)[-1]
        if self.log:
            betas, s2s = self.est[h]
            return float(pi @ np.exp(betas @ x + 0.5 * s2s))
        return float(pi @ (self.est[h] @ x))


class MsHar(_Base):
    """
    Markov-switching HAR (m7), in levels or logs (m7_log): switching intercept, slopes and variance, two regimes.
    Each refit runs `n_starts` random starting values plus the previous refit's estimate and keeps the converged
    fit with the highest likelihood. The forecast mixes the regime regressions with predicted probabilities built
    from filtered probabilities only. In levels, target and regressors are divided by the training window's mean
    variance for the optimiser; in logs they are used as they are, and each regime's forecast is
    exp(fit + sigma2_k / 2). est[h] = (parameters, scale, stress regime).
    """
    name, fallback = "m7", "m3"

    def __init__(self, n_starts: int = 10, seed: int = 0, maxiter: int = 200, log: bool = False):
        super().__init__()
        self.n_starts, self.seed, self.maxiter, self.log = n_starts, seed, maxiter, log
        self.n_fits = 0
        if log:
            self.name, self.fallback = "m7_log", "m3_log"

    @staticmethod
    def _model(y: np.ndarray, X: np.ndarray) -> MarkovRegression:
        return MarkovRegression(y, k_regimes=2, trend="c", exog=X, switching_trend=True, switching_exog=True,
                                switching_variance=True)

    def _pairs(self, win: Window, h: int, scale: float):
        X = har_x(win, log=self.log)[:, 1:] / scale
        y = (log_target(win, h) if self.log else cumulative_target(win.y, h)) / scale
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        return y[ok], X[ok]

    def fit(self, win: Window, h: int) -> bool:
        scale = 1.0 if self.log else float(np.nanmean(win.y))
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
        x = har_x(win, log=self.log)[-1, 1:] / scale
        names = mod.param_names
        f = 0.0
        for k in (0, 1):
            const = params[names.index(f"const[{k}]")]
            beta = np.array([params[names.index(f"x{j}[{k}]")] for j in (1, 2, 3)])
            mean = const + beta @ x
            f += prob[k] * (np.exp(mean + 0.5 * params[names.index(f"sigma2[{k}]")]) if self.log else mean)
        return float(f if self.log else f * scale)


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


def log_models(min_ess: float = 100.0, ms_starts: int = 10) -> list:
    """Robustness check 4: log-HAR in place of levels HAR for M3, M5, M6 and M7."""
    return [Har(log=True), Har(with_regime=True, log=True), RegimeWeightedHar(min_ess, log=True), MsHar(ms_starts, log=True)]


def harq_models() -> list:
    """Robustness check 10: HARQ against HARQ plus prob_stress."""
    return [Har(q=True), Har(with_regime=True, q=True)]
