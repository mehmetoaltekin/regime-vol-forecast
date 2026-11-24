"""
Causal HMM regime series with hmm-regime (import hmm_regime), on daily bars.

One `RegimeCalibration` is one fit on a training window. Inference is the forward filter only: the probabilities
of day s use data up to s. Features are computed bar by bar from current and past rows, so the feature of day s
does not change when later days change.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmm_regime import MarketRegimeHMM, select_model
from hmm_regime.data_loader import compute_log_range_variance, compute_parkinson_volatility

logging.getLogger("hmm_regime").setLevel(logging.ERROR)


@dataclass(frozen=True)
class RegimeConfig:
    n_components: int | tuple[int, ...] = 2      # a tuple selects K by BIC on each training window
    vol_feature: str = "bar"                     # "bar": per-bar log range variance; "rolling": rolling Parkinson volatility
    vol_window: int = 21                         # for vol_feature="rolling"
    feature_cols: tuple[str, ...] | None = None  # overrides vol_feature: any columns of `hmm_features`, sorted by column 0
    sort_by: str = "variance:0"
    n_init: int = 10
    random_state: int = 42

    @property
    def columns(self) -> tuple[str, ...]:
        if self.feature_cols is not None:
            return self.feature_cols
        if self.vol_feature not in ("bar", "rolling"):
            raise ValueError("vol_feature must be 'bar' or 'rolling'")
        return ("log_return", "log_range_var" if self.vol_feature == "bar" else "volatility")


def hmm_features(daily: pd.DataFrame, cfg: RegimeConfig = RegimeConfig()) -> pd.DataFrame:
    """
    Candidate HMM features per day, as the library defines them: log_return (close to close, decimal),
    log_range_var (per-bar log Parkinson variance), volatility (rolling annualised Parkinson), and log_rv
    (log realised variance, for the information-matched check). Rows with a missing selected feature are dropped.

    log_return is the daily return r_cc / 100 when the frame has one: it equals log(close / previous close) on every day
    except the first retained day after an anchor day (PRESPEC 12.6), whose return is measured from the anchor's close and
    not across the whole gap, as for RV, EWMA and GARCH. The first row has no previous close and is dropped either way.
    """
    f = pd.DataFrame(index=daily.index)
    previous = daily["close"].shift(1)
    f["log_return"] = (daily["r_cc"] / 100).where(previous.notna()) if "r_cc" in daily else np.log(daily["close"] / previous)
    f["log_range_var"] = compute_log_range_variance(daily["high"], daily["low"])
    f["volatility"] = compute_parkinson_volatility(daily["high"], daily["low"], window=cfg.vol_window, periods_per_year=252)
    if "rv" in daily:
        f["log_rv"] = np.log(daily["rv"].where(daily["rv"] > 0))
    return f[list(cfg.columns)].dropna()


class RegimeCalibration:
    """A fitted HMM: transition matrix, stress states and forward filter."""

    def __init__(self, cfg: RegimeConfig = RegimeConfig()):
        self.cfg = cfg
        self.model: MarketRegimeHMM | None = None

    def fit(self, X: np.ndarray) -> "RegimeCalibration":
        ks = self.cfg.n_components if isinstance(self.cfg.n_components, tuple) else (self.cfg.n_components,)
        kw = dict(n_init=self.cfg.n_init, random_state=self.cfg.random_state, sort_by=self.cfg.sort_by)
        if len(ks) > 1:
            self.model = select_model(X, ks, criterion="bic", _quiet=True, **kw)
        else:
            self.model = MarketRegimeHMM(n_components=ks[0], **kw).fit(X)
        return self

    @property
    def k(self) -> int:
        return self.model.n_components

    @property
    def A(self) -> np.ndarray:
        return self.model.transition_matrix_.to_numpy()

    @property
    def stress(self) -> np.ndarray:
        return np.asarray(self.model.stress_states_, dtype=bool)

    @property
    def state_var(self) -> np.ndarray | None:
        """Return variance of each state in percent squared; None unless feature 0 is the log return."""
        if self.cfg.columns[0] != "log_return":
            return None
        return self.model.covars_[:, 0, 0] * 1e4

    def diagnostics(self, X: np.ndarray, filt: np.ndarray) -> dict[str, float | int]:
        """
        What the fit looks like, so that the label of the stress state can be audited after the run: the number of stress
        states, the share of the training window that the filter puts in them, the log-likelihood, and for every feature
        `sep_<feature>`: the mean of the stress states minus the mean of the other states (both weighted by the window
        occupancy of the states), in standard deviations of that feature over the window. A stress state that differs
        from the others mainly in the return and not in the range is a split on the sign of the return, not a volatility regime.
        """
        stress, occupancy = self.stress, filt.mean(axis=0)
        out = {"n_stress": int(stress.sum()), "stress_occupancy": float(occupancy[stress].sum()),
               "hmm_loglik": float(self.model.log_likelihood_)}
        sd = X.std(axis=0, ddof=1)
        means = np.asarray(self.model.means_, dtype=float)
        for j, name in enumerate(self.cfg.columns):
            weights = (occupancy[stress].sum(), occupancy[~stress].sum())
            if not all(w > 0 for w in weights) or not sd[j] > 0:
                out[f"sep_{name}"] = np.nan
                continue
            hot = occupancy[stress] @ means[stress, j] / weights[0]
            cold = occupancy[~stress] @ means[~stress, j] / weights[1]
            out[f"sep_{name}"] = float((hot - cold) / sd[j])
        return out

    def filter(self, X: np.ndarray, prior: np.ndarray | None = None) -> np.ndarray:
        """Filtered probabilities P(s_t | data up to t), shape (n, K). `prior` is the filtered vector of the bar before X[0]."""
        return self.model.predict_filtered_proba(X, prior_proba=prior)


def prob_stress(filt: np.ndarray, stress: np.ndarray) -> np.ndarray:
    return filt[:, stress].sum(axis=1)


def ahead(filt: np.ndarray, A: np.ndarray, h: int = 1) -> np.ndarray:
    """h-step-ahead regime probabilities P(s_{t+h} | data up to t) = filtered probabilities x A^h."""
    return filt @ np.linalg.matrix_power(A, h)
