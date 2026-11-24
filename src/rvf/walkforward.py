"""
Walk-forward engine: one rolling training window and one refit calendar shared by the HMM and all models.

Timing. `origin` t is the last day whose data a forecast may use. At a refit origin every model is estimated on the
`window` days ending at t. At every origin each model forecasts the variance of days t+1 .. t+h from data up to t.
Models never receive rows after the origin: they get a `Window` that ends at t.

Regime probabilities. At a refit the HMM is calibrated on the training window and the forward filter is run over
that window with the new parameters. Between refits the parameters stay fixed and the filter moves on one day at a
time. The probability of each origin, as it was computed on that day, is stored as the real-time vintage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from rvf.regime import RegimeCalibration, RegimeConfig, ahead, hmm_features, prob_stress


@dataclass(frozen=True)
class WFConfig:
    window: int = 1000
    refit_every: int = 21
    horizons: tuple[int, ...] = (1,)
    target: str = "rv"                       # column holding the daily variance measure
    insanity_filter: bool = True


@dataclass
class Window:
    """What a model may see: daily rows up to the origin and the regime probabilities under the current calibration."""
    data: pd.DataFrame                        # rows from the start of the current training window to the origin
    n_train: int                              # the first n_train rows are the training window of the last refit
    filt: np.ndarray | None = None            # filtered regime probabilities, shape (len(data), K)
    A: np.ndarray | None = None               # regime transition matrix
    stress: np.ndarray | None = None          # boolean mask of the stress states
    state_var: np.ndarray | None = None       # return variance of each state, percent squared
    target: str = "rv"

    @property
    def train(self) -> "Window":
        n = self.n_train
        return Window(self.data.iloc[:n], n, None if self.filt is None else self.filt[:n], self.A, self.stress,
                      self.state_var, self.target)

    @property
    def y(self) -> np.ndarray:
        return self.data[self.target].to_numpy(dtype=float)

    @property
    def p(self) -> np.ndarray:
        """prob_stress per row."""
        return prob_stress(self.filt, self.stress)

    def pi(self, h: int = 1) -> np.ndarray:
        """h-step-ahead regime probabilities per row, shape (len(data), K)."""
        return ahead(self.filt, self.A, h)


class Model(Protocol):
    name: str
    fallback: str | None                      # model whose forecast is used while this one has no usable estimate

    def fit(self, win: Window, h: int) -> bool:
        """Estimate on win (the training window). Return False if estimation failed; keep the previous estimate then."""

    def predict(self, win: Window, h: int) -> float | None:
        """Forecast of the summed variance of the h days after the last row of win; None if there is no estimate."""


def cumulative_target(y: np.ndarray, h: int) -> np.ndarray:
    """out[s] = y[s+1] + ... + y[s+h]; NaN where the sum runs past the end."""
    c = np.concatenate([[0.0], np.cumsum(y)])
    out = np.full(len(y), np.nan)
    if len(y) > h:
        out[: len(y) - h] = c[h + 1:] - c[1: len(y) - h + 1]
    return out


def refit_origins(n: int, cfg: WFConfig) -> np.ndarray:
    """Row positions of the refit origins: the last day of the first window, then every `refit_every` days."""
    return np.arange(cfg.window - 1, n - 1, cfg.refit_every)


@dataclass
class WFResult:
    forecasts: pd.DataFrame                   # index (target_day, model); columns h, origin, forecast, raw, target, source
    regime: pd.DataFrame                      # per origin: real-time prob_stress, K, refit flag
    counters: pd.DataFrame                    # per model and horizon: refits, failures, fallbacks, insanity replacements
    details: dict = field(default_factory=dict)


def run_walk_forward(daily: pd.DataFrame, models: list, cfg: WFConfig = WFConfig(),
                     regime_cfg: RegimeConfig | None = RegimeConfig()) -> WFResult:
    """
    daily: retained trading days (rvf.realized.retained) with at least open/high/low/close and the target column.
    regime_cfg=None runs without an HMM (models must not ask for regime probabilities).
    """
    if regime_cfg is not None:
        feats = hmm_features(daily, regime_cfg)
        data = daily.loc[feats.index]
        X = feats.to_numpy()
    else:
        data, X = daily, None
    n = len(data)
    if n <= cfg.window:
        raise ValueError(f"{n} days are not enough for a window of {cfg.window}")
    y = data[cfg.target].to_numpy(dtype=float)
    refits = set(refit_origins(n, cfg).tolist())
    names = [m.name for m in models]
    by_name = dict(zip(names, models))
    count = {(m, h): dict(refits=0, failures=0, fallbacks=0, insane=0) for m in names for h in cfg.horizons}
    rows, regime_rows, vintage = [], [], {}
    cal, filt, anchor, bounds = None, None, 0, {}

    for t in range(cfg.window - 1, n - 1):
        refit = t in refits
        if refit:
            anchor = t - cfg.window + 1
            if regime_cfg is not None:
                cal = RegimeCalibration(regime_cfg).fit(X[anchor: t + 1])
                filt = cal.filter(X[anchor: t + 1])
        elif regime_cfg is not None:
            filt = np.vstack([filt, cal.filter(X[t: t + 1], prior=filt[-1])])
        win = Window(data.iloc[anchor: t + 1], cfg.window, filt, *(
            (cal.A, cal.stress, cal.state_var) if cal is not None else (None, None, None)), cfg.target)
        if regime_cfg is not None:
            regime_rows.append((data.index[t], float(win.p[-1]), cal.k, refit))
            vintage[data.index[t]] = filt[-1].copy()            # filtered vector as computed on day t

        for h in cfg.horizons:
            if refit:
                # insanity bounds: the h-day variance sums inside the training window (for h = 1 the daily values)
                yh = np.convolve(win.train.y, np.ones(h), "valid")
                bounds[h] = (yh.min(), yh.max(), yh.mean())
                for m in models:
                    count[(m.name, h)]["refits"] += 1
                    if not m.fit(win.train, h):
                        count[(m.name, h)]["failures"] += 1
            raw = {m.name: m.predict(win, h) for m in models}
            target = y[t + 1: t + 1 + h].sum() if t + h < n else np.nan
            for m in models:
                f, source = raw[m.name], m.name
                if f is None or not np.isfinite(f):
                    source = m.fallback
                    f = raw.get(source) if source is not None else None
                    count[(m.name, h)]["fallbacks"] += 1
                    if f is None or not np.isfinite(f):
                        raise RuntimeError(f"{m.name}: no forecast and no usable fallback at {data.index[t].date()}")
                final = f
                if cfg.insanity_filter and not (bounds[h][0] <= f <= bounds[h][1]):
                    final = bounds[h][2]
                    count[(m.name, h)]["insane"] += 1
                rows.append((data.index[t + 1], m.name, h, data.index[t], final, f, target, source))

    fc = pd.DataFrame(rows, columns=["target_day", "model", "h", "origin", "forecast", "raw", "target", "source"])
    regime = pd.DataFrame(regime_rows, columns=["origin", "prob_stress", "k", "refit"]).set_index("origin")
    counters = pd.DataFrame([dict(model=m, h=h, **c) for (m, h), c in count.items()])
    return WFResult(fc.set_index(["target_day", "model", "h"]).sort_index(), regime, counters, {"vintage_filt": vintage})
