"""
Forecast evaluation: loss functions, Diebold-Mariano tests (per instrument and pooled), Holm correction,
Model Confidence Set, Giacomini-White conditional test and loss splits.

A loss differential is always d = loss(model A) - loss(model B): a negative mean says A is better.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch.bootstrap import MCS
from scipy import stats


def qlike(rv, f):
    """QLIKE loss RV/f - log(RV/f) - 1 (Patton 2011); zero when the forecast equals the outcome."""
    ratio = np.asarray(rv, dtype=float) / np.asarray(f, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):      # a non-positive ratio gives NaN or inf; callers check
        return ratio - np.log(ratio) - 1.0


def mse(rv, f):
    return (np.asarray(rv, dtype=float) - np.asarray(f, dtype=float)) ** 2


LOSSES = {"qlike": qlike, "mse": mse}


def loss_table(forecasts: pd.DataFrame, h: int = 1, loss: str = "qlike") -> pd.DataFrame:
    """Walk-forward forecasts -> losses, one row per target day and one column per model, on the common sample."""
    fc = forecasts.xs(h, level="h")
    fc = fc[fc["target"].notna()]
    out = pd.Series(LOSSES[loss](fc["target"], fc["forecast"]), index=fc.index).unstack("model")
    bad = ~np.isfinite(out.to_numpy())
    if bad.any():
        raise ValueError(f"{int(bad.sum())} non-finite {loss} losses (forecast or target not positive, or a model "
                         "without a forecast for a day)")
    return out


def at_origin(forecasts: pd.DataFrame, by_origin: pd.Series, h: int = 1) -> pd.Series:
    """
    A series indexed by forecast origin (for example WFResult.regime["prob_stress"]) -> the same values indexed by
    the target day of the horizon-h forecasts made at that origin. Use this before `loss_splits` and `gw_test`:
    conditioning variables must be the ones known when the forecast was made.
    """
    fc = forecasts.xs(h, level="h")
    fc = fc[fc["target"].notna()]
    origin = fc.groupby(level="target_day")["origin"].first()
    out = pd.Series(by_origin.reindex(origin.to_numpy()).to_numpy(), index=origin.index, name=by_origin.name)
    if out.isna().any():
        raise ValueError("no value for some forecast origins")
    return out


def default_lag(h: int) -> int:
    return max(h - 1, 5)


def long_run_variance(d: np.ndarray, lag: int) -> float:
    """Newey-West (Bartlett kernel) long-run variance of d, autocovariances divided by n."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    e = d - d.mean()
    lrv = e @ e / n
    for j in range(1, min(lag, n - 1) + 1):
        lrv += 2.0 * (1.0 - j / (lag + 1.0)) * (e[j:] @ e[:-j]) / n
    return float(lrv)


def dm_test(d, h: int = 1, lag: int | None = None, hln: bool = True, alternative: str = "two-sided") -> dict:
    """
    Diebold-Mariano test of E[d] = 0 with Newey-West HAC variance (default lag max(h - 1, 5)).
    hln=True applies the Harvey-Leybourne-Newbold small-sample factor and Student-t(n - 1) critical values.
    alternative: "two-sided", "less" (mean of d below zero) or "greater".
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    lag = default_lag(h) if lag is None else lag
    lrv = long_run_variance(d, lag)
    out = {"mean": float(d.mean()), "n": n, "lag": lag, "stat": np.nan, "pvalue": np.nan}
    if n < 3 or not lrv > 0:
        return out
    stat = d.mean() / np.sqrt(lrv / n)
    if hln:
        stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        dist = stats.t(df=n - 1)
    else:
        dist = stats.norm()
    if alternative == "two-sided":
        p = 2 * dist.sf(abs(stat))
    elif alternative == "less":
        p = dist.cdf(stat)
    elif alternative == "greater":
        p = dist.sf(stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'less' or 'greater'")
    out.update(stat=float(stat), pvalue=float(p))
    return out


def pooled_dm(d: pd.DataFrame, h: int = 1, alternative: str = "two-sided") -> dict:
    """
    Pooled test: each day's loss differentials are averaged over the instruments that have one that day, and a
    Diebold-Mariano test (Newey-West, normal critical values) is applied to the averaged series.
    d: one row per day, one column per instrument; missing values where an instrument has no forecast.
    """
    avg = d.mean(axis=1, skipna=True).dropna()
    out = dm_test(avg.to_numpy(), h=h, hln=False, alternative=alternative)
    out["instruments_per_day"] = float(d.loc[avg.index].notna().sum(axis=1).mean())
    return out


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values, in the order given."""
    p = np.asarray(pvalues, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("p-values must be finite")
    m = len(p)
    order = np.argsort(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def model_confidence_set(losses: pd.DataFrame, size: float = 0.10, reps: int = 10_000, block: int = 10,
                         seed: int = 0) -> pd.DataFrame:
    """Hansen-Lunde-Nason MCS (range statistic, stationary bootstrap). Returns MCS p-values and inclusion per model."""
    mcs = MCS(losses, size=size, reps=reps, block_size=block, method="R", bootstrap="stationary", seed=seed)
    mcs.compute()
    out = mcs.pvalues.rename(columns={"Pvalue": "mcs_pvalue"})
    out["included"] = out.index.isin(mcs.included)
    return out.reindex(losses.columns)


def gw_test(d, instruments, h: int = 1) -> dict:
    """
    Giacomini-White (2006) conditional predictive ability test. d[t] is the loss differential of the forecast made
    at origin t; instruments[t] are variables known at t (a constant is not added). Statistic n * zbar' V^-1 zbar,
    chi-squared with q = number of instruments; V is the sample second moment of z, with a Bartlett HAC of h - 1
    lags when h > 1.
    """
    d = np.asarray(d, dtype=float)
    H = np.asarray(instruments, dtype=float)
    H = H.reshape(len(d), -1)
    ok = np.isfinite(d) & np.isfinite(H).all(axis=1)
    Z = H[ok] * d[ok][:, None]
    n, q = Z.shape
    zbar = Z.mean(axis=0)
    V = Z.T @ Z / n
    for j in range(1, h):
        G = Z[j:].T @ Z[:-j] / n
        V += (1.0 - j / h) * (G + G.T)
    stat = float(n * zbar @ np.linalg.solve(V, zbar))
    return {"stat": stat, "pvalue": float(stats.chi2(q).sf(stat)), "q": q, "n": n}


def regime_masks(p: pd.Series, threshold: float = 0.5, halfwidth: int = 10) -> pd.DataFrame:
    """
    p: prob_stress known at the forecast origin, one value per forecast in time order (see `at_origin`).
    stress: p > threshold; calm: p <= threshold; transition: within `halfwidth` forecasts of a crossing of the
    threshold (the crossing is the first forecast on the new side).
    """
    if p.isna().any():
        raise ValueError("prob_stress has missing values")
    stress = (p > threshold).to_numpy()
    cross = np.flatnonzero(stress[1:] != stress[:-1]) + 1
    near = np.zeros(len(p), dtype=bool)
    for c in cross:
        near[max(0, c - halfwidth): c + halfwidth + 1] = True
    return pd.DataFrame({"stress": stress, "calm": ~stress, "transition": near}, index=p.index)


def loss_splits(losses: pd.DataFrame, p: pd.Series, threshold: float = 0.5, halfwidth: int = 10) -> pd.DataFrame:
    """
    Mean loss per model for all days, stress days, calm days, transition windows and each calendar year.
    p must be indexed by target day and hold prob_stress of the forecast origin: build it with `at_origin`.
    """
    if p.index.name != "target_day" or not p.index.equals(losses.index):
        raise ValueError("p must come from at_origin() and cover exactly the days of the loss table")
    masks = regime_masks(p, threshold, halfwidth)
    parts = {"all": np.ones(len(losses), dtype=bool), **{c: masks[c].to_numpy() for c in masks}}
    for y in sorted(set(losses.index.year)):
        parts[str(y)] = losses.index.year == y
    rows = []
    for name, m in parts.items():
        rows.append({"split": name, "n": int(m.sum()), **losses[m].mean().to_dict()})
    return pd.DataFrame(rows).set_index("split")
