"""
Daily realised measures from M1 bars.

Definitions (server time; returns in percent, variances in percent squared):

  trading day     One server calendar day. Sunday bars are counted with the following Monday (merge_sunday).
  grid            Regular grid of `interval_min` minutes with previous-tick sampling. An M1 bar stamped HH:MM
                  closes at HH:MM + 1 minute; the price at a grid point is the last M1 close at or before it.
                  The day's last grid point carries the day's last included close.
  excluded        Minute-of-day windows whose bars are left out of the grid. The day is cut into segments at
  windows         these windows; grid returns are taken inside segments only.
  session-break   One return per day that collects everything the grid returns do not cover: the move from the
  return          previous retained day's last close to the day's first grid price, and the price change across
                  excluded windows. By construction: sum of grid returns + session-break return = close-to-close
                  return, where the close is the last M1 close of the server day (all bars).
  RV              Sum of squared grid returns, plus the squared session-break return if include_session_break.
  RQ              Realised quarticity (n / 3) * sum of fourth powers of the n grid returns.
  short day       A day whose number of grid returns is below short_day_share times the median day length. Short
                  days are removed; their price move goes into the next retained day's session-break return.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_PER_DAY = 1440


@dataclass(frozen=True)
class RVConfig:
    interval_min: int = 5
    include_session_break: bool = True
    excluded_windows: tuple[tuple[int, int], ...] = ()   # minute-of-day [start, end); start > end wraps midnight
    merge_sunday: bool = True
    short_day_share: float | None = 0.5                  # None: no day is removed
    day_length: str = "span"                             # "span": grid returns per day; "active": intervals with a bar
    median_day_length: float | None = None               # fixed in advance; None: median of the sample given


def excluded_mask(windows: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Boolean array over the 1440 minutes of the day."""
    mask = np.zeros(MIN_PER_DAY, dtype=bool)
    for a, b in windows:
        if not (0 <= a < MIN_PER_DAY and 0 <= b <= MIN_PER_DAY):
            raise ValueError(f"window out of range: {(a, b)}")
        if a < b:
            mask[a:b] = True
        else:                                               # wraps midnight
            mask[a:] = True
            mask[:b] = True
    return mask


def daily_measures(m1: pd.DataFrame, cfg: RVConfig = RVConfig()) -> pd.DataFrame:
    """
    m1: cleaned, time-sorted M1 bars with columns ts, open, high, low, close.
    Returns one row per trading day (index 'tday'). Short days are flagged in 'short' and carry NaN in the
    return-based columns; use `retained()` for the daily series the models see.
    """
    if cfg.day_length not in ("span", "active"):
        raise ValueError("day_length must be 'span' or 'active'")
    ts = m1["ts"]
    if not ts.is_monotonic_increasing or ts.duplicated().any():
        raise ValueError("M1 bars must be time-sorted without duplicates")
    day = ts.dt.normalize().to_numpy()
    mod = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
    is_sun = (ts.dt.day_of_week == 6).to_numpy() & cfg.merge_sunday
    tday = day + np.where(is_sun, np.timedelta64(1, "D"), np.timedelta64(0, "D"))
    m = mod - MIN_PER_DAY * is_sun                          # minutes since the trading day's 00:00; negative on Sunday

    # daily OHLC from all bars of the server day
    g = pd.DataFrame({"tday": tday, "open": m1["open"].to_numpy(), "high": m1["high"].to_numpy(),
                      "low": m1["low"].to_numpy(), "close": m1["close"].to_numpy()}).groupby("tday", sort=True)
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), bars=("close", "size"))

    # grid prices: last included close per (trading day, segment, grid point)
    excl = excluded_mask(cfg.excluded_windows)
    inc = ~excl[mod]
    seg_of = np.cumsum(np.tile(excl, 2))                    # constant between excluded minutes, over [-1440, 1440)
    T, S = tday[inc], seg_of[m[inc] + MIN_PER_DAY]
    K = (m[inc] + cfg.interval_min) // cfg.interval_min     # ceil of the close time (m + 1) on the grid
    P = m1["close"].to_numpy(dtype=float)[inc]
    last = np.ones(len(T), dtype=bool)
    last[:-1] = (T[1:] != T[:-1]) | (S[1:] != S[:-1]) | (K[1:] != K[:-1])
    T, S, K, P = T[last], S[last], K[last], P[last]

    same = (T[1:] == T[:-1]) & (S[1:] == S[:-1])            # consecutive grid prices inside one segment
    r = 100.0 * np.log(P[1:][same] / P[:-1][same])
    grid = pd.DataFrame({"tday": T[1:][same], "r": r, "r2": r ** 2, "r4": r ** 4, "steps": (K[1:] - K[:-1])[same]})
    sums = grid.groupby("tday").agg(sum_r=("r", "sum"), rv_intraday=("r2", "sum"), sum_r4=("r4", "sum"), n_returns=("steps", "sum"))
    out = out.join(sums)
    out["n_active"] = pd.Series(T).value_counts().reindex(out.index)
    for c in ("sum_r", "rv_intraday", "sum_r4", "n_returns", "n_active"):
        out[c] = out[c].fillna(0.0)
    out["n_returns"] = out["n_returns"].astype(int)
    out["n_active"] = out["n_active"].astype(int)

    # short days
    length = out["n_returns"] if cfg.day_length == "span" else out["n_active"]
    if cfg.short_day_share is None:
        out["short"] = False
    else:
        median = cfg.median_day_length if cfg.median_day_length is not None else float(length.median())
        out["short"] = length < cfg.short_day_share * median

    # close-to-close and session-break returns along the retained days
    keep = ~out["short"]
    prev_close = out.loc[keep, "close"].shift(1)
    out["r_cc"] = 100.0 * np.log(out.loc[keep, "close"] / prev_close)
    out["r_break"] = out["r_cc"] - out["sum_r"]
    out["rv"] = out["rv_intraday"] + out["r_break"] ** 2 if cfg.include_session_break else out["rv_intraday"]
    out["rq"] = out["n_returns"] / 3.0 * out["sum_r4"]
    out.loc[~keep, ["rv", "rv_intraday", "rq", "sum_r", "sum_r4"]] = np.nan
    out["parkinson"] = parkinson(out["high"], out["low"])
    out["garman_klass"] = garman_klass(out["open"], out["high"], out["low"], out["close"])
    out.loc[~keep, ["parkinson", "garman_klass"]] = np.nan
    out.index.name = "tday"
    return out


def retained(daily: pd.DataFrame) -> pd.DataFrame:
    """The daily series the models see: short days and the first day (no previous close) removed."""
    return daily[~daily["short"] & daily["r_cc"].notna()].drop(columns="short")


def parkinson(high: pd.Series, low: pd.Series) -> pd.Series:
    """Parkinson (1980) daily variance from the high-low range, in percent squared."""
    return (100.0 * np.log(high / low)) ** 2 / (4.0 * np.log(2.0))


def garman_klass(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Garman-Klass (1980) daily variance from open, high, low and close, in percent squared."""
    hl = 100.0 * np.log(high / low)
    co = 100.0 * np.log(close / open_)
    return 0.5 * hl ** 2 - (2.0 * np.log(2.0) - 1.0) * co ** 2
