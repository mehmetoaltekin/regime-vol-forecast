"""Loading and cleaning of the raw MT5 M1 exports. Times are broker server time."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"
COLS = ["date", "time", "open", "high", "low", "close", "tickvol", "vol", "spread"]


def load_m1(path: Path) -> pd.DataFrame:
    """MT5 export (tab-separated, header <DATE>..<SPREAD>) -> frame with a 'ts' column, in file order."""
    df = pd.read_csv(path, sep="\t", header=0, names=COLS, dtype={"date": "str", "time": "str"})
    df["ts"] = pd.to_datetime(df.pop("date") + " " + df.pop("time"), format="%Y.%m.%d %H:%M:%S")
    return df[["ts", "open", "high", "low", "close", "tickvol", "vol", "spread"]]


def raw_path(symbol: str, raw_dir: Path = RAW) -> Path:
    hits = sorted(raw_dir.glob(f"{symbol}_M1_*.csv"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one raw file for {symbol} in {raw_dir}, found {len(hits)}")
    return hits[0]


@dataclass(frozen=True)
class CleanConfig:
    """
    Bar-level cleaning. Defaults reproduce the earlier, simpler cleaning; the bounce-back filter is off unless
    bounce_sd is set (PRESPEC 12.7: 20, with reversal 0.75 and 20 lookback days).
    """
    start: str | None = None            # first server day kept (inclusive)
    end: str | None = None              # last server day kept (inclusive)
    drop_saturday: bool = True          # Saturday bars are not part of any trading day
    drop_bad_rows: bool = True          # zero or negative prices, high < low
    bounce_sd: float | None = None      # bounce-back filter: one-minute return above this many robust SDs ...
    bounce_reversal: float = 0.75       # ... of which at least this share is reversed in the next minute
    bounce_lookback_days: int = 20      # robust SD from the previous trading days only


def bounce_back_mask(df: pd.DataFrame, sd_mult: float, reversal: float = 0.75, lookback_days: int = 20) -> tuple[np.ndarray, dict]:
    """
    Bars to remove under the bounce-back rule. df: de-duplicated, time-sorted bars without Saturday bars.

    The one-minute return of a bar is 100 * log(close / previous close), defined when the previous bar is exactly one
    minute earlier. The robust SD of a bar on trading day D is 1.4826 times the median absolute deviation of all
    one-minute returns of the `lookback_days` trading days before D (days with bars; past data only; nothing is
    flagged on the first `lookback_days` days). A bar is flagged if |r| > sd_mult * SD and the next bar, exactly one
    minute later and on the same server day, returns at least `reversal` of r in the opposite direction (overshoot
    counts). The last minute of a server day is therefore not judged: its next minute belongs to the next day, and the
    data of a day must not depend on later days. One pass on the returns of the data as given; flagged bars are not
    removed before the next bar is judged.
    """
    ts = df["ts"].to_numpy()
    c = df["close"].to_numpy(dtype=float)
    n = len(df)
    r = np.full(n, np.nan)
    one = np.diff(ts) == np.timedelta64(1, "m")
    r[1:][one] = 100.0 * np.log(c[1:][one] / c[:-1][one])
    day = df["ts"].dt.normalize().to_numpy()
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    bounds = np.r_[starts, n]
    scale = np.full(len(starts), np.nan)
    for i in range(lookback_days, len(starts)):
        x = r[bounds[i - lookback_days]: bounds[i]]
        x = x[np.isfinite(x)]
        if len(x):
            scale[i] = 1.4826 * np.median(np.abs(x - np.median(x)))
    sd = np.repeat(scale, np.diff(bounds))
    nxt = np.full(n, np.nan)
    nxt[:-1] = np.where(day[1:] == day[:-1], r[1:], np.nan)                   # the next bar must be on the same server day
    with np.errstate(invalid="ignore"):
        big = np.abs(r) > sd_mult * sd
        back = -nxt * np.sign(r) >= reversal * np.abs(r)
    flag = big & back & np.isfinite(r) & np.isfinite(nxt)
    first_filtered = int(bounds[min(lookback_days, len(starts))]) if len(starts) else 0
    info = {"bounce_back": int(flag.sum()),
            "bounce_unfiltered_bars": first_filtered,
            "bounce_filter_starts": pd.Timestamp(ts[first_filtered]) if first_filtered < n else pd.NaT,
            "bounce_zero_scale_flagged": int((flag & (sd == 0)).sum())}
    return flag, info


LEGACY_KEYS = ("rows_in", "bad_rows", "duplicates", "saturday_bars", "outside_period", "rows_out")


def clean_m1_full(df: pd.DataFrame, cfg: CleanConfig = CleanConfig()) -> tuple[pd.DataFrame, dict]:
    """
    Returns the cleaned, time-sorted frame and the number of rows removed by each rule, in the order of the rules:
    bad rows (non-positive price or high < low), duplicate timestamps (last row kept), Saturday bars, bounce-back
    bars, then the sample period (bars before the start and after the end are counted separately).
    The bounce-back filter runs before the period is cut, so it has its past days at the start of the sample.
    """
    counts = {"rows_in": len(df)}
    if cfg.drop_bad_rows:
        bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | (df["high"] < df["low"])
        counts["bad_rows"] = int(bad.sum())
        df = df[~bad]
    dup = df["ts"].duplicated(keep="last")
    counts["duplicates"] = int(dup.sum())
    df = df[~dup].sort_values("ts", kind="stable")
    if cfg.drop_saturday:
        sat = df["ts"].dt.day_of_week == 5
        counts["saturday_bars"] = int(sat.sum())
        df = df[~sat]
    if cfg.bounce_sd is not None:
        df = df.reset_index(drop=True)
        flag, info = bounce_back_mask(df, cfg.bounce_sd, cfg.bounce_reversal, cfg.bounce_lookback_days)
        counts.update(info)
        df = df[~flag]
    day = df["ts"].dt.normalize()
    before = (day < pd.Timestamp(cfg.start)) if cfg.start is not None else pd.Series(False, index=df.index)
    after = (day > pd.Timestamp(cfg.end)) if cfg.end is not None else pd.Series(False, index=df.index)
    counts["before_start"] = int(before.sum())
    counts["after_end"] = int(after.sum())
    counts["outside_period"] = int((before | after).sum())
    df = df[~(before | after)].reset_index(drop=True)
    counts["rows_out"] = len(df)
    return df, counts


def clean_m1(df: pd.DataFrame, cfg: CleanConfig = CleanConfig()) -> tuple[pd.DataFrame, dict]:
    """clean_m1_full with the counts of the earlier version (plus the bounce-back counts when that filter is on)."""
    out, full = clean_m1_full(df, cfg)
    keep = LEGACY_KEYS + (("bounce_back",) if cfg.bounce_sd is not None else ())
    return out, {k: full[k] for k in keep if k in full}
