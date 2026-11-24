"""Loading and cleaning of the raw MT5 M1 exports. Times are broker server time."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    """Cleaning rules. The defaults are placeholders until the pre-specification fixes them after the data audit."""
    start: str | None = None            # first server day kept (inclusive)
    end: str | None = None              # last server day kept (inclusive)
    drop_saturday: bool = True          # Saturday bars are not part of any trading day
    drop_bad_rows: bool = True          # zero or negative prices, high < low


def clean_m1(df: pd.DataFrame, cfg: CleanConfig = CleanConfig()) -> tuple[pd.DataFrame, dict]:
    """Returns the cleaned, time-sorted frame and a count of the rows removed by each rule."""
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
    day = df["ts"].dt.normalize()
    keep = pd.Series(True, index=df.index)
    if cfg.start is not None:
        keep &= day >= pd.Timestamp(cfg.start)
    if cfg.end is not None:
        keep &= day <= pd.Timestamp(cfg.end)
    counts["outside_period"] = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    counts["rows_out"] = len(df)
    return df, counts
