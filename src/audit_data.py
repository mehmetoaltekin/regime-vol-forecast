"""
Data audit of the raw MT5 M1 exports (data quality only: no forecasts, no model comparison).

    python src/audit_data.py            # reads data/raw/, writes results/data_audit.md and results/data_audit/*.csv

Every number in the report is computed here from the raw files. Times are broker server time.
Definitions used throughout:
  - trading day   : server-time calendar day; Sunday bars are counted with the following Monday
  - minute of day : server-time HH:MM of the bar's open time
  - in session    : a (year, weekday, minute-of-day) slot that has a bar on at least 50% of that year's traded
                    days of that weekday (empirical, from the data; the broker-declared sessions are printed too)
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT_MD = ROOT / "results/data_audit.md"
OUT_DIR = ROOT / "results/data_audit"

# Pre-specified instrument slots: (asset class, primary symbols, alternate symbol or None)
CLASSES = [
    ("Equity indices", ["US500", "USTEC"], "DE40"),
    ("Precious metals", ["XAUUSD", "XAGUSD"], None),
    ("FX", ["EURUSD", "USDJPY"], "GBPUSD"),
]
ELIGIBILITY_CUTOFF = pd.Timestamp("2014-12-31")      # history must start on or before this day
SAMPLE_END = pd.Timestamp("2026-10-31")
SPREAD_MULT = 3.0                                    # rollover rule: minute median > 3 x all-day median
SHORT_DAY_SHARE = 0.5                                # short-day rule: fewer than 50% of the median day length
STALE_SHARE = 0.5                                    # a "stale day": more than half of its one-minute returns are zero
JUMP_NS = (10, 20, 30)                               # thresholds in robust standard deviations
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
COLS = ["date", "time", "open", "high", "low", "close", "tickvol", "vol", "spread"]


# ----------------------------------------------------------------------------------------------- loading

def load_m1(path: Path) -> pd.DataFrame:
    """MT5 export (tab-separated, header <DATE>..<SPREAD>) -> frame with a 'ts' column, in file order."""
    df = pd.read_csv(path, sep="\t", header=0, names=COLS, dtype={"date": "str", "time": "str"})
    df["ts"] = pd.to_datetime(df.pop("date") + " " + df.pop("time"), format="%Y.%m.%d %H:%M:%S")
    return df[["ts", "open", "high", "low", "close", "tickvol", "vol", "spread"]]


def read_meta(path: Path) -> dict:
    meta: dict = {}
    if path.exists():
        for line in path.read_text(encoding="latin-1").splitlines()[1:]:
            p = line.split("\t", 2)
            if len(p) == 3:
                meta.setdefault(p[0], {})[p[1]] = p[2]
    return meta


# ----------------------------------------------------------------------------------------------- helpers

def hm(m: int) -> str:
    return f"{int(m) // 60:02d}:{int(m) % 60:02d}"


def ranges(minutes, limit: int | None = None) -> str:
    """Sorted minute-of-day values -> '00:00-00:59, 23:15-23:29' (inclusive ranges); a summary above `limit` ranges."""
    ms = sorted(int(x) for x in minutes)
    if not ms:
        return "none"
    out, a, b = [], ms[0], ms[0]
    for x in ms[1:]:
        if x == b + 1:
            b = x
            continue
        out.append((a, b))
        a = b = x
    out.append((a, b))
    if limit is not None and len(out) > limit:
        return f"{len(ms)} minutes in {len(out)} ranges between {hm(ms[0])} and {hm(ms[-1])}"
    return ", ".join(hm(a) if a == b else f"{hm(a)}-{hm(b)}" for a, b in out)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """De-duplicated (last kept), time-sorted frame with calendar columns."""
    d = df.drop_duplicates("ts", keep="last").sort_values("ts", kind="stable").reset_index(drop=True)
    d["day"] = d["ts"].dt.normalize()
    d["wd"] = d["ts"].dt.day_of_week.astype("int8")                        # Monday = 0 .. Sunday = 6
    d["mod"] = (d["ts"].dt.hour * 60 + d["ts"].dt.minute).astype("int16")
    d["year"] = d["ts"].dt.year.astype("int16")
    d["tday"] = d["day"] + pd.to_timedelta((d["wd"] == 6).astype("int8"), unit="D")   # Sunday -> Monday
    return d


# ----------------------------------------------------------------------------------------------- checks

def basic_checks(df: pd.DataFrame) -> dict:
    ts = df["ts"]
    o, h, l, c = (df[k] for k in ("open", "high", "low", "close"))
    step = ts.diff().dt.total_seconds()
    return {
        "bars": len(df),
        "first": ts.min(),
        "last": ts.max(),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "out_of_order": int((step < 0).sum()),
        "off_minute_timestamps": int((ts.dt.second != 0).sum()),
        "nonpositive_price_rows": int(((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)).sum()),
        "missing_values": int(df.isna().any(axis=1).sum()),
        "high_lt_low": int((h < l).sum()),
        "ohlc_inconsistent": int(((h < np.maximum(o, c)) | (l > np.minimum(o, c))).sum()),
        "bars_after_sample_end": int((ts >= SAMPLE_END + pd.Timedelta(days=1)).sum()),
    }


def weekend(d: pd.DataFrame) -> dict:
    sat, sun = d[d["wd"] == 5], d[d["wd"] == 6]
    weeks = d["tday"].dt.to_period("W").nunique()
    first_sun = sun.groupby("day")["mod"].min()
    by_year = sun.groupby("year")["day"].nunique()
    return {
        "saturday_bars": len(sat),
        "saturday_days": int(sat["day"].nunique()),
        "saturday_dates": [str(x.date()) for x in sorted(sat["day"].unique())[:10]],
        "sunday_bars": len(sun),
        "sunday_days": int(sun["day"].nunique()),
        "weeks": int(weeks),
        "sunday_first_bar_mode": hm(first_sun.mode().iloc[0]) if len(first_sun) else "",
        "sunday_first_bar_range": f"{hm(first_sun.min())}..{hm(first_sun.max())}" if len(first_sun) else "",
        "sunday_days_by_year": {int(k): int(v) for k, v in by_year.items()},
    }


def session_slots(d: pd.DataFrame) -> tuple[np.ndarray, list, pd.DataFrame]:
    """in_session[year index, weekday, minute] and the number of traded days per (year, weekday)."""
    years = sorted(d["year"].unique())
    yi = {y: i for i, y in enumerate(years)}
    days = d.groupby(["year", "wd"])["day"].nunique()
    cnt = np.zeros((len(years), 7, 1440), dtype=np.int32)
    np.add.at(cnt, (d["year"].map(yi).to_numpy(), d["wd"].to_numpy(), d["mod"].to_numpy()), 1)
    n = np.zeros((len(years), 7), dtype=np.int32)
    for (y, wd), v in days.items():
        n[yi[y], wd] = v
    with np.errstate(divide="ignore", invalid="ignore"):
        presence = np.where(n[:, :, None] > 0, cnt / n[:, :, None], 0.0)
    return presence >= 0.5, years, days.unstack(fill_value=0)


def missing_minutes(d: pd.DataFrame, in_sess: np.ndarray, years: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-day in-session minutes without a bar, and the interior gaps (consecutive bars more than 1 minute apart)."""
    yi = {y: i for i, y in enumerate(years)}
    y_idx, wd, mod = d["year"].map(yi).to_numpy(), d["wd"].to_numpy(), d["mod"].to_numpy()
    d = d.assign(ins=in_sess[y_idx, wd, mod])
    slots = in_sess.sum(axis=2)
    per_day = d.groupby("day").agg(year=("year", "first"), wd=("wd", "first"), bars=("ts", "size"), in_bars=("ins", "sum"),
                                   first=("mod", "min"), last=("mod", "max"))
    per_day["slots"] = slots[per_day["year"].map(yi).to_numpy(), per_day["wd"].to_numpy()]
    per_day["missing"] = per_day["slots"] - per_day["in_bars"]

    cs = np.concatenate([np.zeros(in_sess.shape[:2] + (1,), dtype=np.int32), np.cumsum(in_sess, axis=2)], axis=2)
    same = (d["day"].to_numpy()[1:] == d["day"].to_numpy()[:-1]) & (np.diff(mod) > 1)
    i = np.nonzero(same)[0]
    gaps = pd.DataFrame({
        "gap_start": d["ts"].to_numpy()[i] + np.timedelta64(1, "m"),
        "gap_end": d["ts"].to_numpy()[i + 1] - np.timedelta64(1, "m"),
        "minutes": mod[i + 1] - mod[i] - 1,
        "in_session_minutes": cs[y_idx[i], wd[i], mod[i + 1]] - cs[y_idx[i], wd[i], mod[i] + 1],
    })
    return per_day, gaps[gaps["in_session_minutes"] > 0].reset_index(drop=True)


def missing_weekdays(d: pd.DataFrame) -> pd.DatetimeIndex:
    """Monday-Friday calendar days between the first and last bar without any bar."""
    all_days = pd.bdate_range(d["day"].min(), d["day"].max())
    return all_days.difference(pd.DatetimeIndex(d["day"].unique()))


def session_table(in_sess: np.ndarray, years: list) -> pd.DataFrame:
    rows = []
    for k, y in enumerate(years):
        row = {"year": int(y)}
        for wd in (6, 0, 2, 4):
            row[WEEKDAYS[wd]] = ranges(np.nonzero(in_sess[k, wd])[0], limit=6)
        rows.append(row)
    return pd.DataFrame(rows)


def day_edges(d: pd.DataFrame) -> pd.DataFrame:
    """Runs of weeks with the same typical first bar (Tue-Fri) and last bar (Mon-Thu) of the day."""
    g = d.groupby("day").agg(wd=("wd", "first"), first=("mod", "min"), last=("mod", "max"))
    g["week"] = g.index.to_period("W").start_time
    mode = lambda s: int(s.mode().iloc[0]) if len(s) else -1
    first = g[g["wd"].between(1, 4)].groupby("week")["first"].agg(mode)
    last = g[g["wd"].between(0, 3)].groupby("week")["last"].agg(mode)
    w = pd.concat([first, last], axis=1).dropna().astype(int)
    key = (w["first"] != w["first"].shift()) | (w["last"] != w["last"].shift())
    runs = w.assign(run=key.cumsum()).reset_index().groupby("run").agg(
        from_week=("week", "min"), to_week=("week", "max"), weeks=("week", "size"), first=("first", "first"), last=("last", "first"))
    runs["first_bar"] = runs.pop("first").map(hm)
    runs["last_bar"] = runs.pop("last").map(hm)
    return runs.reset_index(drop=True)


def dst_shifts(d: pd.DataFrame) -> pd.DataFrame:
    """
    Weekly hour-of-day tick-volume profile against the same year's median profile: the circular shift (in hours,
    -2..+2) with the highest correlation. A shift other than 0 means the intraday pattern sits earlier or later
    in server time that week, as happens while the server's and a market's daylight-saving dates differ.
    """
    x = d[d["wd"] < 5]
    week = x["day"].dt.to_period("W").dt.start_time
    prof = x.groupby([week, x["ts"].dt.hour])["tickvol"].sum().unstack(fill_value=0).reindex(columns=range(24), fill_value=0)
    ndays = x.groupby(week)["day"].nunique()
    prof = prof[ndays.reindex(prof.index) >= 4]
    prof = prof.div(prof.sum(axis=1), axis=0)
    out = []
    for y, grp in prof.groupby(prof.index.year):
        ref = grp.median().to_numpy()
        for wk, row in grp.iterrows():
            v = row.to_numpy()
            cors = {s: np.corrcoef(np.roll(ref, s), v)[0, 1] for s in (-2, -1, 0, 1, 2)}
            best = max(cors, key=cors.get)
            out.append((wk, best, cors[best], cors[0]))
    w = pd.DataFrame(out, columns=["week", "shift_hours", "corr_best", "corr_0"]).sort_values("week")
    w = w[(w["shift_hours"] != 0) & (w["corr_best"] - w["corr_0"] > 0.05)]
    if w.empty:
        return pd.DataFrame(columns=["from_week", "to_week", "weeks", "shift_hours"])
    new = (w["shift_hours"] != w["shift_hours"].shift()) | (w["week"].diff() > pd.Timedelta(days=7))
    return w.assign(run=new.cumsum()).groupby("run").agg(
        from_week=("week", "min"), to_week=("week", "max"), weeks=("week", "size"), shift_hours=("shift_hours", "first")
    ).reset_index(drop=True)


def spread_profile(d: pd.DataFrame) -> dict:
    prof = d.groupby("mod")["spread"].agg(median="median", bars="size").reindex(range(1440))
    all_bars = float(d["spread"].median())
    of_minutes = float(prof["median"].median())
    flagged = prof.index[prof["median"] > SPREAD_MULT * all_bars]
    flagged_b = prof.index[prof["median"] > SPREAD_MULT * of_minutes]
    by_year = []
    for y, g in d.groupby("year"):
        p = g.groupby("mod")["spread"].median()
        med = float(g["spread"].median())
        by_year.append({"year": int(y), "median_spread": med, "zero_spread_share": float((g["spread"] == 0).mean()),
                        "distinct_values": int(g["spread"].nunique()),
                        "minutes_over_3x": ranges(p.index[p > SPREAD_MULT * med], limit=12) if med > 0 else "n/a (median 0)"})
    return {"profile": prof, "median_all_bars": all_bars, "median_of_minute_medians": of_minutes,
            "flagged": list(flagged), "flagged_alt": list(flagged_b), "by_year": pd.DataFrame(by_year)}


def day_lengths(d: pd.DataFrame) -> pd.DataFrame:
    """Per trading day: 5-minute intervals with at least one bar ('active') and grid returns from first to last bar ('span')."""
    b5 = d["mod"] // 5 + (d["day"] != d["tday"]).to_numpy() * -288      # Sunday bins sit before Monday 00:00
    g = d.assign(b5=b5).groupby("tday")["b5"].agg(active="nunique", lo="min", hi="max")
    g["n5_active"] = g.pop("active")
    g["n5_span"] = g.pop("hi") - g.pop("lo")
    g["bars"] = d.groupby("tday").size()
    g["weekday"] = [WEEKDAYS[x] for x in g.index.day_of_week]
    for k in ("n5_active", "n5_span"):
        g[f"short_{k}"] = g[k] < SHORT_DAY_SHARE * g[k].median()
    return g


def jumps(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    One-minute log returns between consecutive bars exactly one minute apart, scaled by the trading day's robust
    standard deviation (1.4826 x median absolute deviation; days where that is zero use the year's value).
    """
    c = d["close"].to_numpy(dtype=float)
    ok = (np.diff(d["ts"].to_numpy()) == np.timedelta64(1, "m")) & (c[1:] > 0) & (c[:-1] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(ok, 100 * np.log(c[1:] / c[:-1]), np.nan)
    x = pd.DataFrame({"ts": d["ts"].to_numpy()[1:], "tday": d["tday"].to_numpy()[1:], "year": d["year"].to_numpy()[1:], "r": r}).dropna()
    mad = lambda s: 1.4826 * (s - s.median()).abs().median()
    s_day = x.groupby("tday")["r"].transform(mad)
    s_year = x.groupby("year")["r"].transform(mad)
    x["scale"] = s_day.where(s_day > 0, s_year)
    x = x[x["scale"] > 0]
    x["z"] = x["r"] / x["scale"]
    nxt = x["r"].shift(-1).where(x["ts"].shift(-1) - x["ts"] == pd.Timedelta(minutes=1))
    x["reverted"] = (nxt * x["r"] < 0) & ((x["r"] + nxt).abs() < 0.25 * x["r"].abs())
    counts = {n: {"jumps": int((x["z"].abs() > n).sum()), "reverted": int(((x["z"].abs() > n) & x["reverted"]).sum())}
              for n in JUMP_NS}
    counts["returns"] = len(x)
    counts["zero_scale_days"] = int((s_day == 0).groupby(x["tday"]).first().sum()) if len(x) else 0
    big = x[x["z"].abs() > min(JUMP_NS)].assign(abs_z=lambda t: t["z"].abs()).sort_values("abs_z", ascending=False)
    return big[["ts", "r", "scale", "z", "reverted"]], counts


def stale_prices(d: pd.DataFrame) -> pd.DataFrame:
    """Per trading day: share of one-minute steps with an unchanged close, and the longest run of unchanged closes."""
    c, t, day = d["close"].to_numpy(), d["ts"].to_numpy(), d["tday"].to_numpy()
    step = (np.diff(t) == np.timedelta64(1, "m")) & (day[1:] == day[:-1])
    flat = step & (c[1:] == c[:-1])
    cs = np.cumsum(flat)
    run = cs - np.maximum.accumulate(np.where(flat, 0, cs))            # length of the flat run ending at each step
    x = pd.DataFrame({"tday": day[1:], "step": step, "flat": flat, "run": run})
    g = x.groupby("tday").agg(steps=("step", "sum"), flat=("flat", "sum"), longest_flat_run=("run", "max"))
    g["zero_return_share"] = g["flat"] / g["steps"].where(g["steps"] > 0)
    return g


def dense_start(dl: pd.DataFrame, window: int = 20):
    """
    First day of the first run of `window` consecutive trading days that each have at least half as many active
    5-minute intervals as the median of the last 500 trading days: where minute-resolution history begins.
    """
    ok = (dl["n5_active"] >= 0.5 * dl["n5_active"].tail(500).median()).to_numpy()
    full = np.convolve(ok, np.ones(window, dtype=int), "valid") == window
    return dl.index[int(np.argmax(full))] if full.any() else None


def missing_runs(days: pd.DatetimeIndex, min_len: int = 3) -> pd.DataFrame:
    """Runs of consecutive missing Monday-Friday days."""
    if len(days) == 0:
        return pd.DataFrame(columns=["from", "to", "weekdays"])
    s = pd.Series(days)
    new = (s - pd.tseries.offsets.BDay(1)) != s.shift()
    runs = s.groupby(new.cumsum()).agg(["min", "max", "size"])
    runs.columns = ["from", "to", "weekdays"]
    return runs[runs["weekdays"] >= min_len].reset_index(drop=True)


def year_table(d: pd.DataFrame, per_day: pd.DataFrame, dl: pd.DataFrame, st: pd.DataFrame, big: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("year")
    t = pd.DataFrame({
        "bars": g.size(),
        "days_with_bars": g["day"].nunique(),
        "median_bars_per_day": per_day.groupby("year")["bars"].median(),
        "in_session_slots": per_day.groupby("year")["slots"].sum(),
        "missing_in_session": per_day.groupby("year")["missing"].sum(),
        "median_spread": g["spread"].median(),
        "median_tickvol": g["tickvol"].median(),
    })
    t["missing_share_%"] = 100 * t["missing_in_session"] / t["in_session_slots"]
    t["short_days_active"] = dl.groupby(dl.index.year)["short_n5_active"].sum()
    t["short_days_span"] = dl.groupby(dl.index.year)["short_n5_span"].sum()
    t["median_zero_return_%"] = 100 * st.groupby(st.index.year)["zero_return_share"].median()
    t["stale_days"] = (st["zero_return_share"] > STALE_SHARE).groupby(st.index.year).sum()
    t[f"jumps_gt_{JUMP_NS[1]}sd"] = big[big["z"].abs() > JUMP_NS[1]].groupby(big["ts"].dt.year).size()
    t[f"jumps_gt_{JUMP_NS[1]}sd"] = t[f"jumps_gt_{JUMP_NS[1]}sd"].fillna(0).astype(int)
    return t.reset_index()


def audit_symbol(sym: str, path: Path) -> dict:
    raw = load_m1(path)
    res = {"symbol": sym, "file": path.name, "basic": basic_checks(raw)}
    d = prepare(raw)
    del raw
    in_sess, years, traded = session_slots(d)
    per_day, gaps = missing_minutes(d, in_sess, years)
    dl = day_lengths(d)
    big, jump_counts = jumps(d)
    st = stale_prices(d)
    res.update({
        "weekend": weekend(d), "sessions": session_table(in_sess, years), "edges": day_edges(d), "dst": dst_shifts(d),
        "per_day": per_day, "gaps": gaps, "missing_weekdays": missing_weekdays(d), "spread": spread_profile(d),
        "day_lengths": dl, "jumps": big, "jump_counts": jump_counts, "years": year_table(d, per_day, dl, st, big),
        "stale": st, "dense_start": dense_start(dl),
        "first_day": d["day"].min(), "last_day": d["day"].max(),
    })
    return res


def eligibility(first_day: dict) -> tuple[list, list, str]:
    """The pre-specified eligibility rule applied to the first M1 bar of each symbol. Returns table rows, retained symbols, verdict."""
    rows, kept, classes = [], [], set()
    for cls, primaries, alt in CLASSES:
        ok = {s: (s in first_day and first_day[s] <= ELIGIBILITY_CUTOFF) for s in primaries + ([alt] if alt else [])}
        alt_free = alt is not None
        for s in primaries:
            if s not in first_day:
                rows.append((cls, s, "no data", "not exported", ""))
                continue
            if ok[s]:
                outcome, repl = "eligible", ""
                kept.append(s); classes.add(cls)
            elif alt_free and ok[alt]:
                outcome, repl, alt_free = "replaced by alternate", alt, False
                kept.append(alt); classes.add(cls)
            elif alt is None:
                outcome, repl = "dropped (no alternate)", ""
            elif not alt_free:
                outcome, repl = "dropped (alternate already used)", ""
            else:
                outcome, repl = "dropped (alternate not eligible or missing)", ""
            rows.append((cls, s, str(first_day[s].date()), outcome, repl))
        if alt:
            rows.append((cls, f"{alt} (alternate)", str(first_day[alt].date()) if alt in first_day else "no data",
                         "eligible" if ok[alt] else "not eligible", ""))
    verdict = ("requirement met" if len(kept) >= 4 and len(classes) >= 2 else "REQUIREMENT NOT MET: the study stops here")
    return rows, kept, f"{len(kept)} instruments from {len(classes)} asset classes: {verdict}"


# ----------------------------------------------------------------------------------------------- report

def md_table(df: pd.DataFrame, floats: str = "{:.2f}") -> str:
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floats.format(v)
        if isinstance(v, pd.Timestamp):
            return str(v.date()) if v == v.normalize() else v.strftime("%Y-%m-%d %H:%M")
        return str(v)
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)])


def symbol_section(r: dict, meta: dict) -> str:
    b, w, sp, dl, jc = r["basic"], r["weekend"], r["spread"], r["day_lengths"], r["jump_counts"]
    m = meta.get(r["symbol"], {})
    pd_, gaps = r["per_day"], r["gaps"]
    runs = missing_runs(r["missing_weekdays"])
    st = r["stale"]
    dense = dl[dl.index >= r["dense_start"]] if r["dense_start"] is not None else dl.iloc[:0]
    stale_month = st.assign(stale=st["zero_return_share"] > STALE_SHARE).groupby(st.index.to_period("M")).agg(
        days=("stale", "size"), stale_days=("stale", "sum"), median_zero_return_share=("zero_return_share", "median"),
        longest_flat_run=("longest_flat_run", "max"))
    stale_month = stale_month[stale_month["stale_days"] > 0].sort_values("stale_days", ascending=False).rename_axis("month").reset_index()
    L = [f"## {r['symbol']}", ""]
    L += [f"- File: `{r['file']}`",
          f"- First bar {b['first']:%Y-%m-%d %H:%M}, last bar {b['last']:%Y-%m-%d %H:%M}, {b['bars']:,} bars "
          f"({b['bars_after_sample_end']:,} after {SAMPLE_END.date()})",
          f"- Chart price mode: **{m.get('chart_mode', 'not recorded')}**; digits {m.get('digits', '?')}; "
          f"description \"{m.get('description', '')}\"",
          f"- Export: series synchronised {m.get('series_synchronized', '?')}, reached the server's first date "
          f"{m.get('reached_server_first_date', '?')} (server first date {m.get('server_first_date', '?')})",
          f"- Minute-resolution history from: **{r['dense_start'].date() if r['dense_start'] is not None else 'never'}** "
          "(first of 20 consecutive trading days with at least half the recent number of active 5-minute intervals)",
          f"- Duplicate timestamps: {b['duplicate_timestamps']:,}; out-of-order rows: {b['out_of_order']:,}; "
          f"timestamps not on a full minute: {b['off_minute_timestamps']:,}; rows with missing values: {b['missing_values']:,}",
          f"- Zero or negative prices: {b['nonpositive_price_rows']:,} rows; `high < low`: {b['high_lt_low']:,} rows; "
          f"high below max(open, close) or low above min(open, close): {b['ohlc_inconsistent']:,} rows", ""]

    L += ["### Weekend bars", "",
          f"- Saturday: {w['saturday_bars']:,} bars on {w['saturday_days']} days"
          + (f" ({', '.join(w['saturday_dates'])}{' ...' if w['saturday_days'] > 10 else ''})" if w["saturday_days"] else ""),
          f"- Sunday: {w['sunday_bars']:,} bars on {w['sunday_days']} days ({w['weeks']} weeks in the file)"
          + (f"; first Sunday bar usually at {w['sunday_first_bar_mode']} (range {w['sunday_first_bar_range']})" if w["sunday_days"] else ""),
          f"- Sundays with bars per year: {w['sunday_days_by_year'] or 'none'}", ""]

    L += ["### Sessions", "",
          f"- Broker-declared quote sessions (0 = Sunday): `{m.get('sessions_quote', 'not recorded')}`",
          f"- Broker-declared trade sessions: `{m.get('sessions_trade', 'not recorded')}`", "",
          "Empirical in-session minutes per year (slots with a bar on at least 50% of that weekday's traded days):", "",
          md_table(r["sessions"]), "",
          "Typical first bar of the day (Tue-Fri) and last bar (Mon-Thu), as runs of weeks with the same values "
          f"({len(r['edges'])} runs; full list in `data_audit/day_edges_{r['symbol']}.csv`):", "",
          md_table(r["edges"].head(40)), ""]

    L += ["### DST shifts", "",
          "Weeks whose hour-of-day tick-volume profile fits the year's profile better after a shift of whole hours "
          "(positive = the pattern is later in server time):", "",
          md_table(r["dst"]) if len(r["dst"]) else "No shifted weeks found.", ""]

    L += ["### Missing minutes inside sessions", "",
          f"- In-session minutes without a bar: {int(pd_['missing'].clip(lower=0).sum()):,} of {int(pd_['slots'].sum()):,} "
          f"({100 * pd_['missing'].clip(lower=0).sum() / max(pd_['slots'].sum(), 1):.2f}%)",
          f"- Per day: median {pd_['missing'].median():.0f}, 95th percentile {pd_['missing'].quantile(0.95):.0f}, max {pd_['missing'].max():.0f}",
          f"- Interior gaps with at least 5 / 30 / 60 in-session minutes missing: "
          f"{int((gaps['in_session_minutes'] >= 5).sum()):,} / {int((gaps['in_session_minutes'] >= 30).sum()):,} / "
          f"{int((gaps['in_session_minutes'] >= 60).sum()):,}",
          f"- Monday-Friday calendar days without any bar: {len(r['missing_weekdays'])} "
          f"(list in `data_audit/missing_weekdays_{r['symbol']}.csv`)", "",
          "Runs of 3 or more consecutive Monday-Friday days without any bar:", "",
          md_table(runs) if len(runs) else "None.", "",
          "Largest interior gaps:", "",
          md_table(gaps.sort_values("in_session_minutes", ascending=False).head(10)), ""]

    flagged_same = sp["flagged"] == sp["flagged_alt"]
    L += ["### Spread profile (rollover rule)", "",
          f"- All-day median of `<SPREAD>`: {sp['median_all_bars']:g} points over all bars; "
          f"{sp['median_of_minute_medians']:g} as the median of the 1440 minute medians",
          f"- Minutes whose median exceeds {SPREAD_MULT:g} x the all-bars median ({len(sp['flagged'])} minutes): {ranges(sp['flagged'])}",
          ("- The same minutes are flagged with the median of minute medians." if flagged_same else
           f"- With the median of minute medians instead ({len(sp['flagged_alt'])} minutes): {ranges(sp['flagged_alt'])}"),
          f"- Per-minute medians: `data_audit/spread_profile_{r['symbol']}.csv`", "",
          "Per year (the rule applied to each year's own bars):", "",
          md_table(sp["by_year"], "{:.3g}"), ""]

    L += ["### Day lengths (short-day rule)", "",
          f"- Trading days: {len(dl):,}",
          f"- 5-minute intervals with at least one bar per day: median {dl['n5_active'].median():.0f}; "
          f"days below {SHORT_DAY_SHARE:.0%} of it: {int(dl['short_n5_active'].sum())}",
          f"- 5-minute grid returns from the day's first to last bar: median {dl['n5_span'].median():.0f}; "
          f"days below {SHORT_DAY_SHARE:.0%} of it: {int(dl['short_n5_span'].sum())}",
          f"- Counted only from the minute-resolution start ({len(dense):,} days): "
          f"{int((dense['n5_active'] < SHORT_DAY_SHARE * dense['n5_active'].median()).sum())} short days by the active count "
          f"(median {dense['n5_active'].median():.0f}), "
          f"{int((dense['n5_span'] < SHORT_DAY_SHARE * dense['n5_span'].median()).sum())} by the span count "
          f"(median {dense['n5_span'].median():.0f})",
          f"- Short days by weekday (active count): "
          f"{dl[dl['short_n5_active']].groupby('weekday').size().to_dict() or 'none'}",
          f"- Daily values: `data_audit/day_lengths_{r['symbol']}.csv`", ""]

    L += ["### Price jumps", "",
          f"- One-minute returns checked: {jc['returns']:,}",
          *[f"- |return| > {n} robust SD: {jc[n]['jumps']:,}, of which reversed by at least 75% in the next minute: {jc[n]['reverted']:,}"
            for n in JUMP_NS],
          f"- All returns above {min(JUMP_NS)} robust SD: `data_audit/jumps_{r['symbol']}.csv`", "",
          "Largest:", "", md_table(r["jumps"].head(10), "{:.3f}"), ""]

    L += ["### Stale prices", "",
          f"A stale day has more than {STALE_SHARE:.0%} of its one-minute returns equal to zero.", "",
          f"- Stale days: {int((st['zero_return_share'] > STALE_SHARE).sum()):,} of {len(st):,}; "
          f"of the {len(dense):,} days from the minute-resolution start: "
          f"{int((st.loc[st.index.isin(dense.index), 'zero_return_share'] > STALE_SHARE).sum()):,}",
          f"- Longest run of unchanged one-minute closes: {int(st['longest_flat_run'].max()):,} minutes "
          f"(on {st['longest_flat_run'].idxmax().date()})", "",
          "Months with stale days (15 worst):", "",
          md_table(stale_month.head(15), "{:.3f}") if len(stale_month) else "None.", ""]

    L += ["### By year", "", md_table(r["years"]), ""]
    return "\n".join(L)


def write_csvs(r: dict) -> None:
    s = r["symbol"]
    r["spread"]["profile"].assign(minute=[hm(m) for m in range(1440)]).to_csv(OUT_DIR / f"spread_profile_{s}.csv", index_label="minute_of_day")
    r["day_lengths"].to_csv(OUT_DIR / f"day_lengths_{s}.csv", index_label="trading_day")
    r["jumps"].to_csv(OUT_DIR / f"jumps_{s}.csv", index=False, float_format="%.6g")
    r["gaps"].to_csv(OUT_DIR / f"gaps_{s}.csv", index=False)
    r["edges"].to_csv(OUT_DIR / f"day_edges_{s}.csv", index=False)
    r["stale"].to_csv(OUT_DIR / f"stale_{s}.csv", index_label="trading_day", float_format="%.4f")
    pd.Series(r["missing_weekdays"], name="day").to_csv(OUT_DIR / f"missing_weekdays_{s}.csv", index=False)


def main() -> None:
    files = {p.name.split("_M1_")[0]: p for p in sorted(RAW.glob("*_M1_*.csv"))}
    if not files:
        sys.exit("no raw files in data/raw")
    meta = read_meta(RAW / "export_meta.tsv")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    order = [s for _, prim, alt in CLASSES for s in prim + ([alt] if alt else [])]
    results = {}
    for s in [x for x in order if x in files] + [x for x in files if x not in order]:
        print(f"auditing {s} ...", flush=True)
        results[s] = audit_symbol(s, files[s])
        write_csvs(results[s])

    rows, kept, verdict = eligibility({s: r["first_day"] for s, r in results.items()})
    term = meta.get("_terminal", {})
    L = ["# Data audit: MT5 M1 exports", "",
         f"Generated by `src/audit_data.py` on {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC from the files in `data/raw/` "
         "(SHA-256 in `results/raw_data_sha256.tsv`). Data quality only; no forecast is computed.", "",
         f"- Server: {term.get('account_server', '?')} ({term.get('account_company', '?')}), terminal build {term.get('build', '?')}, "
         f"connected during the export: {term.get('connected', '?')}",
         f"- Server time minus GMT at export: {term.get('server_minus_gmt_hours', '?')} h "
         "(only meaningful if the terminal was connected)",
         f"- Symbols present: {', '.join(results)}; missing: {', '.join(s for s in order if s not in results) or 'none'}",
         "- All times are broker server time. A trading day is a server calendar day; Sunday bars count with Monday.",
         "- The export script requests bars from 2000-01-01. Each symbol's section gives the server's first date; "
         "anything the server holds before 2000 is not in the files.",
         f"- The files run to the export time; the counts below use every bar in the file, including those after the "
         f"sample end {SAMPLE_END.date()}.", ""]
    last = max(r["basic"]["last"] for r in results.values())
    missing = [s for s in order if s not in results]
    if term.get("connected") != "1" or missing or min(r["basic"]["last"] for r in results.values()) < SAMPLE_END:
        L += ["> **PROVISIONAL.** " + " ".join(filter(None, [
            "The terminal was offline during the export, so the files hold only what was in its local cache." if term.get("connected") != "1" else "",
            f"Missing symbols: {', '.join(missing)}." if missing else "",
            f"At least one file ends before the sample end {SAMPLE_END.date()}." if min(r["basic"]["last"] for r in results.values()) < SAMPLE_END else "",
            "The eligibility outcome below is not final."])), ""]
    L += [
         "## Summary", ""]
    summ = pd.DataFrame([{
        "symbol": s, "first bar": r["basic"]["first"], "minute data from": r["dense_start"], "last bar": r["basic"]["last"],
        "bars": f"{r['basic']['bars']:,}",
        "mode": meta.get(s, {}).get("chart_mode", "?"), "dup": r["basic"]["duplicate_timestamps"],
        "price<=0": r["basic"]["nonpositive_price_rows"], "high<low": r["basic"]["high_lt_low"],
        "Sat bars": r["weekend"]["saturday_bars"], "Sun days": r["weekend"]["sunday_days"],
        "missing in-session %": 100 * r["per_day"]["missing"].clip(lower=0).sum() / max(r["per_day"]["slots"].sum(), 1),
        "spread median": r["spread"]["median_all_bars"], "minutes > 3x": len(r["spread"]["flagged"]),
        "median 5-min/day": r["day_lengths"]["n5_active"].median(), "short days": int(r["day_lengths"]["short_n5_active"].sum()),
        f"jumps > {JUMP_NS[1]} SD": r["jump_counts"][JUMP_NS[1]]["jumps"],
        "stale days": int((r["stale"]["zero_return_share"] > STALE_SHARE).sum()),
        "missing weekdays": len(r["missing_weekdays"]),
    } for s, r in results.items()])
    L += [md_table(summ, "{:.4g}"), "",
          "## Eligibility rule (history must start on or before 2014-12-31)", "",
          md_table(pd.DataFrame(rows, columns=["asset class", "symbol", "first M1 bar", "outcome", "replacement"]).assign(**{
              "minute-resolution history from": lambda t: [
                  (str(results[s.split()[0]]["dense_start"].date()) if results[s.split()[0]]["dense_start"] is not None else "never")
                  if s.split()[0] in results else "" for s in t["symbol"]]})), "",
          "The outcome applies the rule as written, to the first M1 bar in the file. The last column shows where "
          "minute-by-minute history actually begins; where the two dates differ, the bars before the second date are "
          "not one-minute data (see the symbol's section).", "",
          f"- Retained: {', '.join(kept) or 'none'}", f"- {verdict}"]
    if kept and all(k in results for k in kept):
        L.append(f"- First server day on which all retained instruments have data: {max(results[k]['first_day'] for k in kept).date()}")
        dense = [results[k]["dense_start"] for k in kept]
        L.append("- First server day on which all retained instruments have minute-resolution data: "
                 + (str(max(dense).date()) if all(x is not None for x in dense) else "never"))
    L.append("")
    L += [symbol_section(r, meta) for r in results.values()]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
