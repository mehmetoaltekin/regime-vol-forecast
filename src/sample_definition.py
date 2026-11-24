"""
results/sample_definition.md: what each rule of PRESPEC 12.5-12.7 removes from each panel instrument.

    python src/sample_definition.py [--config config/study_config.json]

For every instrument of the study panel (and, for robustness check 6, every instrument on its own full minute-data
history) the report gives the bars removed by each bar-level rule, the days removed by each day-level rule, the first
and last retained day, the burn-in and the first forecast day. It computes data-quality counts only; no forecast, no
return statistic and no model is touched. Every count is checked against an identity before it is written.

Day counts. A "day" is a server calendar day with at least one bar after the bar-level cleaning. The rules are applied
in the order short, stale, anchor; a day removed by an earlier rule is not counted again by a later one. The columns
"flagged" show each rule on its own where that is defined (short, stale), so overlaps are visible.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_study import CONFIG, RESULTS, verify_raw_hashes  # noqa: E402
from rvf import study as S  # noqa: E402
from rvf.data import clean_m1_full, load_m1, raw_path  # noqa: E402
from rvf.realized import daily_measures  # noqa: E402

OUT_MD = RESULTS / "sample_definition.md"
OUT_DIR = RESULTS / "sample_definition"


def md_table(df: pd.DataFrame) -> str:
    def cell(v):
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.4g}"
        if isinstance(v, pd.Timestamp):
            return str(v.date())
        return f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    return "\n".join([head, "|" + "|".join("---" for _ in df.columns) + "|"] +
                     ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)])


def describe(spec: S.InstrumentSpec, raw: pd.DataFrame, cfg: S.StudyConfig, start: str | None = None) -> dict:
    """Counts of one instrument for one sample (start overrides the instrument's start)."""
    clean_cfg = spec.clean_config(start)
    cleaned, bars = clean_m1_full(raw, clean_cfg)
    rv = spec.rv_config()
    d = daily_measures(cleaned, rv)
    excluded_bars = int(sum(((cleaned["ts"].dt.hour * 60 + cleaned["ts"].dt.minute).between(a, b - 1)).sum()
                            if a < b else ((cleaned["ts"].dt.hour * 60 + cleaned["ts"].dt.minute) >= a).sum()
                            + ((cleaned["ts"].dt.hour * 60 + cleaned["ts"].dt.minute) < b).sum()
                            for a, b in rv.excluded_windows))
    starts = bars.get("bounce_filter_starts", pd.NaT)
    unfiltered_in_sample = int((cleaned["ts"] < starts).sum()) if pd.notna(starts) else len(cleaned)
    cand = ~d["short"] & ~d["stale"]
    first_day = cand & ~d["anchor"] & ~d["kept"]            # the first candidate has no previous close, hence no return
    days = {
        "days_with_bars": len(d),
        "flagged_short": int(d["short"].sum()),
        "flagged_stale": int(d["stale"].sum()),
        "short_and_stale": int((d["short"] & d["stale"]).sum()),
        "removed_short": int(d["short"].sum()),
        "removed_stale_not_short": int((d["stale"] & ~d["short"]).sum()),
        "removed_anchor": int(d["anchor"].sum()),
        "first_day_no_return": int(first_day.sum()),
        "retained": int(d["kept"].sum()),
    }
    # identity: every day is in exactly one class
    parts = days["removed_short"] + days["removed_stale_not_short"] + days["removed_anchor"] + days["first_day_no_return"] + days["retained"]
    assert parts == days["days_with_bars"], (spec.symbol, parts, days)
    assert bars["rows_in"] - sum(bars.get(k, 0) for k in ("bad_rows", "duplicates", "saturday_bars", "bounce_back", "outside_period")) == bars["rows_out"] == len(cleaned)
    kept_days = d.index[d["kept"]]
    n_ret = len(kept_days)
    w, b = cfg.window, cfg.burn_in
    out = {"spec": spec, "bars": bars, "excluded_bars": excluded_bars, "unfiltered_in_sample": unfiltered_in_sample, "days": days, "daily": d, "kept_days": kept_days,
           "first_retained": kept_days[0] if n_ret else None, "last_retained": kept_days[-1] if n_ret else None,
           "burn_in_last": kept_days[b - 1] if n_ret >= b else None,
           "first_window": (kept_days[b], kept_days[b + w - 1]) if n_ret >= b + w else None,
           "first_forecast": kept_days[b + w] if n_ret > b + w else None,
           "forecast_days": max(n_ret - b - w, 0), "start": start or spec.start, "end": spec.end}
    return out


def by_year(d: pd.DataFrame) -> pd.DataFrame:
    cand = ~d["short"] & ~d["stale"]
    return pd.DataFrame({"days": d.groupby(d.index.year).size(), "short": d["short"].groupby(d.index.year).sum(),
                         "stale_not_short": (d["stale"] & ~d["short"]).groupby(d.index.year).sum(),
                         "anchor": d["anchor"].groupby(d.index.year).sum(),
                         "retained": d["kept"].groupby(d.index.year).sum()}).astype(int).rename_axis("year").reset_index().astype({"year": str})


def section(title: str, r: dict, cfg: S.StudyConfig) -> str:
    spec, b, dd = r["spec"], r["bars"], r["days"]
    L = [f"### {title}", "",
         f"- Sample: {r['start']} to {r['end']}; asset class: {spec.asset_class}",
         f"- First retained day: **{r['first_retained'].date()}**; last retained day: **{r['last_retained'].date()}**; "
         f"retained days: **{dd['retained']:,}**",
         f"- Burn-in ({cfg.burn_in} retained days) ends {r['burn_in_last'].date() if r['burn_in_last'] is not None else 'n/a'}; "
         f"first training window ({cfg.window} days) "
         + (f"{r['first_window'][0].date()} to {r['first_window'][1].date()}" if r["first_window"] else "n/a")
         + f"; first forecast day: {r['first_forecast'].date() if r['first_forecast'] is not None else 'n/a'}; "
         f"forecast days available at h = 1: {r['forecast_days']:,}", "",
         "Bars (rules of 12.7; the order is the order of the rules):", "",
         md_table(pd.DataFrame([
             ("bars in the file", b["rows_in"]),
             ("non-positive price or high < low", b["bad_rows"]),
             ("duplicate timestamps (last row kept)", b["duplicates"]),
             ("Saturday bars", b["saturday_bars"]),
             ("bounce-back filter", b["bounce_back"]),
             ("  of which flagged with a robust SD of zero", b["bounce_zero_scale_flagged"]),
             ("  bars on the first 20 trading days of the file (not filterable)", b["bounce_unfiltered_bars"]),
             ("  of those, bars inside the sample (the filter starts " + (str(b["bounce_filter_starts"].date())
                                                                       if pd.notna(b["bounce_filter_starts"]) else "never") + ")",
              r["unfiltered_in_sample"]),
             ("before the sample start", b["before_start"]),
             ("after the sample end", b["after_end"]),
             ("bars in the sample", b["rows_out"]),
             ("  of which stamped 23:00-00:59 (excluded window, not used for any price)", r["excluded_bars"]),
         ], columns=["rule", "bars"])), "",
         "Days (rules of 12.6, in the order short, stale, anchor):", "",
         md_table(pd.DataFrame([
             ("server days with at least one bar", dd["days_with_bars"]),
             ("short days (fewer than 132 intervals with a bar)", dd["removed_short"]),
             ("stale days not already short (more than 50% zero one-minute returns)", dd["removed_stale_not_short"]),
             ("anchor days (previous candidate more than 5 calendar days earlier)", dd["removed_anchor"]),
             ("first candidate day (no previous close, no return)", dd["first_day_no_return"]),
             ("retained days", dd["retained"]),
             ("memo: days flagged short", dd["flagged_short"]),
             ("memo: days flagged stale", dd["flagged_stale"]),
             ("memo: days flagged both", dd["short_and_stale"]),
         ], columns=["rule", "days"])), "",
         "By year:", "", md_table(by_year(r["daily"])), ""]
    anchors = r["daily"].index[r["daily"]["anchor"]]
    L += [f"Anchor days: {', '.join(str(x.date()) for x in anchors) if len(anchors) else 'none'}", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=CONFIG)
    args = ap.parse_args()
    cfg = S.StudyConfig.from_json(args.config)
    symbols = [s.symbol for s in cfg.instruments]
    hashes = verify_raw_hashes(symbols)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_rows, long_rows, sections, long_sections = [], [], [], []
    for spec in cfg.instruments:
        print(f"{spec.symbol} ...", flush=True)
        raw = load_m1(raw_path(spec.symbol))
        r = describe(spec, raw, cfg)
        r["daily"].assign(kept=r["daily"]["kept"]).to_csv(
            OUT_DIR / f"{spec.symbol}_days.csv", columns=["bars", "n_active", "zero_share", "short", "stale", "anchor", "kept"],
            index_label="server_day", float_format="%.6f")
        sections.append(section(spec.symbol, r, cfg))
        main_rows.append(r)
        if spec.symbol in cfg.long_history:
            rl = describe(spec, raw, cfg, start=cfg.long_history[spec.symbol])
            long_sections.append(section(f"{spec.symbol} (own minute-data history)", rl, cfg))
            long_rows.append(rl)
        del raw

    def overview(rows):
        return pd.DataFrame([{
            "instrument": r["spec"].symbol, "first retained": r["first_retained"], "last retained": r["last_retained"],
            "retained days": r["days"]["retained"], "short": r["days"]["removed_short"],
            "stale (not short)": r["days"]["removed_stale_not_short"], "anchor": r["days"]["removed_anchor"],
            "duplicates": r["bars"]["duplicates"], "non-positive / high<low": r["bars"]["bad_rows"],
            "bounce-back bars": r["bars"]["bounce_back"], "Saturday bars": r["bars"]["saturday_bars"],
            "first forecast": r["first_forecast"], "forecast days": r["forecast_days"]} for r in rows])

    L = ["# Sample definition", "",
         f"Generated by `src/sample_definition.py` on {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC from the raw files in "
         "`data/raw/` (SHA-256 verified against `results/raw_data_sha256.tsv`) with the rules of `PRESPEC.md` 12.5-12.7 as "
         f"configured in `{args.config.relative_to(ROOT)}`. Data-quality counts only: no forecast, return statistic or model was "
         "computed. Day-level files per instrument: `results/sample_definition/<SYMBOL>_days.csv`.", "",
         f"- PRESPEC version of the configuration: {cfg.prespec_version}",
         f"- Sample: {cfg.instruments[0].start} to {cfg.instruments[0].end}; burn-in {cfg.burn_in} retained days; "
         f"training window {cfg.window} days; refit every {cfg.refit_every} days",
         "- Rules: bars stamped 23:00-00:59 excluded from prices; nominal day 264 intervals, short below 132; stale above 50% zero "
         "one-minute returns; anchor after more than 5 calendar days; bounce-back above 20 robust SD with at least 75% reversed in "
         "the next minute (robust SD = 1.4826 x MAD of one-minute returns over the previous 20 trading days)",
         "", "## How the rules were read where the text leaves room", "",
         "- Stale days use one-minute returns between consecutive bars (exactly one minute apart) outside the excluded window; "
         "a pair that straddles a gap or the window is not a one-minute return.",
         "- The bounce-back filter runs on all bars before the window is excluded and before the sample is cut, so the first days "
         "of the sample have their past days. It needs 20 trading days of history; on the first 20 days of a file nothing is "
         "flagged (counted below). It is one pass: a flagged bar is removed after all bars were judged on the original returns. "
         "A return is reversed by at least 75% if the next one-minute return has the opposite sign and at least 75% of its size "
         "(an overshoot counts). A robust SD of zero is applied as written (any reversed move is then flagged); the count is shown.",
         "- A day with bars only inside the excluded window has no price and counts as a short day.",
         "- 'Day' means server calendar day; there are no Sunday bars on this server.",
         "", "## Raw files", "", *[f"- `{k}`  {v}" for k, v in hashes.items()], "",
         "## Overview: study panel", "", md_table(overview(main_rows)), "",
         "## Overview: robustness check 6 (each instrument on its own minute-data history)", "", md_table(overview(long_rows)), "",
         "## Study panel by instrument", "", *sections, "## Robustness check 6 by instrument", "", *long_sections]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
