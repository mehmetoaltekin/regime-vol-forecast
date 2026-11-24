"""Mutation-testing gaps of the data lane: bar cleaning (rvf.data), the excluded-window day rows (rvf.realized) and the counts
written by sample_definition.py. Every expectation is derived by hand from how the synthetic data were built."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import rvf.study as S
from rvf.data import CleanConfig, bounce_back_mask, clean_m1, clean_m1_full
from rvf.realized import RVConfig, daily_measures, retained

from test_realized import bars
from test_rules import ALL, NO_STALE, PANEL, day_bars, expected_scale, noisy_days, plant, series, to_m1
from test_sample_definition import CLEAN, RULES

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import sample_definition as SD  # noqa: E402


# ------------------------------------------------------------------------------------------ rvf.data

def test_bounce_back_judges_bars_just_after_the_sample_start_with_the_history_before_it():
    """PRESPEC 12.7: the bounce-back filter runs on all bars before the sample is cut, so a spike a few days into the sample has its 20 past days."""
    r, days = noisy_days(40, seed=31)
    sc = expected_scale(r, 30)
    m1 = to_m1(plant(r, 30, 604, 25, 1.0, sc), days)
    spike_ts = m1["ts"].iloc[30 * 1440 + 604]
    out, counts = clean_m1_full(m1, CleanConfig(start=str(days[25].date()), bounce_sd=20))
    assert counts["before_start"] == 25 * 1440                                         # only 15 days are left in the sample ...
    assert counts["bounce_back"] == 1                                                  # ... and the spike on its sixth day is still flagged
    assert counts["bounce_unfiltered_bars"] == 20 * 1440 and counts["bounce_filter_starts"] == days[20]   # counted on the uncut file
    assert spike_ts not in set(out["ts"]) and counts["rows_out"] == 15 * 1440 - 1
    legacy_out, legacy = clean_m1(m1, CleanConfig(start=str(days[25].date()), bounce_sd=20))
    assert legacy["bounce_back"] == 1 and len(legacy_out) == len(out)


def test_bounce_back_filter_starts_on_the_twenty_first_trading_day_of_the_file():
    """PRESPEC 12.7: nothing is flagged on the first 20 trading days; the 21st day (index 20) is the first one with a robust SD."""
    r, days = noisy_days(40, seed=32)
    sc = expected_scale(r, 20)                                                         # history of day 20: days 0-19
    on_day_20, info = bounce_back_mask(to_m1(plant(r, 20, 604, 30, 1.0, sc), days), 20)
    assert on_day_20[20 * 1440 + 604] and on_day_20.sum() == 1 and info["bounce_back"] == 1
    assert info["bounce_unfiltered_bars"] == 20 * 1440 and info["bounce_filter_starts"] == days[20]
    on_day_19, info = bounce_back_mask(to_m1(plant(r, 19, 604, 400, 1.0, sc), days), 20)   # a huge spike, but only 19 past days
    assert on_day_19.sum() == 0 and info["bounce_back"] == 0
    assert info["bounce_unfiltered_bars"] == 20 * 1440 and info["bounce_filter_starts"] == days[20]


def test_every_non_positive_price_and_every_inverted_range_is_a_bad_row():
    """PRESPEC 12.7: a bar with a non-positive open, high, low or close, or with high below low, is removed and counted as a bad row."""
    def frame():
        return bars(pd.bdate_range("2024-01-02", periods=3), np.arange(60, 70), 100.0 + 0.1 * np.arange(30))
    for col in ("open", "low", "close"):
        for value in (0.0, -1.0):
            m1 = frame()
            m1.loc[7, col] = value                                                     # the other three prices stay positive
            out, counts = clean_m1_full(m1, CleanConfig())
            assert counts["bad_rows"] == 1 and counts["rows_out"] == 29 and m1["ts"].iloc[7] not in set(out["ts"]), (col, value)
    m1 = frame()
    m1.loc[7, ["high", "low"]] = 0.0                                                   # a zero high (with its low): positive close
    assert clean_m1_full(m1, CleanConfig())[1]["bad_rows"] == 1
    m1 = frame()
    m1.loc[7, ["high", "low"]] = [99.0, 101.0]                                         # all positive, high < low
    assert clean_m1_full(m1, CleanConfig())[1]["bad_rows"] == 1
    # one of each in one frame: five bad rows; a flat bar (high == low) and tiny positive prices are fine
    m1 = frame()
    m1.loc[3, "open"] = -1.0
    m1.loc[8, "low"] = -2.0
    m1.loc[12, ["high", "low"]] = 0.0
    m1.loc[15, "close"] = 0.0
    m1.loc[20, ["high", "low"]] = [99.0, 101.0]
    m1.loc[21, ["open", "high", "low", "close"]] = 100.0
    m1.loc[25, ["open", "high", "low", "close"]] = 1e-9
    out, counts = clean_m1_full(m1, CleanConfig())
    assert counts["bad_rows"] == 5 and counts["rows_out"] == 25
    assert set(out["ts"]) == set(m1["ts"]) - set(m1["ts"].iloc[[3, 8, 12, 15, 20]])
    off, counts = clean_m1_full(m1, CleanConfig(drop_bad_rows=False))
    assert "bad_rows" not in counts and counts["rows_out"] == 30


# ------------------------------------------------------------------------------------------ rvf.realized

def test_a_day_with_bars_only_inside_the_excluded_window_keeps_its_row_and_is_short():
    """PRESPEC 12.6: a day whose bars all lie in 23:00-00:59 has no price; it keeps a row, counts as short and is removed from the daily series."""
    days = pd.bdate_range("2024-01-02", periods=3)
    level = lambda day, base: day_bars(day, np.arange(60, 1380), np.full(1320, base) + 0.5 * (np.arange(1320) % 2))
    inside = day_bars(days[1], np.arange(10, 31), 90.0 + np.arange(21))               # 00:10-00:30 only
    d = daily_measures(series([level(days[0], 100), inside, level(days[2], 103)]), RVConfig(**PANEL))
    assert list(d.index) == list(days)                                                 # the middle day is not dropped
    mid = d.loc[days[1]]
    assert mid["bars"] == 21 and mid["n_active"] == 0 and mid["n_returns"] == 0
    assert bool(mid["short"]) and not bool(mid["stale"]) and not bool(mid["kept"]) and not bool(mid["anchor"])
    assert np.isnan(mid["close"]) and np.isnan(mid["open"]) and np.isnan(mid["rv"]) and np.isnan(mid["zero_share"])
    # its prices leave no trace: the next day is measured from the last retained candidate's close
    assert list(retained(d).index) == [days[2]]
    assert d.loc[days[2], "r_cc"] == pytest.approx(100 * np.log(d.loc[days[2], "close"] / d.loc[days[0], "close"]))
    assert not d.loc[days[0], "short"] and not d.loc[days[2], "short"]


# ------------------------------------------------------------------------------------------ sample_definition.py: counts

# Server days of the planted file (positions in `DAYS`): 53 weekdays from 27 Nov 2023 with seven missing (8-16 Jan 2024).
SHORT = (10, 12, 17, 20, 26, 33, 44)             # bars from 03:00 on are missing: 24 active intervals in the window logic
FROZEN = (12, 15, 38, 42, 47)                    # one price all day (12 is short and frozen)
SPIKES = (25, 40)                                # one-minute spikes of 60 robust SD that reverse completely
SAMPLE = ("2023-12-05", "2024-02-14")            # sample days 6-50; days 0-5 are before the start, 51-52 after the end
ANCHOR_DAY = 30                                  # 17 Jan 2024: the previous candidate is 5 Jan, 12 calendar days earlier


@pytest.fixture(scope="module")
def planted():
    days = pd.bdate_range("2023-11-27", periods=60).delete(list(range(30, 37)))
    assert len(days) == 53 and days[29] == pd.Timestamp("2024-01-05") and days[ANCHOR_DAY] == pd.Timestamp("2024-01-17")
    r = np.random.default_rng(41).normal(0, 0.01, (53, 1440))
    planted_r = r
    for i in SPIKES:
        planted_r = plant(planted_r, i, 604, 60, 1.0, expected_scale(r, i))
    m1 = to_m1(planted_r, days)
    day = m1["ts"].dt.normalize()
    for i in FROZEN:
        rows = day == days[i]
        m1.loc[rows, ["open", "high", "low", "close"]] = m1.loc[rows, "close"].iloc[0]
    pos = lambda i, minute: i * 1440 + minute
    fridays = [i for i in range(6, 53) if days[i].day_of_week == 4][:3]
    saturday = m1.iloc[[pos(i, 700) for i in fridays]].assign(ts=lambda d: d["ts"] + pd.Timedelta(days=1))
    dup = m1.iloc[[pos(8, 100)]].assign(close=lambda d: d["close"] * 1.0001)
    bad_at = [pos(30, 300), pos(31, 200), pos(41, 500), pos(46, 900)]
    bad = m1.iloc[bad_at].assign(low=lambda d: d["high"] * 2)                         # high < low on four bars of the sample
    short_bars = day.isin([days[i] for i in SHORT]) & (m1["ts"].dt.hour >= 3)
    base = m1[~short_bars].drop(index=bad_at)
    raw = pd.concat([base, saturday, dup, bad], ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)
    return raw, days


def sample_spec(rules=RULES):
    return S.InstrumentSpec("SYN", SAMPLE[0], SAMPLE[1], rules, "FX", CLEAN)


def planted_expectations(days):
    """Hand-derived counts of the planted file (see the constants above)."""
    in_sample = list(range(6, 51))
    removed = set(SHORT) | set(FROZEN) | {ANCHOR_DAY}
    kept = [i for i in in_sample[1:] if i not in removed]                              # day 6 is the first candidate: no return
    exp = {"rows_in": 53 * 1440 - 7 * 1260 + 3 + 1,                                   # seven days lost 1260 bars; 3 Saturday bars, 1 duplicate
           "bad_rows": 4, "duplicates": 1, "saturday_bars": 3, "bounce_back": 2, "before_start": 6 * 1440, "after_end": 2 * 1440,
           "rows_out": 45 * 1440 - 7 * 1260 - 4 - 2, "zero_scale": 0,
           "unfiltered_bars": 20 * 1440 - 3 * 1260,                                    # days 0-19; days 10, 12 and 17 are short
           "unfiltered_in_sample": 14 * 1440 - 3 * 1260,                               # days 6-19
           "excluded_bars": 38 * 120 + 7 * 60,                                         # 23:00-00:59 holds 120 bars of a full day, 60 of a short one
           "days_with_bars": 45, "short": 7, "stale_not_short": 4, "anchor": 1, "first_day": 1, "retained": len(kept),
           "flagged_short": 7, "flagged_stale": 5, "both": 1}
    assert exp["retained"] == 32 and exp["rows_out"] == 55974
    return exp, kept


def test_describe_counts_each_rule_of_the_planted_file_exactly(planted):
    """Every bar and day count of the sample-definition report (PRESPEC 12.5-12.7) equals the hand count of the planted defects."""
    raw, days = planted
    exp, kept = planted_expectations(days)
    cfg = S.StudyConfig("t", [sample_spec()], window=10, burn_in=5)
    r = SD.describe(sample_spec(), raw, cfg)
    b, d = r["bars"], r["days"]
    assert len(raw) == exp["rows_in"] and b["rows_in"] == exp["rows_in"]
    for key in ("bad_rows", "duplicates", "saturday_bars", "bounce_back", "before_start", "after_end", "rows_out"):
        assert b[key] == exp[key], key
    assert b["bounce_unfiltered_bars"] == exp["unfiltered_bars"] and b["bounce_filter_starts"] == days[20]
    assert b["bounce_zero_scale_flagged"] == exp["zero_scale"]
    assert r["excluded_bars"] == exp["excluded_bars"]                                  # 23:00-00:59 inclusive of 23:00 and 00:00, exclusive of 01:00
    assert r["unfiltered_in_sample"] == exp["unfiltered_in_sample"]                    # the first bar of day 20 itself is already filtered
    assert d == {"days_with_bars": 45, "flagged_short": 7, "flagged_stale": 5, "short_and_stale": 1, "removed_short": 7,
                 "removed_stale_not_short": 4, "removed_anchor": 1, "first_day_no_return": 1, "retained": 32}
    assert list(r["kept_days"]) == [days[i] for i in kept]
    assert list(r["daily"].index[r["daily"]["anchor"]]) == [days[ANCHOR_DAY]]
    assert r["first_retained"] == days[kept[0]] == pd.Timestamp("2023-12-06") and r["last_retained"] == days[50]
    assert r["burn_in_last"] == days[kept[4]] and r["first_window"] == (days[kept[5]], days[kept[14]])
    assert r["first_forecast"] == days[kept[15]] and r["forecast_days"] == 32 - 15
    by_year = SD.by_year(r["daily"])
    assert by_year.to_dict("list") == {"year": ["2023", "2024"], "days": [19, 26], "short": [4, 3],
                                       "stale_not_short": [1, 3], "anchor": [0, 1], "retained": [13, 19]}


def test_describe_edge_cases_for_the_burn_in_window_and_first_forecast_boundaries(planted):
    """The first window and first forecast exist exactly from burn_in + window and burn_in + window + 1 retained days."""
    raw, days = planted
    _, kept = planted_expectations(days)
    k = lambda i: days[kept[i]]
    run = lambda b, w: SD.describe(sample_spec(), raw, S.StudyConfig("t", [sample_spec()], window=w, burn_in=b))
    r = run(32, 5)                                                                     # exactly as many retained days as the burn-in
    assert r["burn_in_last"] == k(31) and r["first_window"] is None and r["first_forecast"] is None and r["forecast_days"] == 0
    r = run(2, 30)                                                                     # exactly burn_in + window retained days
    assert r["burn_in_last"] == k(1) and r["first_window"] == (k(2), k(31))
    assert r["first_forecast"] is None and r["forecast_days"] == 0
    r = run(2, 29)                                                                     # one forecast day
    assert r["first_window"] == (k(2), k(30)) and r["first_forecast"] == k(31) and r["forecast_days"] == 1
    r = run(33, 1)                                                                     # fewer retained days than the burn-in
    assert r["burn_in_last"] is None and r["first_window"] is None and r["first_forecast"] is None and r["forecast_days"] == 0


def test_excluded_bars_sum_wrapping_and_plain_windows_and_unfiltered_bars_without_the_filter(planted):
    """excluded_bars counts the bars of every excluded window, [start, end) for a plain window and 23:00-00:59 for the wrapping one."""
    raw, days = planted
    cfg = S.StudyConfig("t", [sample_spec()], window=10, burn_in=5)
    two = SD.describe(sample_spec({**RULES, "excluded_windows": [[1380, 60], [600, 660]]}), raw, cfg)
    assert two["excluded_bars"] == 38 * (120 + 60) + 7 * 60 - 2                        # 10:00-10:59 exists on full days only; the 2 spike bars (10:04) are gone
    plain = SD.describe(sample_spec({**RULES, "excluded_windows": [[600, 660]]}), raw, cfg)
    assert plain["excluded_bars"] == 38 * 60 - 2
    none = SD.describe(sample_spec({**RULES, "excluded_windows": []}), raw, cfg)
    assert none["excluded_bars"] == 0
    # without the bounce-back filter every bar of the sample is unfiltered (and nothing is removed by it)
    unf = S.InstrumentSpec("SYN", *SAMPLE, RULES, "FX", {})
    r = SD.describe(unf, raw, cfg)
    assert r["bars"]["rows_out"] == 55974 + 2 and r["unfiltered_in_sample"] == 55974 + 2 and "bounce_back" not in r["bars"]


# ------------------------------------------------------------------------------------------ sample_definition.py: the written report

def parse_table(text, heading):
    """Rows (dicts of the stripped cell strings) of the first markdown table after the line that starts with `heading`."""
    lines = text.splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith(heading))
    rows = []
    for line in lines[i + 1:]:
        if line.startswith("|"):
            rows.append([c.strip() for c in line.strip()[1:-1].split("|")])
        elif rows:
            break
    return [dict(zip(rows[0], row)) for row in rows[2:]]


def cell(rows, label_start, column):
    hits = [row[column] for row in rows if next(iter(row.values())).startswith(label_start)]
    assert len(hits) == 1, (label_start, hits)
    return hits[0]


def test_section_writes_every_count_of_the_planted_file_in_its_own_row(planted):
    """Each row of the bar and day tables, the by-year table and the headline lines of a section carry the hand-counted value."""
    raw, days = planted
    exp, kept = planted_expectations(days)
    cfg = S.StudyConfig("t", [sample_spec()], window=10, burn_in=5)
    text = SD.section("SYN", SD.describe(sample_spec(), raw, cfg), cfg)
    bars_t = parse_table(text, "Bars (rules of 12.7")
    for label, key in (("bars in the file", "rows_in"), ("non-positive price", "bad_rows"), ("duplicate timestamps", "duplicates"),
                       ("Saturday bars", "saturday_bars"), ("bounce-back filter", "bounce_back"),
                       ("of which flagged with a robust SD", "zero_scale"), ("bars on the first 20 trading days", "unfiltered_bars"),
                       ("of those, bars inside the sample", "unfiltered_in_sample"), ("before the sample start", "before_start"),
                       ("after the sample end", "after_end"), ("bars in the sample", "rows_out"), ("of which stamped 23:00-00:59", "excluded_bars")):
        assert cell(bars_t, label, "bars") == f"{exp[key]:,}", label                   # thousands separators included
    assert f"the filter starts {days[20].date()}" in cell(bars_t, "of those, bars inside the sample", "rule") + ")"
    days_t = parse_table(text, "Days (rules of 12.6")
    for label, key in (("server days with at least one bar", "days_with_bars"), ("short days", "short"), ("stale days not already short", "stale_not_short"),
                       ("anchor days", "anchor"), ("first candidate day", "first_day"), ("retained days", "retained"),
                       ("memo: days flagged short", "flagged_short"), ("memo: days flagged stale", "flagged_stale"),
                       ("memo: days flagged both", "both")):
        assert cell(days_t, label, "days") == str(exp[key]), label
    year_t = parse_table(text, "By year:")
    assert [[row[c] for c in ("year", "days", "short", "stale_not_short", "anchor", "retained")] for row in year_t] == \
        [["2023", "19", "4", "1", "0", "13"], ["2024", "26", "3", "3", "1", "19"]]
    assert f"First retained day: **{days[kept[0]].date()}**; last retained day: **{days[50].date()}**; retained days: **32**" in text
    assert (f"Burn-in (5 retained days) ends {days[kept[4]].date()}; first training window (10 days) {days[kept[5]].date()} to "
            f"{days[kept[14]].date()}; first forecast day: {days[kept[15]].date()}; forecast days available at h = 1: 17") in text
    assert f"Anchor days: {days[ANCHOR_DAY].date()}" in text


def test_md_table_formats_integers_floats_booleans_and_dates():
    """The markdown tables use thousands separators for integers, four significant digits for floats, empty cells for NaN and plain dates."""
    df = pd.DataFrame({"i": [1234567, 5], "f": [0.123456789, np.nan], "g": [1234567.0, 2.0], "b": [True, False],
                       "t": [pd.Timestamp("2024-03-05 10:00"), pd.Timestamp("2025-01-01")], "s": ["x", "y"]})
    assert SD.md_table(df).splitlines() == ["| i | f | g | b | t | s |", "|---|---|---|---|---|---|",
                                            "| 1,234,567 | 0.1235 | 1.235e+06 | True | 2024-03-05 | x |",
                                            "| 5 |  | 2 | False | 2025-01-01 | y |"]


# ------------------------------------------------------------------------------------------ sample_definition.py: main()

# four tiny instruments: server days 0-15 from 1 Jan 2024; sample days 5-14; long-history start = day 0
SMOKE = {"AAA": dict(dups=[(7, 100)], sats=[(4, 700), (9, 700)], short=[], frozen=[], bad=[]),
         "BBB": dict(dups=[(6, 100), (7, 100)], sats=[(4, 700), (9, 700), (14, 700)], short=[8], frozen=[], bad=[]),
         "CCC": dict(dups=[(6, 100), (7, 100), (10, 300)], sats=[(9, 700)], short=[11], frozen=[9], bad=[]),
         "DDD": dict(dups=[], sats=[(4, 700), (4, 701), (9, 700), (14, 700)], short=[], frozen=[], bad=[(6, 300), (12, 300)])}


def smoke_raw(plan, seed):
    r, days = noisy_days(16, seed=seed)
    m1 = to_m1(r, days)
    day = m1["ts"].dt.normalize()
    for i in plan["frozen"]:
        rows = day == days[i]
        m1.loc[rows, ["open", "high", "low", "close"]] = m1.loc[rows, "close"].iloc[0]
    pos = lambda i, minute: i * 1440 + minute
    saturday = m1.iloc[[pos(i, mi) for i, mi in plan["sats"]]].assign(ts=lambda d: d["ts"] + pd.Timedelta(days=1))
    dup = m1.iloc[[pos(i, mi) for i, mi in plan["dups"]]].assign(close=lambda d: d["close"] * 1.0001)
    bad_at = [pos(i, mi) for i, mi in plan["bad"]]
    bad = m1.iloc[bad_at].assign(low=lambda d: d["high"] * 2)
    short_bars = day.isin([days[i] for i in plan["short"]]) & (m1["ts"].dt.hour >= 3)
    base = m1[~short_bars].drop(index=bad_at)
    return pd.concat([base, saturday, dup, bad], ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True), days


def test_main_writes_the_report_overviews_and_day_files_for_every_instrument(tmp_path, monkeypatch):
    """main() with the default paths writes one overview row, one section and one day file per instrument and a long-history overview."""
    root = tmp_path.resolve()
    (root / "config").mkdir()
    instruments = [{"symbol": s, "asset_class": "FX" if s in ("AAA", "BBB") else "Precious metals", "start": "2024-01-08", "end": "2024-01-19",
                    "rv": RULES, "clean": {"bounce_sd": 1000, "bounce_reversal": 0.75, "bounce_lookback_days": 3}} for s in SMOKE]   # no bar is flagged
    (root / "config/study_config.json").write_text(json.dumps({
        "prespec_version": "0.4", "instruments": instruments, "window": 3, "burn_in": 2,
        "long_history": {"AAA": "2024-01-01", "CCC": "2024-01-01"}}))
    frames = {s: smoke_raw(plan, seed=50 + k) for k, (s, plan) in enumerate(SMOKE.items())}
    days = frames["AAA"][1]
    monkeypatch.setattr(SD, "ROOT", root)
    monkeypatch.setattr(SD, "CONFIG", root / "config/study_config.json")
    monkeypatch.setattr(SD, "OUT_MD", root / "results/sample_definition.md")
    monkeypatch.setattr(SD, "OUT_DIR", root / "results/sample_definition")
    monkeypatch.setattr(SD, "verify_raw_hashes", lambda symbols: {f"{s}_M1_syn.csv": "00ff" for s in symbols})
    monkeypatch.setattr(SD, "raw_path", lambda symbol: Path(symbol))
    monkeypatch.setattr(SD, "load_m1", lambda path: frames[path.name][0].copy())
    monkeypatch.setattr(sys, "argv", ["sample_definition.py"])
    SD.main()
    text = (root / "results/sample_definition.md").read_text()

    def expected(sym, first, last):
        plan = SMOKE[sym]
        cand = [i for i in range(first, last + 1) if i not in plan["short"] and i not in plan["frozen"]]
        kept = cand[1:]                                                                # the first candidate has no previous close
        return {"instrument": sym, "first retained": str(days[kept[0]].date()), "last retained": str(days[kept[-1]].date()),
                "retained days": str(len(kept)), "short": str(sum(first <= i <= last for i in plan["short"])),
                "stale (not short)": str(sum(first <= i <= last for i in plan["frozen"])), "anchor": "0",
                "duplicates": str(len(plan["dups"])), "non-positive / high<low": str(len(plan["bad"])), "bounce-back bars": "0",
                "Saturday bars": str(len(plan["sats"])), "first forecast": str(days[kept[5]].date()), "forecast days": str(len(kept) - 5)}
    assert parse_table(text, "## Overview: study panel") == [expected(s, 5, 14) for s in SMOKE]
    assert parse_table(text, "## Overview: robustness check 6") == [expected(s, 0, 14) for s in ("AAA", "CCC")]
    assert [e["retained days"] for e in parse_table(text, "## Overview: study panel")] == ["9", "8", "7", "9"]
    assert [e["retained days"] for e in parse_table(text, "## Overview: robustness check 6")] == ["14", "12"]
    for s in SMOKE:
        assert f"### {s}\n" in text
    assert "### AAA (own minute-data history)\n" in text and "### CCC (own minute-data history)\n" in text and "BBB (own" not in text
    assert "retained days: **14**" in text.split("### AAA (own minute-data history)")[1].split("###")[0]
    assert "`AAA_M1_syn.csv`  00ff" in text and "`config/study_config.json`" in text
    for s in SMOKE:
        csv = (root / f"results/sample_definition/{s}_days.csv").read_text().splitlines()
        assert csv[0] == "server_day,bars,n_active,zero_share,short,stale,anchor,kept"
        assert len(csv) == 1 + 10                                                      # one line per server day of the sample
    ccc = pd.read_csv(root / "results/sample_definition/CCC_days.csv", index_col="server_day")
    assert list(ccc.index[ccc["short"]]) == [str(days[11].date())] and list(ccc.index[ccc["stale"]]) == [str(days[9].date())]
    assert not ccc["anchor"].any() and list(ccc.index[~ccc["kept"]]) == [str(days[i].date()) for i in (5, 9, 11)]
