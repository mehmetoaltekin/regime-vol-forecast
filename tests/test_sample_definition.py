"""The sample-definition report counts what each rule removes: checked on data with planted defects."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import rvf.study as S

from test_rules import expected_scale, noisy_days, plant, to_m1

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import sample_definition as SD  # noqa: E402

RULES = {"excluded_windows": [[1380, 60]], "ohlc_from_included": True, "day_length": "active", "median_day_length": 264,
         "short_day_share": 0.5, "stale_share": 0.5, "anchor_gap_days": 5}
CLEAN = {"bounce_sd": 20, "bounce_reversal": 0.75, "bounce_lookback_days": 20}


@pytest.fixture(scope="module")
def planted():
    r, days = noisy_days(60, seed=21)
    sc = expected_scale(r, 30)
    m1 = to_m1(plant(r, 30, 604, 25, 1.0, sc), days)                                  # a spike on day 30 (a grid price)
    day = m1["ts"].dt.normalize()
    m1 = m1[~((day == days[35]) & (m1["ts"].dt.hour >= 3))]                           # day 35: three hours of bars only (short)
    flat = day == days[40]
    m1.loc[flat, "close"] = m1.loc[flat, "close"].iloc[0]                             # day 40: a frozen feed (stale)
    saturday = m1.iloc[[4 * 1440 + 700]].assign(ts=lambda d: d["ts"] + pd.Timedelta(days=1))   # day 4 is a Friday: a Saturday bar
    dup = m1.iloc[[20 * 1440 + 10]].assign(close=lambda d: d["close"] * 1.0001)
    raw = pd.concat([m1, saturday, dup], ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)
    return raw, days


def test_describe_counts_every_rule_exactly(planted):
    raw, days = planted
    spec = S.InstrumentSpec("SYN", "2024-01-03", "2024-03-15", RULES, "FX", CLEAN)
    cfg = S.StudyConfig("t", [spec], window=10, burn_in=5)
    r = SD.describe(spec, raw, cfg)
    b, d = r["bars"], r["days"]
    assert (b["bad_rows"], b["duplicates"], b["saturday_bars"], b["bounce_back"]) == (0, 1, 1, 1)
    assert b["before_start"] == 2 * 1440 and b["after_end"] == 5 * 1440              # 1-2 January; 18-22 March
    assert b["bounce_unfiltered_bars"] == 20 * 1440 and b["bounce_zero_scale_flagged"] == 0
    assert b["rows_out"] == b["rows_in"] - b["duplicates"] - b["saturday_bars"] - b["bounce_back"] - b["outside_period"]
    assert r["excluded_bars"] > 0
    assert d["removed_short"] == 1 and d["removed_stale_not_short"] == 1 and d["removed_anchor"] == 0
    assert d["first_day_no_return"] == 1 and d["flagged_short"] == 1 and d["flagged_stale"] == 1 and d["short_and_stale"] == 0
    in_sample = [x for x in days if pd.Timestamp("2024-01-03") <= x <= pd.Timestamp("2024-03-15")]
    assert d["days_with_bars"] == len(in_sample)
    assert d["retained"] == len(in_sample) - 1 - 1 - 1
    kept = r["kept_days"]
    assert days[35] not in kept and days[40] not in kept and kept[0] == days[3]       # the first candidate day has no return
    assert r["first_retained"] == days[3] and r["last_retained"] == in_sample[-1]
    assert r["burn_in_last"] == kept[4] and r["first_window"] == (kept[5], kept[14]) and r["first_forecast"] == kept[15]
    assert r["forecast_days"] == d["retained"] - 15
    by_year = SD.by_year(r["daily"]).iloc[0]
    assert by_year["days"] == d["days_with_bars"] and by_year["retained"] == d["retained"] and by_year["short"] == 1
    text = SD.section("SYN", r, cfg)
    for needle in ("First retained day", "bounce-back filter", "short days", "stale days", "anchor days", "By year"):
        assert needle in text


def test_a_day_that_is_both_short_and_stale_is_counted_once():
    r, days = noisy_days(40, seed=22)
    m1 = to_m1(r, days)
    day = m1["ts"].dt.normalize()
    sparse = (day == days[30]) & (m1["ts"].dt.hour >= 3)
    m1 = m1[~sparse].copy()
    both = m1["ts"].dt.normalize() == days[30]
    m1.loc[both, "close"] = 100.0                                                       # short and frozen
    spec = S.InstrumentSpec("SYN", None, None, RULES, "FX", {})
    cfg = S.StudyConfig("t", [spec], window=5, burn_in=2)
    d = SD.describe(spec, m1, cfg)["days"]
    assert d["flagged_short"] == 1 and d["flagged_stale"] == 1 and d["short_and_stale"] == 1
    assert d["removed_short"] == 1 and d["removed_stale_not_short"] == 0
    assert d["days_with_bars"] == 40 and d["retained"] == 40 - 1 - 1


def test_the_report_shows_a_config_path_given_relative_to_the_working_directory(tmp_path, monkeypatch):
    """`sample_definition.py --config config/study_config.json` crashed after all the computation, when the path was written."""
    import sample_definition as SD
    monkeypatch.chdir(SD.ROOT)
    assert SD.shown(Path("config/study_config.json")) == "config/study_config.json"
    assert SD.shown(SD.ROOT / "config" / "study_config.json") == "config/study_config.json"
    assert SD.shown(Path("config/../config/study_config.json")) == "config/study_config.json"
    outside = tmp_path / "other.json"
    assert SD.shown(outside) == str(outside.resolve())                                 # a file outside the repository is shown in full
