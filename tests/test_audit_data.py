"""Unit tests for src/audit_data.py on synthetic M1 data with planted defects."""
import numpy as np
import pandas as pd
import pytest

import audit_data as ad


def make_m1(start="2024-01-01", weeks=8, open_min=60, close_min=23 * 60, seed=0) -> pd.DataFrame:
    """Monday-Friday bars for every minute in [open_min, close_min), random-walk prices, spread 10."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=5 * weeks)
    ts = np.concatenate([(d + pd.to_timedelta(np.arange(open_min, close_min), unit="m")).to_numpy() for d in days])
    close = 100 * np.exp(np.cumsum(rng.normal(0, 2e-4, len(ts))))
    open_ = np.concatenate([[100.0], close[:-1]])
    return pd.DataFrame({
        "ts": ts, "open": open_, "high": np.maximum(open_, close) * 1.0001, "low": np.minimum(open_, close) * 0.9999,
        "close": close, "tickvol": 10, "vol": 0, "spread": 10,
    })


def test_ranges_formats_inclusive_runs():
    assert ad.ranges([0, 1, 2, 59, 1439]) == "00:00-00:02, 00:59, 23:59"
    assert ad.ranges([]) == "none"


def test_load_m1_reads_the_mt5_export_format(tmp_path):
    p = tmp_path / "X_M1_1_2.csv"
    p.write_bytes(b"<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\r\n"
                  b"2024.01.02\t01:00:00\t1.10\t1.20\t1.00\t1.15\t7\t0\t12\r\n"
                  b"2024.01.02\t01:01:00\t1.15\t1.16\t1.14\t1.15\t3\t0\t11\r\n")
    df = ad.load_m1(p)
    assert list(df["ts"]) == [pd.Timestamp("2024-01-02 01:00"), pd.Timestamp("2024-01-02 01:01")]
    assert df["close"].tolist() == [1.15, 1.15] and df["spread"].tolist() == [12, 11]


def test_basic_checks_count_planted_defects():
    df = make_m1(weeks=1)
    df = pd.concat([df, df.iloc[[5, 6, 7]]], ignore_index=True)        # 3 duplicate timestamps, appended out of order
    df.loc[10, "high"] = df.loc[10, "low"] * 0.5                         # high < low
    df.loc[20, ["open", "close"]] = 0.0                                  # zero price
    b = ad.basic_checks(df)
    assert b["duplicate_timestamps"] == 3
    assert b["out_of_order"] == 1
    assert b["high_lt_low"] == 1
    assert b["nonpositive_price_rows"] == 1
    assert b["bars"] == 5 * 22 * 60 + 3


def test_prepare_removes_duplicates_and_moves_sunday_to_monday():
    df = make_m1(weeks=1)
    sunday = df.iloc[:30].assign(ts=pd.Timestamp("2024-01-07 23:00") + pd.to_timedelta(np.arange(30), unit="m"))
    d = ad.prepare(pd.concat([df, df.iloc[:4], sunday], ignore_index=True))
    assert d["ts"].is_unique and d["ts"].is_monotonic_increasing
    sun = d[d["wd"] == 6]
    assert len(sun) == 30 and (sun["tday"] == pd.Timestamp("2024-01-08")).all()
    w = ad.weekend(d)
    assert w["sunday_bars"] == 30 and w["sunday_days"] == 1 and w["saturday_bars"] == 0
    assert w["sunday_first_bar_mode"] == "23:00"


def test_missing_minutes_finds_a_planted_gap_and_ignores_the_daily_break():
    df = make_m1(weeks=8)
    day = pd.Timestamp("2024-01-17")
    hole = (df["ts"] >= day + pd.Timedelta(hours=10)) & (df["ts"] < day + pd.Timedelta(hours=10, minutes=10))
    d = ad.prepare(df[~hole])
    in_sess, years, _ = ad.session_slots(d)
    assert in_sess[0, 2].sum() == 22 * 60                               # Wednesday: 01:00-22:59 in session
    assert not in_sess[0, 2, :60].any() and not in_sess[0, 5].any()     # the break and Saturday are not
    per_day, gaps = ad.missing_minutes(d, in_sess, years)
    assert per_day.loc[day, "missing"] == 10
    assert per_day["missing"].sum() == 10
    assert len(gaps) == 1
    g = gaps.iloc[0]
    assert g["gap_start"] == day + pd.Timedelta(hours=10) and g["minutes"] == 10 and g["in_session_minutes"] == 10


def test_missing_weekdays_lists_days_without_bars():
    df = make_m1(weeks=2)
    d = ad.prepare(df[df["ts"].dt.normalize() != pd.Timestamp("2024-01-10")])
    assert list(ad.missing_weekdays(d)) == [pd.Timestamp("2024-01-10")]


def test_spread_rule_flags_exactly_the_wide_minutes():
    df = make_m1(weeks=4)
    mod = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df.loc[mod.between(60, 64), "spread"] = 31                           # 5 minutes just above 3 x 10
    df.loc[mod.between(65, 69), "spread"] = 30                           # exactly 3 x: not above, not flagged
    sp = ad.spread_profile(ad.prepare(df))
    assert sp["median_all_bars"] == 10
    assert sp["flagged"] == [60, 61, 62, 63, 64]
    assert sp["by_year"].loc[0, "minutes_over_3x"] == "01:00-01:04"


def test_day_lengths_counts_and_short_day_rule():
    df = make_m1(weeks=4)
    half = pd.Timestamp("2024-01-18")
    df = df[~((df["ts"].dt.normalize() == half) & (df["ts"].dt.hour >= 9))]      # 01:00-08:59 only: 96 of 264 intervals
    sunday = df.iloc[:60].assign(ts=pd.Timestamp("2024-01-07 23:00") + pd.to_timedelta(np.arange(60), unit="m"))
    dl = ad.day_lengths(ad.prepare(pd.concat([df, sunday], ignore_index=True)))
    assert dl["n5_active"].median() == 264 and dl["n5_span"].median() == 263
    assert dl.loc[half, "n5_active"] == 96 and dl.loc[half, "n5_span"] == 95
    assert dl.index[dl["short_n5_active"]].tolist() == [half]
    monday = pd.Timestamp("2024-01-08")                                           # Sunday 23:00-23:59 counted with Monday
    assert dl.loc[monday, "n5_active"] == 264 + 12
    assert dl.loc[monday, "n5_span"] == 263 + 24


def test_jumps_detects_a_planted_spike_and_its_reversal():
    df = make_m1(weeks=2, seed=1)
    i = int(np.flatnonzero(df["ts"] == pd.Timestamp("2024-01-03 12:00"))[0])
    df.loc[i, "close"] = df.loc[i, "close"] * 1.02                       # +2% for one bar, about 100 SD
    big, counts = ad.jumps(ad.prepare(df))
    assert counts[30]["jumps"] == 2 and counts[30]["reverted"] == 1      # the spike (reverted) and the way back
    top = big.iloc[0]
    assert top["ts"] in (pd.Timestamp("2024-01-03 12:00"), pd.Timestamp("2024-01-03 12:01"))
    assert abs(top["r"]) == pytest.approx(100 * np.log(1.02), rel=0.05)
    clean, c0 = ad.jumps(ad.prepare(make_m1(weeks=2, seed=1)))
    assert c0[10]["jumps"] == 0 and len(clean) == 0


def test_dst_shift_detects_a_week_moved_by_one_hour():
    df = make_m1(weeks=12, open_min=0, close_min=1440)
    hour = df["ts"].dt.hour
    week = df["ts"].dt.to_period("W").dt.start_time
    shifted = week == pd.Timestamp("2024-02-12")
    df["tickvol"] = np.where(hour == np.where(shifted, 17, 16), 500, 10)          # the "open" is one hour later that week
    out = ad.dst_shifts(ad.prepare(df))
    assert len(out) == 1
    assert out.loc[0, "from_week"] == pd.Timestamp("2024-02-12") and out.loc[0, "shift_hours"] == 1


def test_eligibility_rule():
    d = lambda s: pd.Timestamp(s)
    ok = {"US500": d("2012-01-03"), "USTEC": d("2013-05-01"), "DE40": d("2012-01-03"), "XAUUSD": d("2010-01-04"),
          "XAGUSD": d("2010-01-04"), "EURUSD": d("2009-01-02"), "USDJPY": d("2014-12-31"), "GBPUSD": d("2009-01-02")}
    _, kept, verdict = ad.eligibility(ok)
    assert kept == ["US500", "USTEC", "XAUUSD", "XAGUSD", "EURUSD", "USDJPY"] and "requirement met" in verdict

    late = dict(ok, USTEC=d("2015-01-02"), XAGUSD=d("2016-01-04"), USDJPY=d("2015-01-01"), GBPUSD=d("2015-06-01"))
    rows, kept, verdict = ad.eligibility(late)
    assert kept == ["US500", "DE40", "XAUUSD", "EURUSD"] and "requirement met" in verdict
    outcome = {r[1]: r[3] for r in rows}
    assert outcome["USTEC"] == "replaced by alternate"
    assert outcome["XAGUSD"] == "dropped (no alternate)"
    assert outcome["USDJPY"] == "dropped (alternate not eligible or missing)"

    both = dict(ok, US500=d("2016-01-04"), USTEC=d("2016-01-04"))               # one alternate cannot replace two
    rows, kept, _ = ad.eligibility(both)
    assert kept == ["DE40", "XAUUSD", "XAGUSD", "EURUSD", "USDJPY"]
    assert {r[1]: r[3] for r in rows}["USTEC"] == "dropped (alternate already used)"

    few = {"US500": d("2012-01-03"), "USTEC": d("2012-01-03"), "DE40": d("2012-01-03")}
    _, kept, verdict = ad.eligibility(few)
    assert kept == ["US500", "USTEC"] and "NOT MET" in verdict


def test_stale_prices_flags_a_frozen_day():
    df = make_m1(weeks=2)
    frozen = df["ts"].dt.normalize() == pd.Timestamp("2024-01-04")
    df.loc[frozen, ["open", "high", "low", "close"]] = 123.0
    st = ad.stale_prices(ad.prepare(df))
    day = st.loc[pd.Timestamp("2024-01-04")]
    assert day["zero_return_share"] == 1.0 and day["longest_flat_run"] == 22 * 60 - 1
    others = st.drop(pd.Timestamp("2024-01-04"))
    assert (others["zero_return_share"] < 0.01).all() and (others["longest_flat_run"] <= 2).all()


def test_dense_start_skips_sparse_early_history():
    dense = make_m1(start="2024-03-04", weeks=12)                       # more dense than sparse days: the reference is the recent median
    daily = pd.DataFrame({"ts": pd.bdate_range("2024-01-01", "2024-03-01")}).assign(
        open=100.0, high=100.0, low=100.0, close=100.0, tickvol=1, vol=0, spread=0)       # one bar per day
    dl = ad.day_lengths(ad.prepare(pd.concat([daily, dense], ignore_index=True)))
    assert ad.dense_start(dl) == pd.Timestamp("2024-03-04")
    assert ad.dense_start(dl.iloc[-10:]) is None                        # fewer days than the 20-day window


def test_missing_runs_groups_consecutive_business_days():
    days = pd.DatetimeIndex(["2025-04-18", "2025-07-17", "2025-07-18", "2025-07-21", "2025-07-22", "2025-12-25"])
    runs = ad.missing_runs(days)
    assert len(runs) == 1
    assert runs.loc[0, "from"] == pd.Timestamp("2025-07-17") and runs.loc[0, "to"] == pd.Timestamp("2025-07-22")
    assert runs.loc[0, "weekdays"] == 4


def test_ranges_summarises_above_the_limit():
    assert ad.ranges([0, 2, 4, 6], limit=3) == "4 minutes in 4 ranges between 00:00 and 00:06"
    assert ad.ranges([0, 2, 4], limit=3) == "00:00, 00:02, 00:04"
