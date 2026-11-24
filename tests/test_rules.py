"""Unit tests for the data rules of PRESPEC 0.4 (sections 12.5-12.7): rollover window, day rules, anchor rule, bounce-back filter."""
import numpy as np
import pandas as pd
import pytest

from rvf.data import CleanConfig, bounce_back_mask, clean_m1, clean_m1_full
from rvf.realized import RVConfig, daily_measures, retained

from test_realized import bars

PANEL = dict(excluded_windows=((1380, 60),), ohlc_from_included=True, day_length="active", median_day_length=264.0,
             short_day_share=0.5, stale_share=0.5, anchor_gap_days=5)
NO_STALE = {**PANEL, "stale_share": None}
ALL = np.arange(1440)


def day_bars(day, minutes, close):
    return flat(day, minutes, close)


def flat(days, minutes, close):
    """Bars with open = high = low = close: a bar carries only its own price, none of the previous bar's."""
    m1 = bars(np.atleast_1d(days) if not isinstance(days, pd.Timestamp) else [days], minutes, close)
    for c in ("open", "high", "low"):
        m1[c] = m1["close"]
    return m1


def walk(n, seed, step=0.01):
    """A log-price path of n bars (percent steps of about `step`), as prices."""
    return 100.0 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, step, n)) / 100)


# ------------------------------------------------------------------------------------------ rollover window 23:00-00:59

def test_bars_in_the_rollover_window_do_not_matter_for_any_daily_measure():
    days = pd.bdate_range("2024-01-02", periods=6)
    close = walk(6 * 1440, 1).reshape(6, 1440)
    other = close.copy()
    window = np.r_[0:60, 1380:1440]
    other[:, window] = np.random.default_rng(2).uniform(50, 150, (6, len(window)))      # anything inside the window
    other[2, 1385] = 1e3                                                              # including a huge spike
    a = daily_measures(flat(days, ALL, close.ravel()), RVConfig(**NO_STALE))
    b = daily_measures(flat(days, ALL, other.ravel()), RVConfig(**NO_STALE))
    cols = ["open", "high", "low", "close", "rv", "rv_intraday", "rq", "r_cc", "r_break", "n_returns", "n_active",
            "parkinson", "garman_klass"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    # the window edges: 22:59 is a price, 23:00 and 00:59 are not, 01:00 is again
    d = a.iloc[2]
    assert d["close"] == close[2, 1379] and d["open"] == close[2, 60] and d["n_active"] == 264 and d["n_returns"] == 263
    assert d["high"] == close[2, 60:1380].max() and d["low"] == close[2, 60:1380].min()
    spike = close.copy()
    spike[3, 1379] *= 1.1                                                              # a bar stamped 22:59 is inside the day
    assert daily_measures(flat(days, ALL, spike.ravel()), RVConfig(**NO_STALE)).iloc[3]["close"] == spike[3, 1379]
    spike = close.copy()
    spike[3, 60] *= 1.1                                                                # and so is a bar stamped 01:00
    assert daily_measures(flat(days, ALL, spike.ravel()), RVConfig(**NO_STALE)).iloc[3]["high"] >= spike[3, 60]


def test_the_price_change_across_the_window_is_part_of_the_session_break_return():
    days = pd.bdate_range("2024-01-02", periods=3)
    close = np.full((3, 1440), 100.0)
    close[1, 1380:] = 102.0                 # the level moves inside 23:00-23:59 of day 2 and stays
    close[2, :] = 102.0
    d = daily_measures(flat(days, ALL, close.ravel()), RVConfig(**NO_STALE))
    # day 2 closes at 100 (22:59), day 3 opens at 102 (01:00): the whole move is day 3's session-break return
    assert d.loc[days[1], "close"] == 100.0 and d.loc[days[2], "close"] == 102.0
    assert d.loc[days[2], "r_break"] == pytest.approx(100 * np.log(1.02)) and d.loc[days[2], "rv_intraday"] == 0.0
    assert d.loc[days[2], "r_cc"] == pytest.approx(100 * np.log(1.02)) and d.loc[days[2], "rv"] == pytest.approx((100 * np.log(1.02)) ** 2)
    assert d.loc[days[1], "rv"] == pytest.approx(0.0)                                  # the dip in the window is not day 2's RV
    # a dip that lives only inside the window and is gone by 01:00 leaves no trace at all
    close = np.full((3, 1440), 100.0)
    close[1, 1380:] = 90.0
    close[2, :60] = 90.0
    d = daily_measures(flat(days, ALL, close.ravel()), RVConfig(**NO_STALE))
    assert np.allclose(d["rv"].iloc[1:], 0.0) and np.allclose(d["r_cc"].iloc[1:], 0.0)
    assert d["low"].min() == 100.0 and d["parkinson"].max() == 0.0


# ------------------------------------------------------------------------------------------ short days

def sparse(day, n_intervals):
    """n_intervals consecutive five-minute intervals from 01:00, all minutes present."""
    m = np.arange(60, 60 + 5 * n_intervals)
    return day_bars(day, m, walk(len(m), int(day.day)))


def series(days_and_bars):
    return pd.concat(days_and_bars, ignore_index=True)


def test_short_day_threshold_counts_intervals_with_a_bar():
    days = pd.bdate_range("2024-01-02", periods=7)
    parts = [sparse(days[0], 264), sparse(days[1], 132), sparse(days[2], 131), sparse(days[3], 264)]
    hole = np.r_[60:60 + 5 * 100, 1100:1100 + 5 * 32]                                  # 100 + 32 intervals, a hole in between
    parts.append(day_bars(days[4], hole, walk(len(hole), 5)))
    hole = np.r_[60:60 + 5 * 100, 1100:1100 + 5 * 31]                                  # 100 + 31 intervals
    parts.append(day_bars(days[5], hole, walk(len(hole), 6)))
    parts.append(day_bars(days[6], np.array([60, 1375]), np.array([100.0, 101.0])))    # two bars far apart: 2 intervals
    d = daily_measures(series(parts), RVConfig(**NO_STALE))
    assert list(d["n_active"]) == [264, 132, 131, 264, 132, 131, 2]
    assert list(d["short"]) == [False, False, True, False, False, True, True]          # fewer than 132 is short, 132 is not
    assert list(d["kept"]) == [False, True, False, True, True, False, False]           # day 1 has no previous close


def test_a_removed_day_passes_its_price_change_to_the_next_retained_day():
    days = pd.bdate_range("2024-01-02", periods=4)
    a, c = np.full(1320, 100.0), np.full(1320, 103.0)
    b = np.array([120.0, 121.0])                                                       # a short day with a wild price
    parts = [day_bars(days[0], np.arange(60, 1380), a), day_bars(days[1], np.array([60, 61]), b),
             day_bars(days[2], np.arange(60, 1380), c), day_bars(days[3], np.arange(60, 1380), c)]
    d = daily_measures(series(parts), RVConfig(**NO_STALE))
    assert list(d["short"]) == [False, True, False, False]
    assert d.loc[days[2], "r_cc"] == pytest.approx(100 * np.log(103 / 100))            # from day 1's close, not day 2's
    assert d.loc[days[2], "r_break"] == pytest.approx(100 * np.log(1.03)) and np.isnan(d.loc[days[1], "r_cc"])
    assert d.loc[days[3], "rv"] == pytest.approx(0.0)
    assert list(retained(d).index) == [days[2], days[3]]


# ------------------------------------------------------------------------------------------ stale days

def stale_day(day, zeros, steps=10):
    """steps one-minute returns, `zeros` of them exactly zero (the others a tick up or down)."""
    close = [100.0]
    for i in range(steps):
        close.append(close[-1] if i < zeros else close[-1] + (0.5 if i % 2 else -0.5))
    return day_bars(day, np.arange(60, 60 + steps + 1), np.array(close))


def test_stale_day_needs_more_than_half_zero_returns():
    days = pd.bdate_range("2024-01-02", periods=5)
    parts = [stale_day(days[0], 0), stale_day(days[1], 5), stale_day(days[2], 6), stale_day(days[3], 10), stale_day(days[4], 4)]
    d = daily_measures(series(parts), RVConfig(short_day_share=None, stale_share=0.5, excluded_windows=((1380, 60),)))
    assert list(d["zero_share"]) == [0.0, 0.5, 0.6, 1.0, 0.4]
    assert list(d["stale"]) == [False, False, True, True, False]                       # 0.5 is not more than 0.5
    off = daily_measures(series(parts), RVConfig(short_day_share=None, excluded_windows=((1380, 60),)))
    assert not off["stale"].any() and off["zero_share"].isna().all()


def test_stale_share_ignores_the_window_and_gaps_between_bars():
    days = pd.bdate_range("2024-01-02", periods=2)
    close = walk(1440, 3, step=0.5)                                                    # moving prices ...
    close[:60] = close[1380:] = 100.0                                                  # ... but flat inside the window
    d = daily_measures(day_bars(days[0], ALL, close), RVConfig(**NO_STALE))
    assert d.iloc[0]["zero_share"] == 0.0 or np.isnan(d.iloc[0]["zero_share"]) or d.iloc[0]["zero_share"] < 0.01
    # bars two minutes apart are not one-minute returns: a day of flat bars every other minute has no steps at all
    sparse_flat = day_bars(days[1], np.arange(60, 1380, 2), np.full(660, 100.0))
    d = daily_measures(sparse_flat, RVConfig(**NO_STALE))
    assert np.isnan(d.iloc[0]["zero_share"]) and not d.iloc[0]["stale"]


def test_stale_day_is_removed_and_its_move_goes_to_the_next_retained_day():
    days = pd.bdate_range("2024-01-02", periods=4)
    moving = lambda day, base: day_bars(day, np.arange(60, 1380), base + 0.5 * (np.arange(1320) % 2))
    frozen = day_bars(days[1], np.arange(60, 1380), np.full(1320, 150.0))
    d = daily_measures(series([moving(days[0], 100), frozen, moving(days[2], 110), moving(days[3], 110)]), RVConfig(**PANEL))
    assert list(d["stale"]) == [False, True, False, False] and not d["short"].any()
    assert d.loc[days[2], "r_cc"] == pytest.approx(100 * np.log(d.loc[days[2], "close"] / d.loc[days[0], "close"]))
    assert list(retained(d).index) == [days[2], days[3]]


# ------------------------------------------------------------------------------------------ anchor rule

def test_anchor_rule_gap_of_more_than_five_calendar_days():
    d = pd.Timestamp
    full = lambda day, level: day_bars(day, np.arange(60, 1380), np.full(1320, level) + 0.5 * (np.arange(1320) % 2))
    thin = lambda day: day_bars(day, np.array([60, 61]), np.array([200.0, 201.0]))
    plan = [("2024-01-04", full, 100), ("2024-01-05", full, 101),                        # Thu, Fri
            ("2024-01-08", thin, None), ("2024-01-09", thin, None),                      # removed (short)
            ("2024-01-10", full, 102),                                                   # Wed: previous retained Fri, 5 days: kept
            ("2024-01-11", thin, None), ("2024-01-12", thin, None), ("2024-01-15", thin, None),
            ("2024-01-18", full, 103),                                                   # Thu: previous retained Wed 10th, 8 days: anchor
            ("2024-01-19", full, 104),                                                   # Fri: follows the anchor by one day: kept
            ("2024-01-22", thin, None), ("2024-01-23", thin, None), ("2024-01-24", thin, None),
            ("2024-01-25", full, 105),                                                   # Thu: previous retained Fri 19th, 6 days: anchor
            ("2024-01-26", full, 106)]
    parts = [fn(d(day), lvl) if lvl is not None else fn(d(day)) for day, fn, lvl in plan]
    m1 = series(parts)
    out = daily_measures(m1, RVConfig(**PANEL))
    assert list(out.index[out["anchor"]]) == [d("2024-01-18"), d("2024-01-25")]
    assert list(retained(out).index) == [d("2024-01-05"), d("2024-01-10"), d("2024-01-19"), d("2024-01-26")]
    for day in ("2024-01-18", "2024-01-25"):                                             # anchors: no RV, no return
        assert np.isnan(out.loc[day, "rv"]) and np.isnan(out.loc[day, "r_cc"]) and not out.loc[day, "kept"]
    # the day after an anchor is measured from the anchor's close; a gap of exactly 5 days is not an anchor
    for day, prev in (("2024-01-19", "2024-01-18"), ("2024-01-26", "2024-01-25"), ("2024-01-10", "2024-01-05")):
        assert out.loc[day, "r_cc"] == pytest.approx(100 * np.log(out.loc[day, "close"] / out.loc[prev, "close"]))
    assert out.loc["2024-01-05", "r_cc"] == pytest.approx(100 * np.log(out.loc["2024-01-05", "close"] / out.loc["2024-01-04", "close"]))
    assert np.isnan(out.loc["2024-01-04", "r_cc"]) and not out.loc["2024-01-04", "kept"]            # the first day has no return
    without = daily_measures(m1, RVConfig(**{**PANEL, "anchor_gap_days": None}))
    assert not without["anchor"].any() and list(retained(without).index) == [d(x) for x in (
        "2024-01-05", "2024-01-10", "2024-01-18", "2024-01-19", "2024-01-25", "2024-01-26")]
    # without the anchor rule the day after the gap would carry the multi-day return from 10 January's close
    assert without.loc["2024-01-18", "r_cc"] == pytest.approx(100 * np.log(without.loc["2024-01-18", "close"] / out.loc["2024-01-10", "close"]))
    assert np.isnan(out.loc["2024-01-18", "r_cc"])


def test_the_anchor_gap_is_counted_from_the_previous_candidate_not_from_a_stale_day():
    d = pd.Timestamp
    full = lambda day, level: day_bars(day, np.arange(60, 1380), np.full(1320, level) + 0.5 * (np.arange(1320) % 2))
    frozen = lambda day: day_bars(day, np.arange(60, 1380), np.full(1320, 150.0))
    parts = [full(d("2024-01-04"), 100), full(d("2024-01-05"), 101), frozen(d("2024-01-08")), frozen(d("2024-01-09")),
             frozen(d("2024-01-10")), full(d("2024-01-11"), 102)]                        # Fri 5th -> Thu 11th: 6 days
    out = daily_measures(series(parts), RVConfig(**PANEL))
    assert list(out["stale"]) == [False, False, True, True, True, False]
    assert list(out.index[out["anchor"]]) == [d("2024-01-11")]


# ------------------------------------------------------------------------------------------ bounce-back filter

def noisy_days(n_days, seed=0, step=0.01, minutes=1440):
    """n_days of minute returns (percent) as an array (n_days, minutes) and the day timestamps."""
    return np.random.default_rng(seed).normal(0, step, (n_days, minutes)), pd.bdate_range("2024-01-01", periods=n_days)


def to_m1(r, days, minutes=1440, drop=()):
    close = (100.0 * np.exp(np.cumsum(r.ravel()) / 100))
    m1 = bars(days, np.arange(minutes), close)
    return m1.drop(index=list(drop)).reset_index(drop=True) if len(drop) else m1


def expected_scale(r, i, lookback=20):
    x = r[i - lookback: i].ravel()[1:] if False else r[i - lookback: i].ravel()
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def plant(r, day, minute, size_in_sd, back, scale):
    """Spike of size_in_sd robust SDs at (day, minute), followed by a return of -back * spike; no noise on the next bar."""
    r = r.copy()
    x = size_in_sd * scale
    r[day, minute] = x
    r[day, minute + 1] = -back * x
    return r


def test_bounce_back_flags_spikes_above_20_robust_sd_that_reverse_at_least_75_percent():
    r, days = noisy_days(40, seed=1)
    sc = expected_scale(r, 30)
    flagged = lambda rr: bounce_back_mask(to_m1(rr, days), 20, 0.75, 20)[0]
    pos = 30 * 1440 + 600
    for size, back, expect in ((25, 1.0, True), (25, 0.7501, True), (25, 0.7499, False), (25, 1.5, True),   # reversal share
                               (21, 1.0, True), (19, 1.0, False), (25, 0.0, False), (25, -0.5, False)):  # size; no/continuing move
        f = flagged(plant(r, 30, 600, size, back, sc))
        assert f[pos] == expect, (size, back)
        assert f.sum() == int(expect)                                                    # nothing else is flagged
    f = flagged(plant(r, 30, 600, -25, 1.0, sc))                                       # a spike down
    assert f[pos] and f.sum() == 1
    # the scale is 1.4826 x the MAD of one-minute returns over the 20 previous trading days
    rr = plant(r, 30, 600, 1, 1.0, sc)
    for mult, expect in ((19.9, False), (20.1, True)):
        rr[30, 600] = mult * sc
        rr[30, 601] = -mult * sc
        assert flagged(rr)[pos] == expect


def test_bounce_back_uses_past_data_only_and_needs_twenty_days_of_history():
    r, days = noisy_days(40, seed=2)
    sc = expected_scale(r, 30)
    pos = 30 * 1440 + 600
    base = plant(r, 30, 600, 20.5, 1.0, sc)                                            # just above the threshold
    mask = bounce_back_mask(to_m1(base, days), 20)[0]
    assert mask[pos] and mask.sum() == 1
    # later days do not matter ... (a scale that also used day 31 or 32 would move above the spike)
    for k in (31, 32):
        later = base.copy()
        later[k] *= 40
        assert bounce_back_mask(to_m1(later, days), 20)[0][pos]
    later = base.copy()
    later[31:] *= 40
    assert np.array_equal(bounce_back_mask(to_m1(later, days), 20)[0][: 31 * 1440], mask[: 31 * 1440])
    # ... and neither do the other bars of the spike's own day: the scale comes from the previous days
    own = base.copy()
    own[30, :600] *= 40
    own[30, 602:] *= 40
    assert bounce_back_mask(to_m1(own, days), 20)[0][pos]
    # the previous days do: noisier history raises the threshold above the spike
    loud = base.copy()
    loud[10:30] *= 3
    assert not bounce_back_mask(to_m1(loud, days), 20)[0][pos]
    # and only the 20 days before: day 9 is outside the window of day 30
    old_noise = base.copy()
    old_noise[9] *= 40
    assert bounce_back_mask(to_m1(old_noise, days), 20)[0][pos]
    one_day_older = base.copy()
    one_day_older[10] *= 40
    assert bounce_back_mask(to_m1(one_day_older, days), 20)[0][pos] == (expected_scale(one_day_older, 30) * 20 < 20.5 * sc)
    # on the first 20 days there is no history: nothing is flagged, and the bars are counted
    early = plant(r, 10, 600, 400, 1.0, sc)
    mask, info = bounce_back_mask(to_m1(early, days), 20)
    assert mask.sum() == 0 and info["bounce_unfiltered_bars"] == 20 * 1440 and info["bounce_back"] == 0
    mask, info = bounce_back_mask(to_m1(base, days), 20)
    assert info["bounce_back"] == 1 and info["bounce_zero_scale_flagged"] == 0


def test_bounce_back_leaves_unreversed_moves_and_unjudgeable_bars_alone():
    r, days = noisy_days(40, seed=3)
    sc = expected_scale(r, 30)
    pos = 30 * 1440 + 600
    jump = plant(r, 30, 600, 50, 0.0, sc)                                              # a real jump: no reversal
    assert bounce_back_mask(to_m1(jump, days), 20)[0].sum() == 0
    spike = plant(r, 30, 600, 25, 1.0, sc)
    m1 = to_m1(spike, days, drop=[pos + 1])                                            # the reversing bar is missing
    assert bounce_back_mask(m1, 20)[0].sum() == 0
    m1 = to_m1(spike, days, drop=[pos - 1])                                            # the bar before is missing: no return for the spike
    assert bounce_back_mask(m1, 20)[0].sum() == 0
    a = plant(r, 30, 600, 25, 1.0, sc)
    a[30, 800] = 30 * sc
    a[30, 801] = -29 * sc
    assert sorted(np.flatnonzero(bounce_back_mask(to_m1(a, days), 20)[0])) == [pos, 30 * 1440 + 800]


def test_clean_m1_removes_the_flagged_bar_and_counts_every_rule():
    r, days = noisy_days(40, seed=4)
    sc = expected_scale(r, 30)
    spike = to_m1(plant(r, 30, 600, 25, 1.0, sc), days)
    i_dup, i_bad = 5 * 1440 + 100, 5 * 1440 + 200
    dup = spike.iloc[[i_dup]].assign(close=spike["close"].iloc[i_dup] * 1.0005)       # a duplicate timestamp with other prices
    bad = spike.iloc[[i_bad]].assign(low=spike["high"].iloc[i_bad] * 2)               # high < low on another bar
    raw = pd.concat([spike.drop(index=i_bad), dup, bad], ignore_index=True)
    cfg = CleanConfig(start="2024-01-03", end="2024-02-16", bounce_sd=20)
    out, counts = clean_m1_full(raw, cfg)
    assert counts["rows_in"] == len(spike) + 1 and counts["bad_rows"] == 1 and counts["duplicates"] == 1
    assert counts["bounce_back"] == 1 and counts["saturday_bars"] == 0
    assert counts["before_start"] == 2 * 1440 and counts["after_end"] == 5 * 1440     # 1-2 January; 19-23 February
    assert counts["outside_period"] == counts["before_start"] + counts["after_end"]
    assert counts["rows_out"] == (40 - 2 - 5) * 1440 - 1 - 1                          # the replaced bad row and the spike bar
    assert counts["rows_out"] == len(out) and out["ts"].is_monotonic_increasing and out["ts"].is_unique
    assert pd.Timestamp("2024-01-01 00:00") not in set(out["ts"])
    pos_ts = spike["ts"].iloc[30 * 1440 + 600]
    assert pos_ts not in set(out["ts"])                                                # the spike bar is gone
    kept = out[out["ts"] == spike["ts"].iloc[i_dup]]
    assert len(kept) == 1 and kept["close"].iloc[0] == pytest.approx(dup["close"].iloc[0])      # duplicates: the last row stays
    legacy_out, legacy_counts = clean_m1(raw, cfg)
    assert legacy_counts["bounce_back"] == 1 and "bounce_unfiltered_bars" not in legacy_counts
    off_out, off_counts = clean_m1(raw, CleanConfig(start="2024-01-03", end="2024-02-16"))
    assert "bounce_back" not in off_counts and len(off_out) == len(out) + 1
    # a removed spike no longer enters RV: here it sits on the bar stamped 10:04, the price of the 10:05 grid point
    grid_spike = to_m1(plant(r, 30, 604, 25, 1.0, sc), days)
    rv_raw = daily_measures(grid_spike, RVConfig(**NO_STALE))
    cleaned, counts2 = clean_m1(grid_spike, CleanConfig(bounce_sd=20))
    rv_clean = daily_measures(cleaned, RVConfig(**NO_STALE))
    assert counts2["bounce_back"] == 1 and len(cleaned) == len(grid_spike) - 1
    assert rv_clean.loc[days[30], "rv_intraday"] < rv_raw.loc[days[30], "rv_intraday"] - 0.5 * (25 * sc) ** 2


def test_bounce_back_scale_zero_is_applied_as_written_and_counted():
    days = pd.bdate_range("2024-01-01", periods=30)
    close = np.full(30 * 1440, 100.0)                                                  # a dead market: MAD = 0
    pos = 25 * 1440 + 600
    close[pos] = 100.5
    m1 = bars(days, ALL, close)
    mask, info = bounce_back_mask(m1, 20, 0.75, 20)
    assert mask[pos] and mask.sum() == 1 and info["bounce_zero_scale_flagged"] == 1    # |r| > 20 x 0 and reversed: flagged
