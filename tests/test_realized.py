"""Unit tests for rvf.realized and rvf.data on synthetic M1 data."""
import numpy as np
import pandas as pd
import pytest

from rvf.data import CleanConfig, clean_m1
from rvf.realized import RVConfig, daily_measures, excluded_mask, garman_klass, parkinson, retained


def bars(days, minutes, close):
    """M1 frame from a list of days, the minute-of-day values present on each day, and a flat array of closes."""
    ts = np.concatenate([(pd.Timestamp(d) + pd.to_timedelta(np.asarray(minutes), unit="m")).to_numpy() for d in days])
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"ts": ts, "open": open_, "high": np.maximum(open_, close), "low": np.minimum(open_, close),
                         "close": close, "tickvol": 1, "vol": 0, "spread": 10})


def brownian(n_days, sigma_day_pct, minutes=np.arange(1440), seed=0, start="2024-01-01"):
    """Driftless log-price with daily volatility sigma_day_pct over the 1440 minutes; no overnight move."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    step = sigma_day_pct / 100 / np.sqrt(1440)
    logp = np.cumsum(rng.normal(0, step, n_days * 1440)).reshape(n_days, 1440)
    return bars(days, minutes, 100 * np.exp(logp[:, minutes]).ravel()), days


NO_SHORT = RVConfig(short_day_share=None)


def test_rv_recovers_a_known_variance():
    m1, _ = brownian(400, sigma_day_pct=1.2)
    d = retained(daily_measures(m1, NO_SHORT))
    # one calendar day of a 1.2% daily-vol Brownian motion: integrated variance 1.44 percent squared
    assert d["rv"].mean() == pytest.approx(1.44, rel=0.02)
    assert daily_measures(m1, RVConfig(interval_min=1, short_day_share=None))["rv"].mean() == pytest.approx(1.44, rel=0.02)
    assert d["rq"].mean() == pytest.approx(1.44 ** 2, rel=0.05)                     # integrated quarticity sigma^4
    assert d["parkinson"].mean() == pytest.approx(1.44, rel=0.06)
    assert d["garman_klass"].mean() == pytest.approx(1.44, rel=0.06)


def test_grid_uses_the_last_close_at_or_before_each_grid_point():
    # bars stamped 00:00..00:14; a bar stamped HH:MM closes one minute later
    close = 100 * np.exp(np.arange(1, 16) / 1000.0)                                 # log-price 0.001 per minute
    day0 = bars(["2024-01-01"], [0], [100.0])
    m1 = pd.concat([day0, bars(["2024-01-02"], np.arange(15), close)], ignore_index=True)
    d = daily_measures(m1, NO_SHORT).loc["2024-01-02"]
    # grid points 00:05, 00:10, 00:15 take the closes of the bars stamped 00:04, 00:09, 00:14
    assert d["n_returns"] == 2 and d["n_active"] == 3
    assert d["rv_intraday"] == pytest.approx(2 * (100 * 0.005) ** 2)
    assert d["r_break"] == pytest.approx(100 * 0.005)                              # previous close 100 -> price at 00:05
    assert d["r_cc"] == pytest.approx(100 * 0.015)
    assert d["rv"] == pytest.approx(3 * 0.25)


def test_missing_bars_give_zero_returns_not_fewer_grid_points():
    minutes = np.r_[0:60, 180:240]                                                 # a two-hour hole
    close = np.r_[np.full(60, 100.0), np.full(60, 101.0)]
    m1 = pd.concat([bars(["2024-01-01"], [0], [100.0]), bars(["2024-01-02"], minutes, close)], ignore_index=True)
    d = daily_measures(m1, NO_SHORT).loc["2024-01-02"]
    assert d["n_returns"] == 47 and d["n_active"] == 24                            # grid 00:05 .. 04:00
    assert d["rv_intraday"] == pytest.approx((100 * np.log(1.01)) ** 2)            # the whole move is one grid return


def test_returns_add_up_to_close_to_close_with_windows_sundays_and_gaps():
    m1, days = brownian(30, 1.0, seed=3)
    rng = np.random.default_rng(1)
    m1 = m1[rng.random(len(m1)) > 0.2]                                             # 20% of the bars missing
    sunday = bars(["2024-01-07"], np.arange(1380, 1440), np.linspace(99, 98, 60))  # Sunday 23:00-23:59
    m1 = pd.concat([m1, sunday]).sort_values("ts").reset_index(drop=True)
    cfg = RVConfig(excluded_windows=((1435, 10), (600, 615)), short_day_share=None)
    d = daily_measures(m1, cfg)
    # independent close-to-close return: last M1 close of each trading day, Sunday counted with Monday
    tday = m1["ts"].dt.normalize() + pd.to_timedelta((m1["ts"].dt.day_of_week == 6).astype(int), unit="D")
    last_close = m1.groupby(tday)["close"].last()
    r_cc = 100 * np.log(last_close / last_close.shift(1))
    pd.testing.assert_series_equal(d["r_cc"], r_cc, check_names=False)
    assert np.allclose((d["sum_r"] + d["r_break"]).iloc[1:], r_cc.iloc[1:])
    assert np.allclose(d["rv"].iloc[1:], (d["rv_intraday"] + d["r_break"] ** 2).iloc[1:])
    assert pd.Timestamp("2024-01-07") not in d.index                               # Sunday is part of Monday
    assert d.loc["2024-01-08", "bars"] > d["bars"].median()


def test_excluded_window_moves_go_into_the_session_break_return():
    minutes = np.arange(60, 1380)                                                  # 01:00-22:59
    flat = np.full(len(minutes), 100.0)
    spike = flat.copy()
    spike[(minutes >= 600) & (minutes < 610)] = 99.0                               # a 1% dip for ten minutes at 10:00
    step = flat.copy()
    step[minutes >= 605] = 101.0                                                   # a 1% step inside the window
    day0 = bars(["2024-01-01"], minutes, flat)
    cfg = RVConfig(excluded_windows=((600, 610),), short_day_share=None)

    d_spike = daily_measures(pd.concat([day0, bars(["2024-01-02"], minutes, spike)], ignore_index=True), cfg).loc["2024-01-02"]
    assert d_spike["rv"] == pytest.approx(0.0, abs=1e-12)                          # dip and recovery cancel inside the window
    d_none = daily_measures(pd.concat([day0, bars(["2024-01-02"], minutes, spike)], ignore_index=True), NO_SHORT).loc["2024-01-02"]
    assert d_none["rv"] == pytest.approx(2 * (100 * np.log(0.99)) ** 2)            # without the window: down and up

    d_step = daily_measures(pd.concat([day0, bars(["2024-01-02"], minutes, step)], ignore_index=True), cfg).loc["2024-01-02"]
    assert d_step["rv_intraday"] == pytest.approx(0.0, abs=1e-12)
    assert d_step["r_break"] == pytest.approx(100 * np.log(1.01))                  # nothing is dropped from the day's return
    # bars stamped 10:00-10:09 close at 10:01-10:10: the grid points 10:05 and 10:10 have no included price, so the
    # three grid returns ending at 10:05, 10:10 and 10:15 are replaced by the session-break return
    assert d_step["n_returns"] == d_none["n_returns"] - 3


def test_session_break_can_be_left_out():
    m1, _ = brownian(5, 1.0, seed=2)
    m1["close"] = m1["close"] * np.where(m1["ts"].dt.normalize() >= pd.Timestamp("2024-01-03"), 1.05, 1.0)   # overnight gap
    with_break = daily_measures(m1, NO_SHORT)
    without = daily_measures(m1, RVConfig(include_session_break=False, short_day_share=None))
    assert np.allclose(without["rv"], with_break["rv_intraday"])
    gap_day = pd.Timestamp("2024-01-03")
    assert with_break.loc[gap_day, "rv"] - without.loc[gap_day, "rv"] == pytest.approx(with_break.loc[gap_day, "r_break"] ** 2)
    assert with_break.loc[gap_day, "r_break"] > 4.0


def test_short_days_are_removed_and_their_move_carried_forward():
    m1, days = brownian(12, 1.0, seed=5)
    half = days[6]
    m1 = m1[~((m1["ts"].dt.normalize() == half) & (m1["ts"].dt.hour >= 8))]         # 8 of 24 hours
    d = daily_measures(m1, RVConfig())
    assert d.index[d["short"]].tolist() == [half]
    assert np.isnan(d.loc[half, "rv"]) and np.isnan(d.loc[half, "r_cc"])
    nxt, prev = days[7], days[5]
    assert d.loc[nxt, "r_cc"] == pytest.approx(100 * np.log(d.loc[nxt, "close"] / d.loc[prev, "close"]))
    assert d.loc[nxt, "sum_r"] + d.loc[nxt, "r_break"] == pytest.approx(d.loc[nxt, "r_cc"])
    r = retained(d)
    assert half not in r.index and days[0] not in r.index and len(r) == 10
    # a median fixed in advance gives the same flags without looking at the sample
    fixed = daily_measures(m1, RVConfig(median_day_length=287.0))
    assert fixed["short"].equals(d["short"])


def test_day_length_can_count_active_intervals():
    minutes = np.r_[0:60, 1380:1440]                                               # two active hours, 22 hours apart
    m1, _ = brownian(6, 1.0, minutes=minutes, seed=4)
    span = daily_measures(m1, NO_SHORT)
    assert (span["n_returns"] == 287).all() and (span["n_active"] == 24).all()
    full, _ = brownian(6, 1.0, seed=4, start="2024-02-01")
    both = pd.concat([m1, full], ignore_index=True)
    assert not daily_measures(both, RVConfig(day_length="span"))["short"].any()
    assert daily_measures(both, RVConfig(day_length="active"))["short"].sum() == 6


def test_a_day_does_not_depend_on_later_data():
    m1, days = brownian(20, 1.0, seed=6)
    cfg = RVConfig(excluded_windows=((0, 5),), median_day_length=287.0)
    base = daily_measures(m1, cfg)
    later = m1["ts"] >= days[10]
    changed = m1.copy()
    changed.loc[later, ["open", "high", "low", "close"]] *= 1.5
    cols = ["rv", "rv_intraday", "r_cc", "r_break", "rq", "parkinson", "garman_klass", "n_returns"]
    pd.testing.assert_frame_equal(daily_measures(changed, cfg).loc[: days[9], cols], base.loc[: days[9], cols])
    pd.testing.assert_frame_equal(daily_measures(m1[~later], cfg)[cols], base.loc[: days[9], cols])


def test_excluded_mask_and_range_formulas():
    mask = excluded_mask(((1435, 10), (600, 615)))
    assert mask.sum() == 5 + 10 + 15 and mask[1439] and mask[0] and mask[9] and not mask[10] and mask[600] and not mask[615]
    with pytest.raises(ValueError):
        excluded_mask(((0, 1500),))
    h, l, o, c = pd.Series([110.0]), pd.Series([100.0]), pd.Series([100.0]), pd.Series([105.0])
    hl, co = 100 * np.log(1.1), 100 * np.log(1.05)
    assert parkinson(h, l).iloc[0] == pytest.approx(hl ** 2 / (4 * np.log(2)))
    assert garman_klass(o, h, l, c).iloc[0] == pytest.approx(0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2)


def test_clean_m1_counts_what_it_removes():
    m1 = bars(["2024-01-05", "2024-01-06", "2024-01-08"], np.arange(10), np.full(30, 100.0))   # Fri, Sat, Mon
    m1 = pd.concat([m1, m1.iloc[[3]]], ignore_index=True)                           # one duplicate
    m1.loc[1, "close"] = 0.0
    m1.loc[2, "high"] = 50.0                                                         # high < low
    out, counts = clean_m1(m1, CleanConfig(end="2024-01-05"))
    assert counts == {"rows_in": 31, "bad_rows": 2, "duplicates": 1, "saturday_bars": 10, "outside_period": 10, "rows_out": 8}
    assert out["ts"].is_monotonic_increasing and out["ts"].dt.normalize().nunique() == 1


def test_unsorted_input_is_rejected():
    m1, _ = brownian(2, 1.0)
    with pytest.raises(ValueError):
        daily_measures(m1.iloc[::-1], NO_SHORT)
