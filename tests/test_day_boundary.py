"""The data of a server day must not depend on later days: the bounce-back filter at midnight."""
import numpy as np

from rvf.data import CleanConfig, bounce_back_mask, clean_m1
from rvf.realized import RVConfig, daily_measures

from test_rules import NO_STALE, expected_scale, noisy_days, to_m1

NO_WINDOW = RVConfig(**{**NO_STALE, "excluded_windows": ()})                          # the robustness target without the rollover window


def spike_over_midnight(seed=6):
    r, days = noisy_days(40, seed=seed)
    sc = expected_scale(r, 30)
    r[30, 1439] = 25 * sc                                                              # the bar stamped 23:59 of day 30 jumps ...
    r[31, 0] = -25 * sc                                                                # ... and the bar stamped 00:00 of day 31 reverses it
    return r, days, sc


def test_the_last_bar_of_a_server_day_is_not_judged_by_the_first_bar_of_the_next():
    """A bar stamped 23:59 is not flagged by the 00:00 bar of the next server day (PRESPEC 7: no information after the close of day t)."""
    r, days, sc = spike_over_midnight()
    m1 = to_m1(r, days)
    mask, info = bounce_back_mask(m1, 20)
    assert mask.sum() == 0 and info["bounce_back"] == 0
    inside = r.copy()                                                                  # the same spike one minute earlier, inside the day
    inside[30, 1439], inside[31, 0] = 0.0, 0.0
    inside[30, 1438], inside[30, 1439] = 25 * sc, -25 * sc
    mask = bounce_back_mask(to_m1(inside, days), 20)[0]
    assert np.flatnonzero(mask).tolist() == [30 * 1440 + 1438]
    start = r.copy()                                                                   # and a spike at 00:00 that reverses at 00:01 is judged as usual
    start[31, 0], start[31, 1], start[30, 1439] = 25 * sc, -25 * sc, 0.0
    assert np.flatnonzero(bounce_back_mask(to_m1(start, days), 20)[0]).tolist() == [31 * 1440]


def test_the_cleaned_bars_and_the_daily_measures_of_a_day_do_not_change_when_later_days_are_added():
    """What the day-t forecast origin sees is the same with and without the bars after day t, also without the rollover window."""
    r, days, sc = spike_over_midnight(seed=7)
    full = to_m1(r, days)
    upto = full.iloc[: 31 * 1440].reset_index(drop=True)                               # data that end with the 23:59 bar of day 30
    cleaned_full, _ = clean_m1(full, CleanConfig(bounce_sd=20))
    cleaned_upto, counts = clean_m1(upto, CleanConfig(bounce_sd=20))
    assert counts["bounce_back"] == 0
    assert cleaned_full.iloc[: len(cleaned_upto)].reset_index(drop=True).equals(cleaned_upto)
    a = daily_measures(cleaned_full, NO_WINDOW).loc[days[:31]]
    b = daily_measures(cleaned_upto, NO_WINDOW)
    assert a["close"].iloc[30] == full["close"].iloc[30 * 1440 + 1439]                 # the 23:59 bar stays: it is the close of day 30
    cols = ["open", "high", "low", "close", "rv", "rq", "r_cc", "n_returns"]
    assert a[cols].reset_index(drop=True).equals(b[cols].reset_index(drop=True))
