"""Shared synthetic data for the tests."""
import numpy as np
import pandas as pd
import pytest


def make_daily(n=420, seed=0, start="2015-01-01") -> pd.DataFrame:
    """Daily rows like rvf.realized.retained: a two-regime volatility process with persistent regimes."""
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        stay = 0.97 if state[t - 1] == 0 else 0.92
        state[t] = state[t - 1] if rng.random() < stay else 1 - state[t - 1]
    sigma = np.where(state == 0, 0.7, 2.2)                                   # daily volatility in percent
    r = rng.normal(0, sigma)
    close = 100 * np.exp(np.cumsum(r) / 100)
    open_ = np.concatenate([[100.0], close[:-1]])
    wick = np.abs(rng.normal(0, sigma)) / 100
    high = np.maximum(open_, close) * np.exp(wick)
    low = np.minimum(open_, close) * np.exp(-wick)
    rv = sigma ** 2 * rng.chisquare(40, n) / 40
    d = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "r_cc": r, "rv": rv,
                      "rq": rv ** 2 * rng.chisquare(10, n) / 10, "state": state},
                     index=pd.bdate_range(start, periods=n, name="tday"))
    return d


@pytest.fixture(scope="session")
def daily():
    return make_daily()
