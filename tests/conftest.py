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


def perturb_after(daily: pd.DataFrame, cut: int, seed: int = 99) -> pd.DataFrame:
    """Rows from position `cut` on get new closes, independent highs and lows, new returns and scaled variances."""
    rng = np.random.default_rng(seed)
    out = daily.copy()
    n = len(out) - cut
    close = out["close"].iloc[cut - 1] * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    open_ = close * np.exp(rng.normal(0, 0.01, n))
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.02, n)))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.02, n)))
    for col, val in (("open", open_), ("high", hi), ("low", lo), ("close", close), ("r_cc", rng.normal(0, 3, n)),
                     ("rv", out["rv"].to_numpy()[cut:] * rng.uniform(2, 9, n))):
        out.iloc[cut:, out.columns.get_loc(col)] = val
    return out


@pytest.fixture(scope="session")
def daily():
    return make_daily()
