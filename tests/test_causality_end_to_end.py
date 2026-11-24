"""
Causality from raw M1 bars through the daily bundle to the walk-forward forecasts (PRESPEC 4, 6, 7, 16).

Every other no-look-ahead test starts from daily frames or stub models. This one starts from synthetic minute bars,
cuts them at a midweek day, perturbs everything after the cut in several ways, and requires every output dated up to
the cut (daily bundle, forecasts, regime series) to be bit-identical to a run on data that simply ends at the cut.
"""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import rvf.models as M
import rvf.study as S
from rvf.walkforward import run_walk_forward

RULES = {"interval_min": 5, "include_session_break": True, "excluded_windows": [[1380, 60]], "merge_sunday": True,
         "short_day_share": 0.5, "day_length": "active", "median_day_length": 264, "ohlc_from_included": True,
         "stale_share": 0.5, "anchor_gap_days": 5}
CLEAN = {"bounce_sd": 20, "bounce_reversal": 0.75, "bounce_lookback_days": 20}
CUT_DAY = 267                                         # position among the weekdays; day 267 is a Wednesday
# window 73 with refit 21 and burn-in 22 makes the cut day a refit origin; the cut one day earlier is not one
DAY = pd.Timedelta(days=1)
# a variant per kind of input: the primary run, a target that keeps the rollover window, another HMM feature
VARIANTS = ("primary", "rv_no_rollover", "hmm_rolling")


def spec() -> S.InstrumentSpec:
    return S.InstrumentSpec("SYN", None, None, RULES, "FX", CLEAN)


def study_config() -> S.StudyConfig:
    return S.StudyConfig(prespec_version="test", instruments=[spec()], window=73, refit_every=21, burn_in=22,
                         alt_windows=(60, 100), horizons=(1, 5), min_ess=20, ms_starts=1)


def make_raw(n_days=330, seed=5, start="2020-01-06"):
    """
    Weekday M1 bars 00:00-23:59 in two volatility regimes, Sunday-evening bars before each Monday, noise in the rollover
    window, 5 % of the bars missing, three short days and one frozen day. Returns the bars and the weekdays.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    state = np.zeros(n_days, int)
    for t in range(1, n_days):
        state[t] = state[t - 1] if rng.random() < (0.97 if state[t - 1] == 0 else 0.92) else 1 - state[t - 1]
    sig = np.where(state == 0, 0.6, 2.0) * rng.uniform(0.8, 1.2, n_days)
    r = rng.normal(0, 1, (n_days, 1440)) * (sig[:, None] / np.sqrt(1440))
    c = 100 * np.exp(np.cumsum(r.ravel()) / 100)
    ts = (days.values[:, None] + (np.arange(1440) * np.timedelta64(1, "m"))[None, :]).ravel()
    minute = np.tile(np.arange(1440), n_days)
    roll = (minute >= 1380) | (minute < 60)
    c[roll] *= np.exp(rng.normal(0, 0.0005, roll.sum()))
    o = np.concatenate([[c[0]], c[:-1]])
    wick = np.abs(rng.normal(0, 0.00005, len(c)))
    df = pd.DataFrame({"ts": ts, "open": o, "high": np.maximum(o, c) * np.exp(wick), "low": np.minimum(o, c) * np.exp(-wick),
                       "close": c, "tickvol": 1, "vol": 0, "spread": 10})
    sun = []
    for monday in [d for d in days if d.day_of_week == 0][1:]:
        tt = monday - DAY + pd.Timedelta(hours=21) + pd.to_timedelta(np.arange(180), unit="m")
        cc = df.loc[df["ts"] < monday, "close"].iloc[-1] * np.exp(np.cumsum(rng.normal(0, 0.00005, 180)))
        sun.append(pd.DataFrame({"ts": tt, "open": cc, "high": cc * 1.00002, "low": cc * 0.99998, "close": cc,
                                 "tickvol": 1, "vol": 0, "spread": 10}))
    df = pd.concat([df] + sun, ignore_index=True)
    df = df[rng.random(len(df)) > 0.05]
    for k in (60, 120, 200):                                          # short days: only 01:00-05:00 is traded
        d = days[k]
        mins = (df["ts"] - d) // pd.Timedelta(minutes=1)
        df = df[~((df["ts"] >= d) & (df["ts"] < d + DAY) & ~mins.between(60, 300))]
    d = days[90]                                                       # a frozen day: one price all day
    m = (df["ts"] >= d) & (df["ts"] < d + DAY)
    v = df.loc[m, "close"].iloc[0]
    for col in ("open", "high", "low", "close"):
        df.loc[m, col] = v
    return df.sort_values("ts").reset_index(drop=True), days


def after(df, cut):
    return df["ts"] >= cut + DAY


def scramble(df, cut, seed=1):
    """New closes (a random walk of factors), independent new highs and lows, then half of the bars removed."""
    rng = np.random.default_rng(seed)
    out, m = df.copy(), after(df, cut)
    n = int(m.sum())
    fac = np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    for c in ("open", "high", "low", "close"):
        out.loc[m, c] = out.loc[m, c].to_numpy() * fac
    out.loc[m, "high"] = out.loc[m, ["open", "close"]].max(axis=1) * np.exp(np.abs(rng.normal(0, 0.0003, n)))
    out.loc[m, "low"] = out.loc[m, ["open", "close"]].min(axis=1) * np.exp(-np.abs(rng.normal(0, 0.0003, n)))
    out.loc[m, "high"] *= 1.03
    out.loc[m, "low"] *= 0.97
    return out[~(after(out, cut) & (rng.random(len(out)) < 0.5))].reset_index(drop=True)


def events(df, cut):
    """A frozen day, an eight-day hole (an anchor day), a one-minute spike and a spike straddling midnight."""
    out = df.copy()
    d1 = cut + 6 * DAY
    m = (out["ts"] >= d1) & (out["ts"] < d1 + DAY)
    v = out.loc[m, "close"].iloc[0]
    for c in ("open", "high", "low", "close"):
        out.loc[m, c] = v
    d2 = cut + 15 * DAY
    out = out[~((out["ts"] >= d2) & (out["ts"] < d2 + 8 * DAY))]
    spike = out.index[out["ts"] == cut + 30 * DAY + pd.Timedelta(hours=10)]
    assert len(spike) == 1
    out.loc[spike, ["close", "high"]] *= 1.05
    out = out.reset_index(drop=True)
    ts = out["ts"]
    last = (ts.dt.hour == 23) & (ts.dt.minute == 59) & (ts > cut + 20 * DAY) & (ts.shift(-1) == ts + pd.Timedelta(minutes=1))
    night = out.index[last][:1]                                         # 23:59 jumps +3 %, the 00:00 bar is back
    assert len(night) == 1
    out.loc[night, ["close", "high"]] *= 1.03
    return out


def run_all(raw):
    """Bundle and, per variant, forecasts and regime series of one raw data set."""
    cfg = study_config()
    bundle = S.build_bundle(spec(), raw=raw)
    out = {"bundle": bundle}
    for v in VARIANTS:
        setup = S.variant_setup(v, cfg)
        models = setup["factory"]() if v == "primary" else [M.PureHmm(), M.Ewma(), M.Har(), M.Har(with_regime=True),
                                                            M.RegimeWeightedHar(20)]      # all of M0-M7 only once: time
        with S.single_thread():
            res = run_walk_forward(S.variant_daily(bundle, v), models, setup["wf"], replace(setup["regime"], n_init=2))
        out[v] = (res.forecasts, res.regime)
    return out


@pytest.fixture(scope="module")
def runs():
    S.limit_threads()
    raw, days = make_raw()
    cut = days[CUT_DAY]
    assert cut.day_name() == "Wednesday"
    cases = {"orig": raw, "trunc": raw[raw["ts"] < cut + DAY].reset_index(drop=True),
             "scrambled": scramble(raw, cut), "events": events(raw, cut),
             "scrambled_early": scramble(raw, days[CUT_DAY - 1], seed=2)}       # cut one day earlier: not a refit origin
    out = {k: run_all(v) for k, v in cases.items()}
    out["cut"] = cut
    return out


@pytest.mark.parametrize("case", ["orig", "scrambled", "events"])
def test_nothing_dated_up_to_the_cut_depends_on_later_minute_bars(runs, case):
    """PRESPEC 4, 6, 7: bundle rows, forecasts and regime series up to a cut do not change with the bars after it."""
    cut, trunc, full = runs["cut"], runs["trunc"], runs[case]
    assert trunc["bundle"].index[-1] == cut and len(trunc["bundle"]) > 250
    pd.testing.assert_frame_equal(full["bundle"].loc[:cut], trunc["bundle"], check_exact=True)
    for v in VARIANTS:
        fa, ra = trunc[v]
        fb, rb = full[v]
        assert len(fa) > 400 and fa.index.get_level_values("target_day").max() == cut
        pd.testing.assert_frame_equal(fb.loc[fa.index, ["forecast", "raw", "origin", "source"]],
                                      fa[["forecast", "raw", "origin", "source"]], check_exact=True)
        pd.testing.assert_frame_equal(rb.loc[ra.index], ra, check_exact=True)


@pytest.mark.parametrize("case", ["scrambled", "events", "scrambled_early"])
def test_the_forecasts_made_on_the_cut_day_itself_do_not_see_the_next_day(runs, case):
    """PRESPEC 4, 6: the forecasts of origin = cut (they need no later bar) are the same whatever the bars after the cut are."""
    orig, pert = runs["orig"], runs[case]
    cut = runs["cut"] - (DAY if case == "scrambled_early" else pd.Timedelta(0))     # the last day before the perturbation
    for v in VARIANTS:
        fa, fb = orig[v][0], pert[v][0]
        keep = fa.index[fa["origin"] <= cut]
        assert (fa.loc[keep, "origin"] == cut).sum() >= 2 and len(keep) > 400          # includes the origin on the cut day
        assert orig[v][1].loc[cut, "refit"] == (case != "scrambled_early")             # the cut is a refit origin (window 73) or not
        pd.testing.assert_frame_equal(fb.loc[keep, ["forecast", "raw", "origin", "source"]],
                                      fa.loc[keep, ["forecast", "raw", "origin", "source"]], check_exact=True)
        pd.testing.assert_frame_equal(pert[v][1].loc[:cut], orig[v][1].loc[:cut], check_exact=True)
    assert pert["bundle"].loc[:cut].equals(orig["bundle"].loc[:cut])


@pytest.mark.parametrize("case", ["scrambled", "events"])
def test_the_perturbation_reaches_the_later_bundle_and_forecasts(runs, case):
    """Guard for the test above: the perturbed bars really change the later days, so identical earlier rows mean something."""
    cut, orig, pert = runs["cut"], runs["orig"], runs[case]
    later = orig["bundle"].index[orig["bundle"].index > cut][:40]
    assert len(later) == 40
    for col in ("rv", "rv_no_rollover", "parkinson", "r_cc"):
        a = orig["bundle"].reindex(later)[col]
        b = pert["bundle"].reindex(later)[col]
        assert not np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True), col
    for v in VARIANTS:
        fa, fb = orig[v][0], pert[v][0]
        late = fa.index[fa["origin"] > cut + 25 * DAY]
        assert len(late) > 50
        assert not np.allclose(fa.loc[late, "forecast"].to_numpy(), fb["forecast"].reindex(late).to_numpy(), equal_nan=True), v

