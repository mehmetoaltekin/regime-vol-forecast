"""Unit tests for the evaluation module on examples with known answers."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from rvf import evaluate as E


def test_loss_functions():
    assert E.qlike(1.3, 1.3) == pytest.approx(0.0)
    assert E.qlike(2.0, 1.0) == pytest.approx(2 - np.log(2) - 1)
    assert E.qlike(1.0, 2.0) == pytest.approx(0.5 - np.log(0.5) - 1)
    assert E.qlike(1.0, 2.0) < E.qlike(2.0, 1.0)                      # under-prediction costs more
    assert np.allclose(E.mse([1.0, 3.0], [2.0, 1.0]), [1.0, 4.0])
    assert np.allclose(E.qlike([4.0, 9.0], [2.0, 3.0]), E.qlike([2.0, 3.0], [1.0, 1.0]))    # scale-free


def test_long_run_variance_by_hand_and_against_statsmodels():
    d = np.array([1.0, 2.0, 3.0, 4.0])
    assert E.long_run_variance(d, 0) == pytest.approx(1.25)
    assert E.long_run_variance(d, 1) == pytest.approx(1.25 + 2 * 0.5 * 0.3125)
    x = np.random.default_rng(0).normal(size=400).cumsum() * 0.05 + np.random.default_rng(1).normal(size=400)
    for lag in (0, 5, 21):
        ref = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False}).bse[0] ** 2
        assert E.long_run_variance(x, lag) / len(x) == pytest.approx(ref)


def test_dm_statistic_on_a_known_example():
    d = [1.0, 2.0, 3.0, 4.0]
    plain = E.dm_test(d, h=1, lag=0, hln=False)
    assert plain["stat"] == pytest.approx(2.5 / np.sqrt(1.25 / 4))
    assert plain["pvalue"] == pytest.approx(2 * stats.norm.sf(plain["stat"]))
    corrected = E.dm_test(d, h=1, lag=0, hln=True)
    assert corrected["stat"] == pytest.approx(plain["stat"] * np.sqrt(3 / 4))
    assert corrected["pvalue"] == pytest.approx(2 * stats.t(3).sf(corrected["stat"]))
    h3 = E.dm_test(d, h=3, lag=0, hln=True)
    assert h3["stat"] == pytest.approx(plain["stat"] * np.sqrt((4 + 1 - 6 + 3 * 2 / 4) / 4))
    assert E.dm_test(d, h=1)["lag"] == 5 and E.dm_test(d, h=22)["lag"] == 21
    assert E.dm_test(d, lag=0, alternative="greater")["pvalue"] < 0.05 < E.dm_test(d, lag=0, alternative="less")["pvalue"]
    assert np.isnan(E.dm_test([1.0, 1.0, 1.0])["stat"])                # zero variance: no statistic
    assert E.dm_test([1.0, np.nan, 2.0, 4.0, 3.0], lag=0)["n"] == 4


def test_dm_has_the_right_size_and_finds_a_real_difference():
    rng = np.random.default_rng(5)
    rej = np.mean([E.dm_test(rng.normal(size=300))["pvalue"] < 0.05 for _ in range(1500)])
    assert 0.03 < rej < 0.08
    better = rng.normal(-0.3, 1.0, 400)                                # model A has the lower loss
    assert E.dm_test(better, alternative="less")["pvalue"] < 0.001
    assert E.dm_test(better, alternative="greater")["pvalue"] > 0.99


def test_pooled_dm_averages_available_instruments():
    rng = np.random.default_rng(6)
    idx = pd.bdate_range("2020-01-01", periods=300)
    d = pd.DataFrame(rng.normal(-0.1, 1.0, (300, 3)), index=idx, columns=["a", "b", "c"])
    d.iloc[:100, 2] = np.nan                                           # instrument c starts later
    out = E.pooled_dm(d, alternative="less")
    avg = np.r_[d.iloc[:100, :2].mean(axis=1), d.iloc[100:].mean(axis=1)]
    ref = E.dm_test(avg, hln=False, alternative="less")
    assert out["stat"] == pytest.approx(ref["stat"]) and out["pvalue"] == pytest.approx(ref["pvalue"])
    assert out["n"] == 300 and out["instruments_per_day"] == pytest.approx((100 * 2 + 200 * 3) / 300)


def test_holm_adjustment():
    assert np.allclose(E.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(E.holm([0.5, 0.9]), [1.0, 1.0])
    assert np.allclose(E.holm([0.02]), [0.02])
    with pytest.raises(ValueError):
        E.holm([0.02, np.nan])


def test_model_confidence_set_drops_a_clearly_worse_model():
    rng = np.random.default_rng(7)
    base = rng.chisquare(3, 600)
    losses = pd.DataFrame({"good": base + rng.normal(0, 0.1, 600), "same": base + rng.normal(0, 0.1, 600),
                           "bad": base + 1.0 + rng.normal(0, 0.1, 600)})
    out = E.model_confidence_set(losses, reps=500, seed=1)
    assert out.loc["good", "included"] and out.loc["same", "included"] and not out.loc["bad", "included"]
    assert out.loc["bad", "mcs_pvalue"] < 0.10 and list(out.index) == ["good", "same", "bad"]


def test_gw_test_formula_and_power():
    rng = np.random.default_rng(8)
    n = 500
    p = rng.uniform(0, 1, n)
    H = np.column_stack([np.ones(n), p])
    d0 = rng.normal(0, 1, n)                                           # unrelated to the instruments
    Z = H * d0[:, None]
    zbar = Z.mean(axis=0)
    ref = n * zbar @ np.linalg.inv(Z.T @ Z / n) @ zbar
    out = E.gw_test(d0, H)
    assert out["stat"] == pytest.approx(ref) and out["q"] == 2 and out["pvalue"] == pytest.approx(stats.chi2(2).sf(ref))
    d1 = rng.normal(0, 1, n) + 1.5 * (p - 0.5)                         # zero on average, predictable from p
    assert abs(d1.mean()) < 0.15 and E.gw_test(d1, H)["pvalue"] < 0.001 < E.dm_test(d1)["pvalue"]
    rej = np.mean([E.gw_test(rng.normal(size=300), np.column_stack([np.ones(300), rng.uniform(size=300)]))["pvalue"] < 0.05
                   for _ in range(1000)])
    assert 0.03 < rej < 0.08
    assert E.gw_test(d0, H, h=5)["stat"] != pytest.approx(ref)


def test_regime_masks_and_loss_splits():
    idx = pd.bdate_range("2023-12-01", periods=60, name="target_day")
    p = pd.Series(0.1, index=idx)
    p.iloc[30:40] = 0.9                                                # stress from forecast 30 to 39
    masks = E.regime_masks(p, halfwidth=3)
    assert masks["stress"].sum() == 10 and masks["calm"].sum() == 50
    assert list(np.flatnonzero(masks["transition"])) == [*range(27, 34), *range(37, 44)]    # 30 +/- 3 and 40 +/- 3
    losses = pd.DataFrame({"a": np.arange(60.0), "b": 1.0}, index=idx)
    out = E.loss_splits(losses, p, halfwidth=3)
    assert out.loc["all", "n"] == 60 and out.loc["stress", "a"] == pytest.approx(34.5) and out.loc["calm", "n"] == 50
    assert out.loc["transition", "n"] == 14 and out.loc["transition", "a"] == pytest.approx(35.0)
    assert out.loc["2023", "n"] == (idx.year == 2023).sum() and out.loc["2024", "a"] == pytest.approx(losses["a"][idx.year == 2024].mean())
    assert (out["b"] == 1.0).all()
    with pytest.raises(ValueError):                                    # indexed by origin, not by target day
        E.loss_splits(losses, p.rename_axis("origin"))
    with pytest.raises(ValueError):
        E.loss_splits(losses, p.iloc[:-1])
    with pytest.raises(ValueError):
        E.regime_masks(p.where(p.index != idx[5]))


def test_conditioning_variables_are_taken_at_the_forecast_origin():
    days = pd.bdate_range("2024-01-01", periods=6)
    rows = [(days[i + 1], m, 1, days[i], 1.0, 1.0 if i < 4 else np.nan) for i in range(5) for m in ("m3", "m5")]
    fc = pd.DataFrame(rows, columns=["target_day", "model", "h", "origin", "forecast", "target"]).set_index(["target_day", "model", "h"])
    p_origin = pd.Series([0.1, 0.9, 0.2, 0.8, 0.3, 0.7], index=days.rename("origin"), name="prob_stress")
    p = E.at_origin(fc, p_origin)
    assert p.index.name == "target_day" and list(p.index) == list(days[1:5])       # the forecast without a target is left out
    assert list(p) == [0.1, 0.9, 0.2, 0.8]                                         # target day d gets p of the day before
    losses = E.loss_table(fc)
    out = E.loss_splits(losses, p, halfwidth=0)
    assert out.loc["stress", "n"] == 2 and out.loc["calm", "n"] == 2
    with pytest.raises(ValueError):
        E.at_origin(fc, p_origin.iloc[1:])                                         # an origin without a value


def test_loss_table_from_walk_forward_output():
    days = pd.bdate_range("2024-01-01", periods=3, name="target_day")
    rows = [(d, m, 1, 2.0 if m == "m3" else 1.0, t) for d, t in zip(days, [1.0, 2.0, np.nan]) for m in ("m3", "m5")]
    fc = pd.DataFrame(rows, columns=["target_day", "model", "h", "forecast", "target"]).set_index(["target_day", "model", "h"])
    tab = E.loss_table(fc, 1, "qlike")
    assert list(tab.index) == list(days[:2]) and list(tab.columns) == ["m3", "m5"]      # the day without a target is dropped
    assert tab.loc[days[1], "m3"] == pytest.approx(0.0) and tab.loc[days[1], "m5"] == pytest.approx(2 - np.log(2) - 1)
    assert E.loss_table(fc, 1, "mse").loc[days[0], "m3"] == pytest.approx(1.0)
    bad = fc.copy()
    bad.loc[(days[0], "m5", 1), "forecast"] = -1.0                     # a non-positive forecast has no QLIKE
    with pytest.raises(ValueError, match="non-finite"):
        E.loss_table(bad, 1, "qlike")
