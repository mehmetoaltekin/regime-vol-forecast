"""
Tests that pin the evaluation tables of rvf.study.evaluate_run and the regime splits of rvf.evaluate against values
that are derived here without the code under test: a statsmodels HAC variance, scipy distributions, a hand-written
Giacomini-White statistic, Holm procedure, regime masks and effect sizes.

The stored results are small synthetic walk-forward results (three instruments, two of them on one calendar and one
that starts later, horizons 1, 5 and 22), so evaluate_run is called directly and no model is estimated. At every horizon
the losses, the origin of each forecast and the regime probability of each origin differ, so a horizon that is not
passed on shows up in the numbers.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

import rvf.evaluate as E
import rvf.study as S
from rvf.walkforward import WFResult

CFG = S.StudyConfig(prespec_version="t", instruments=[], mcs_reps=10, mcs_block=2)
HORIZONS = (1, 5, 22)
LAGS = {1: 5, 5: 5, 22: 21}                                       # PRESPEC 8.2: Newey-West lag max(h - 1, 5)
CAL = pd.bdate_range("2019-10-01", periods=220, name="day")
LAST = 140                                                         # target positions run from the instrument's start to here
STARTS = {"AAA": 70, "BBB": 70, "CCC": 100}                        # CCC has no forecast on the first 30 days of the others
MODELS = ["m3", "m5", "m6", "m7", "m4", "m2c"]
INSTRUMENT_COLUMNS = ["h", "instrument"]


# ------------------------------------------------------------------------------------------------ synthetic results

def _origin_probability(rng) -> pd.Series:
    """prob_stress per forecast origin: calm except two blocks of origins, with noise inside each level."""
    pos = np.arange(len(CAL))
    stress = ((pos >= 80) & (pos < 95)) | ((pos >= 118) & (pos < 132))
    p = np.where(stress, 0.8, 0.2) + rng.uniform(-0.1, 0.1, len(pos))
    return pd.Series(p, index=CAL.rename("origin"), name="prob_stress")


def make_panel(models, starts=STARTS, horizons=HORIZONS, seed=0, identical=None):
    """
    Stored walk-forward results. The forecast of horizon h for target position i was made at origin position i - h, and
    forecasts and targets are drawn anew for every horizon. truth[(symbol, h)] = (qlike losses by model, prob_stress of
    each target day's origin), both built here with plain numpy. identical = (symbol, a, b): model a repeats model b there.
    """
    rng = np.random.default_rng(seed)
    results, truth = {}, {}
    for sym, start in starts.items():
        pos = np.arange(start, LAST)
        n = len(pos)
        p_origin = _origin_probability(rng)
        frames = []
        for h in horizons:
            base = np.exp(rng.normal(0, 0.3, n))
            target = base * np.exp(rng.normal(0, 0.4, n))
            fcs = {m: base * np.exp(0.03 * i + rng.normal(0, 0.15, n)) for i, m in enumerate(models)}
            if identical and identical[0] == sym:
                fcs[identical[1]] = fcs[identical[2]].copy()
            ratio = {m: target / f for m, f in fcs.items()}
            truth[(sym, h)] = (pd.DataFrame({m: r - np.log(r) - 1 for m, r in ratio.items()}, index=CAL[pos]),
                               p_origin.iloc[pos - h].to_numpy())
            for m, f in fcs.items():
                frames.append(pd.DataFrame({"target_day": CAL[pos], "model": m, "h": h, "origin": CAL[pos - h], "forecast": f,
                                            "raw": f, "target": target, "source": m}))
        fc = pd.concat(frames).set_index(["target_day", "model", "h"]).sort_index()
        regime = pd.DataFrame({"prob_stress": p_origin, "k": 2, "refit": np.arange(len(CAL)) % 20 == 0, "hmm_converged": True,
                               "hmm_warnings": 0, "hmm_failed": False})
        counters = pd.DataFrame([dict(model=m, h=h, refits=3, failures=0) for m in models for h in horizons])
        results[sym] = WFResult(fc, regime, counters)
    return results, truth


def stub_mcs(losses, **kw):
    """The model confidence set is not under test here; it only has to return a frame of the right shape."""
    return pd.DataFrame({"mcs_pvalue": 1.0, "included": True, "status": "ok"}, index=losses.columns)


@pytest.fixture(scope="module")
def panel():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(E, "model_confidence_set", stub_mcs)
        results, truth = make_panel(MODELS)
        tables = S.evaluate_run(results, CFG)
    return SimpleNamespace(results=results, truth=truth, tables=tables)


# ------------------------------------------------------------------------------------------------ references

def ref_se2(d, lag):
    """Variance of the mean of d with a Bartlett Newey-West variance (statsmodels)."""
    return sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False}).bse[0] ** 2


def hln_factor(n, h):
    return np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)


def ref_dm(d, h, alt, hln=True):
    """(stat, pvalue) of the Diebold-Mariano test of PRESPEC 8.3 (hln) or 8.2 (no hln, normal critical values)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    stat = d.mean() / np.sqrt(ref_se2(d, max(h - 1, 5)))
    dist = stats.norm
    if hln:
        stat, dist = stat * hln_factor(n, h), stats.t(n - 1)
    p = {"less": dist.cdf(stat), "greater": dist.sf(stat), "two-sided": 2 * dist.sf(abs(stat))}[alt]
    return stat, p


def ref_effect(l_ref, l_mod, h, hln):
    """Effect size of PRESPEC 8.6 with d = reference - model."""
    d = np.asarray(l_ref) - np.asarray(l_mod)
    n = len(d)
    se = np.sqrt(ref_se2(d, max(h - 1, 5)))
    if hln:
        se, crit = se / hln_factor(n, h), stats.t(n - 1).ppf(0.975)
    else:
        crit = stats.norm.ppf(0.975)
    ref_mean = np.mean(l_ref)
    return dict(n=n, mean_d=d.mean(), ref_mean_loss=ref_mean, se_d=se, rel_improvement_pct=100 * d.mean() / ref_mean,
                ci_low_pct=100 * (d.mean() - crit * se) / ref_mean, ci_high_pct=100 * (d.mean() + crit * se) / ref_mean)


def ref_holm(p):
    """Holm step-down: sort, multiply by the number of tests not yet rejected, running maximum, cap at one."""
    p = list(p)
    m = len(p)
    out, running = [0.0] * m, 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda j: p[j])):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return np.array(out)


def ref_gw(d, H, h):
    """Giacomini-White statistic n zbar' V^-1 zbar, V with Bartlett weights 1 - j/h for lags 1 .. h - 1, written with loops."""
    Z = np.asarray(H, dtype=float) * np.asarray(d, dtype=float)[:, None]
    n, q = Z.shape
    zbar = Z.mean(axis=0)
    V = sum(np.outer(z, z) for z in Z) / n
    for j in range(1, h):
        G = sum(np.outer(Z[t], Z[t - j]) for t in range(j, n)) / n
        V = V + (1 - j / h) * (G + G.T)
    stat = n * zbar @ np.linalg.solve(V, zbar)
    return stat, stats.chi2(q).sf(stat)


def ref_masks(p, threshold=0.5, halfwidth=10):
    """Stress, calm and transition flags: a crossing is the first forecast on the new side of the threshold."""
    stress = np.asarray(p) > threshold
    crossings = [i for i in range(1, len(stress)) if stress[i] != stress[i - 1]]
    transition = np.array([any(abs(i - c) <= halfwidth for c in crossings) for i in range(len(stress))])
    return stress, ~stress, transition


def pick(df, **kw):
    """The one row of a table that matches all the given column values."""
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    assert m.sum() == 1, (kw, int(m.sum()))
    return df[m].iloc[0]


def diff(panel_or_truth, sym, h, a, b):
    losses = panel_or_truth[(sym, h)][0]
    return (losses[a] - losses[b]).to_numpy()


# ------------------------------------------------------------------------------------------------ per-instrument tests

def test_the_tests_are_one_sided_for_h1_h2_and_the_garch_pair_and_two_sided_for_h3(panel):
    """PRESPEC 2 and 8.3: H1, H2 and M4 vs M2c are one-sided (a better than b), H3 is two-sided; p-values follow the direction."""
    dm = panel.tables["dm_by_instrument"]
    expected = {"H1": ("less", {"m5 vs m3"}), "H2": ("less", {"m6 vs m3"}), "M4_vs_M2c": ("less", {"m4 vs m2c"}),
                "H3": ("two-sided", {"m5 vs m7", "m6 vs m7"})}
    for fam, (alt, comps) in expected.items():
        f = dm[dm["family"] == fam]
        assert set(f["alternative"]) == {alt} and set(f["comparison"]) == comps and len(f) == len(comps) * 3 * len(HORIZONS), fam
        for _, row in f.iterrows():
            a, b = row["comparison"].split(" vs ")
            stat, p = ref_dm(diff(panel.truth, row["instrument"], row["h"], a, b), row["h"], alt)
            assert row["stat"] == pytest.approx(stat, rel=1e-8) and row["pvalue"] == pytest.approx(p, rel=1e-8), (fam, dict(row))
    # the data carry both signs, so a reversed direction or a reversed pair would be visible
    h1 = dm[(dm["family"] == "H1") & (dm["h"] == 1)]
    assert (h1["mean_diff"] != 0).all() and 0.0 < h1["pvalue"].min() and h1["pvalue"].max() < 1.0
    assert not np.allclose(h1["pvalue"], [ref_dm(diff(panel.truth, s, 1, "m5", "m3"), 1, "greater")[1] for s in h1["instrument"]])


def test_the_horizon_reaches_the_per_instrument_tests_and_their_lag(panel):
    """PRESPEC 8.3: the per-instrument DM test uses the horizon's losses, HLN factor and lag max(h - 1, 5) (5, 5 and 21 for h = 1, 5, 22)."""
    dm = panel.tables["dm_by_instrument"]
    assert {h: set(dm[dm["h"] == h]["lag"]) for h in HORIZONS} == {h: {LAGS[h]} for h in HORIZONS}
    for h in HORIZONS:
        for sym in STARTS:
            row = pick(dm, h=h, family="H1", instrument=sym)
            stat, p = ref_dm(diff(panel.truth, sym, h, "m5", "m3"), h, "less")
            n = len(panel.truth[(sym, h)][0])
            assert row["n"] == n and row["mean_diff"] == pytest.approx(diff(panel.truth, sym, h, "m5", "m3").mean(), rel=1e-10)
            assert row["stat"] == pytest.approx(stat, rel=1e-8) and row["pvalue"] == pytest.approx(p, rel=1e-8)
    # the horizon changes the number, so a test that always used h = 1 would fail the comparison above
    s1 = ref_dm(diff(panel.truth, "AAA", 22, "m5", "m3"), 1, "less")[0]
    assert pick(dm, h=22, family="H1", instrument="AAA")["stat"] != pytest.approx(s1, rel=1e-3)


def test_mean_losses_come_from_the_horizons_own_forecasts(panel):
    """PRESPEC 8.1: the loss table of horizon h holds the QLIKE of the horizon-h forecasts on that instrument's own days."""
    ml = panel.tables["mean_loss"]
    for h in HORIZONS:
        for sym in STARTS:
            losses = panel.truth[(sym, h)][0]
            fc = panel.results[sym].forecasts.xs(h, level="h")
            mse = ((fc["forecast"] - fc["target"]) ** 2).groupby(level="model").mean()      # the other loss, from the plain forecasts
            for m in MODELS:
                row = pick(ml, h=h, instrument=sym, model=m)
                assert row["mean_loss"] == pytest.approx(losses[m].mean(), rel=1e-10) and row["n"] == len(losses)
                assert row["mean_other"] == pytest.approx(mse[m], rel=1e-10) and row["loss"] == "qlike"


# ------------------------------------------------------------------------------------------------ pooled tests and effect sizes

def pooled_reference(panel, a, b, h):
    """Daily average over the instruments that have a loss differential that day, from the plain loss differentials."""
    series = {s: pd.Series(diff(panel.truth, s, h, a, b), index=panel.truth[(s, h)][0].index) for s in STARTS}
    daily = pd.DataFrame(series)
    return daily, daily.mean(axis=1).to_numpy()


def test_the_pooled_test_averages_the_available_instruments_at_every_horizon(panel):
    """PRESPEC 8.2: the pooled DM test is Newey-West on the daily cross-instrument average (normal critical values), lag max(h - 1, 5), on an unequal calendar."""
    pooled = panel.tables["dm_pooled"]
    assert set(pooled["lag"]) == {5, 21} and len(pooled) == 5 * len(HORIZONS)
    for h in HORIZONS:
        for fam, comps in {"H1": [("m5", "m3", "less")], "H2": [("m6", "m3", "less")], "M4_vs_M2c": [("m4", "m2c", "less")],
                           "H3": [("m5", "m7", "two-sided"), ("m6", "m7", "two-sided")]}.items():
            for a, b, alt in comps:
                row = pick(pooled, h=h, family=fam, comparison=f"{a} vs {b}")
                daily, avg = pooled_reference(panel, a, b, h)
                stat, p = ref_dm(avg, h, alt, hln=False)
                assert row["alternative"] == alt and row["lag"] == LAGS[h] and row["n"] == 70
                assert row["mean_diff"] == pytest.approx(avg.mean(), rel=1e-10)
                assert row["stat"] == pytest.approx(stat, rel=1e-8) and row["pvalue"] == pytest.approx(p, rel=1e-8)
                assert row["instruments_per_day"] == pytest.approx((30 * 2 + 40 * 3) / 70)
    # the lag is the only place where the horizon enters a pooled test at h = 5, so h = 22 must be at lag 21
    daily, avg = pooled_reference(panel, "m5", "m3", 22)
    assert pick(pooled, h=22, family="H1", comparison="m5 vs m3")["stat"] != pytest.approx(ref_dm(avg, 1, "less", hln=False)[0], rel=1e-6)


def test_effect_sizes_per_instrument_and_pooled_at_every_horizon(panel):
    """PRESPEC 8.6: effect sizes use d = reference - model, relative improvement over the reference mean, HAC interval with the horizon's lag (HLN per instrument, normal pooled)."""
    eff = panel.tables["effect_sizes"]
    assert set(eff["lag"]) == {5, 21}
    pairs = {"H1": ("m5", "m3"), "H2": ("m6", "m3"), "M4_vs_M2c": ("m4", "m2c")}
    pairs_h3 = [("H3", "m5", "m7"), ("H3", "m6", "m7")]
    for h in HORIZONS:
        for fam, a, b in [(f, *ab) for f, ab in pairs.items()] + pairs_h3:
            for sym in STARTS:
                losses = panel.truth[(sym, h)][0]
                ref = ref_effect(losses[b], losses[a], h, hln=True)
                row = pick(eff, h=h, family=fam, model=a, reference=b, scope=sym)
                assert row["lag"] == LAGS[h]
                for k, v in ref.items():
                    assert row[k] == pytest.approx(v, rel=1e-8), (h, fam, a, b, sym, k)
            # pooled: both losses averaged over the instruments that have them that day, normal critical values
            lb = pd.DataFrame({s: panel.truth[(s, h)][0][b] for s in STARTS})
            la = pd.DataFrame({s: panel.truth[(s, h)][0][a] for s in STARTS})
            ref = ref_effect(lb.mean(axis=1).to_numpy(), (lb.mean(axis=1) - (lb - la).mean(axis=1)).to_numpy(), h, hln=False)
            row = pick(eff, h=h, family=fam, model=a, reference=b, scope="pooled")
            assert row["lag"] == LAGS[h] and row["instruments_per_day"] == pytest.approx((30 * 2 + 40 * 3) / 70)
            for k, v in ref.items():
                assert row[k] == pytest.approx(v, rel=1e-8), (h, fam, a, b, "pooled", k)


def test_the_dm_mean_difference_is_the_negative_of_the_effect_size_difference(panel):
    """PRESPEC 8.6: the DM loss differential is L(a) - L(b) and the effect-size differential L(b) - L(a), so the means are negatives, per instrument and pooled."""
    dm, pooled, eff = (panel.tables[k] for k in ("dm_by_instrument", "dm_pooled", "effect_sizes"))
    for _, row in dm.iterrows():
        a, b = row["comparison"].split(" vs ")
        e = pick(eff, h=row["h"], family=row["family"], model=a, reference=b, scope=row["instrument"])
        assert row["mean_diff"] == pytest.approx(-e["mean_d"], rel=1e-10) and row["n"] == e["n"]
    for _, row in pooled.iterrows():
        a, b = row["comparison"].split(" vs ")
        e = pick(eff, h=row["h"], family=row["family"], model=a, reference=b, scope="pooled")
        assert row["mean_diff"] == pytest.approx(-e["mean_d"], rel=1e-10) and row["n"] == e["n"] == 70
        assert row["instruments_per_day"] == pytest.approx(e["instruments_per_day"])
    # on the shared days the later instrument adds to the average, so the unequal calendar is really exercised
    assert pick(pooled, h=1, family="H1", comparison="m5 vs m3")["n"] == 70 > len(panel.truth[("CCC", 1)][0])


# ------------------------------------------------------------------------------------------------ conditional analysis

def test_the_conditional_test_uses_a_constant_and_the_origin_probability_at_every_horizon(panel):
    """PRESPEC 8.5: Giacomini-White with instruments (1, p_t) where p_t is known at the forecast origin; Bartlett HAC with h - 1 lags."""
    gw = panel.tables["gw_conditional"]
    assert len(gw) == 2 * 3 * len(HORIZONS)
    for h in HORIZONS:
        for sym in STARTS:
            p = panel.truth[(sym, h)][1]
            for a in ("m5", "m6"):
                row = pick(gw, h=h, instrument=sym, comparison=f"{a} vs m3")
                stat, pval = ref_gw(diff(panel.truth, sym, h, a, "m3"), np.column_stack([np.ones(len(p)), p]), h)
                assert row["n"] == len(p)
                assert row["stat"] == pytest.approx(stat, rel=1e-8) and row["pvalue"] == pytest.approx(pval, rel=1e-8, abs=1e-12)


def test_gw_test_hac_weights_for_horizons_above_one():
    """PRESPEC 8.5 and Giacomini-White (2006): for h > 1 the weighted second moment adds Bartlett-weighted autocovariances of lags 1 to h - 1."""
    rng = np.random.default_rng(12)
    n = 300
    e = rng.normal(size=n + 1)
    p = 0.5 + 0.25 * (e[1:] + 0.8 * e[:-1])                              # autocorrelated instrument and differential
    d = rng.normal(size=n + 1)
    d = 0.3 * (p - 0.5) + d[1:] + 0.7 * d[:-1]
    for H in (np.column_stack([np.ones(n), p]), np.column_stack([np.ones(n), p, p ** 2])):
        for h in (1, 3, 6):
            out = E.gw_test(d, H, h)
            stat, pval = ref_gw(d, H, h)
            assert out["stat"] == pytest.approx(stat, rel=1e-9) and out["pvalue"] == pytest.approx(pval, rel=1e-8, abs=1e-14)
            assert out["q"] == H.shape[1] and out["n"] == n
    assert ref_gw(d, np.column_stack([np.ones(n), p]), 3)[0] != pytest.approx(ref_gw(d, np.column_stack([np.ones(n), p]), 1)[0], rel=1e-3)


def test_loss_splits_condition_on_the_origin_probability_of_each_horizon(panel):
    """PRESPEC 8.5: stress (p > 0.5), calm and +/-10-forecast transition windows are built from the prob_stress of each forecast's origin at its horizon."""
    sp = panel.tables["loss_splits"]
    for h in HORIZONS:
        for sym in STARTS:
            losses, p = panel.truth[(sym, h)]
            stress, calm, transition = ref_masks(p)
            assert 0 < transition.sum() < len(p) and 0 < stress.sum() < len(p)
            for name, m in (("all", np.ones(len(p), dtype=bool)), ("stress", stress), ("calm", calm), ("transition", transition)):
                row = pick(sp, h=h, instrument=sym, split=name)
                assert row["n"] == m.sum(), (h, sym, name)
                for model in ("m3", "m5", "m6"):
                    assert row[model] == pytest.approx(losses[model].to_numpy()[m].mean(), rel=1e-10), (h, sym, name, model)
    # the horizons differ in which origins they use: the stress shares of h = 1 and h = 22 are not the same
    shares = {h: pick(sp, h=h, instrument="AAA", split="stress")["n"] for h in (1, 22)}
    assert shares[1] != shares[22]


# ------------------------------------------------------------------------------------------------ Holm families

def test_h3_holm_family_holds_both_comparisons_of_all_instruments(panel):
    """PRESPEC 8.3: Holm is applied separately for H1, H2, M4 vs M2c and for H3 with both comparisons together (ten tests for five instruments; here six)."""
    dm = panel.tables["dm_by_instrument"]
    for h in HORIZONS:
        for fam, size in (("H1", 3), ("H2", 3), ("M4_vs_M2c", 3), ("H3", 6)):
            f = dm[(dm["family"] == fam) & (dm["h"] == h)]
            assert len(f) == size
            assert np.allclose(f["pvalue_holm"], ref_holm(f["pvalue"]), rtol=1e-10), (fam, h)
        f = dm[(dm["family"] == "H3") & (dm["h"] == h)]
        per_comparison = np.concatenate([ref_holm(g["pvalue"]) for _, g in f.groupby("comparison", sort=False)])
        assert (f["pvalue_holm"].to_numpy() >= per_comparison - 1e-12).all()
    # the data make the family adjustment bigger than a per-comparison one somewhere, so the grouping is visible
    f = dm[(dm["family"] == "H3") & (dm["h"] == 1)]
    per_comparison = pd.concat([pd.Series(ref_holm(g["pvalue"]), index=g.index) for _, g in f.groupby("comparison")])
    assert (f["pvalue_holm"] > per_comparison.loc[f.index] + 1e-6).any()


def test_holm_treats_a_missing_p_value_as_one_in_the_family():
    """PRESPEC 8.3: an instrument without a statistic (identical forecasts) counts as p = 1 in the Holm family and gets the adjusted p-value 1."""
    results, _ = make_panel(["m5", "m6", "m7"], starts={"AAA": 70, "BBB": 70, "CCC": 70}, horizons=(1,), seed=4,
                            identical=("CCC", "m5", "m7"))
    t = S.evaluate_run(results, CFG, pooled=False, extras=False, effects=False)["dm_by_instrument"]
    h3 = t[t["family"] == "H3"]
    assert len(h3) == 6 and h3["pvalue"].isna().sum() == 1
    gone = h3[h3["pvalue"].isna()].iloc[0]
    assert gone["comparison"] == "m5 vs m7" and gone["instrument"] == "CCC" and gone["mean_diff"] == 0.0
    assert gone["pvalue_holm"] == 1.0
    p = h3["pvalue"].fillna(1.0).to_numpy()
    assert np.allclose(h3["pvalue_holm"], ref_holm(p), rtol=1e-10)
    assert not np.allclose(ref_holm(p), ref_holm(np.nan_to_num(h3["pvalue"].to_numpy(), nan=0.0)))      # a zero would change the others


# ------------------------------------------------------------------------------------------------ directions of the robustness families

@pytest.mark.parametrize("variant, models, expected", [
    ("log_har", ["m3_log", "m5_log", "m6_log", "m7_log"],
     {"H1": ("less", {"m5_log vs m3_log"}), "H2": ("less", {"m6_log vs m3_log"}),
      "H3": ("two-sided", {"m5_log vs m7_log", "m6_log vs m7_log"})}),
    ("harq", ["m3_q", "m5_q"], {"HARQ": ("less", {"m5_q vs m3_q"})}),
])
def test_the_log_and_harq_families_keep_the_direction_of_the_levels_families(variant, models, expected):
    """PRESPEC 10 checks 4 and 10: the log-HAR and HARQ comparisons are one-sided for H1 and H2 (HARQ-X better) and two-sided for H3, and are tested that way."""
    results, truth = make_panel(models, horizons=(1, 5), seed=9)
    families = S.variant_setup(variant, CFG)["families"]
    t = S.evaluate_run(results, CFG, families=families, extras=False, effects=False)
    for table, kind in ((t["dm_by_instrument"], "instrument"), (t["dm_pooled"], "pooled")):
        assert set(table["family"]) == set(expected)
        for fam, (alt, comps) in expected.items():
            f = table[table["family"] == fam]
            assert set(f["alternative"]) == {alt} and set(f["comparison"]) == comps
            for _, row in f.iterrows():
                a, b = row["comparison"].split(" vs ")
                if kind == "instrument":
                    stat, p = ref_dm(diff(truth, row["instrument"], row["h"], a, b), row["h"], alt)
                else:
                    stat, p = ref_dm(pooled_reference(SimpleNamespace(truth=truth), a, b, row["h"])[1], row["h"], alt, hln=False)
                assert row["stat"] == pytest.approx(stat, rel=1e-8) and row["pvalue"] == pytest.approx(p, rel=1e-8), (fam, dict(row))


# ------------------------------------------------------------------------------------------------ regime masks and defaults

def _split_p():
    """120 forecasts: 0.5 exactly (calm) in 5..9, a stress run 40..79 that starts with 0.5000001, calm again from 80."""
    idx = pd.bdate_range("2023-01-02", periods=120, name="target_day")
    p = pd.Series(0.1, index=idx)
    p.iloc[5:10] = 0.5
    p.iloc[40] = 0.5000001
    p.iloc[41:80] = 0.9
    return p


def test_the_stress_boundary_is_strict_and_the_default_window_is_ten_forecasts_around_each_crossing():
    """PRESPEC 8.5: stress is p > 0.5 (0.5 is calm), transitions are +/-10 trading days around each crossing, and these are the defaults."""
    p = _split_p()
    masks = E.regime_masks(p)                                           # no keyword arguments: the defaults are the specification
    assert list(np.flatnonzero(masks["stress"])) == list(range(40, 80))
    assert masks["calm"].sum() == 80 and not masks["stress"].iloc[5:10].any() and masks["calm"].iloc[5:10].all()
    assert list(np.flatnonzero(masks["transition"])) == [*range(30, 51), *range(70, 91)]         # 21 forecasts per isolated crossing
    ref = ref_masks(p)
    for got, want in zip((masks["stress"], masks["calm"], masks["transition"]), ref):
        assert (got.to_numpy() == want).all()


def test_loss_splits_defaults_are_the_specified_threshold_and_window():
    """PRESPEC 8.5: loss_splits called without keywords uses the 0.5 threshold (strict) and the +/-10 window."""
    p = _split_p()
    losses = pd.DataFrame({"a": np.arange(120.0), "b": 2.0}, index=p.index)
    out = E.loss_splits(losses, p)
    assert out.loc["all", "n"] == 120 and out.loc["stress", "n"] == 40 and out.loc["calm", "n"] == 80
    assert out.loc["stress", "a"] == pytest.approx(np.arange(40, 80).mean()) and out.loc["calm", "a"] == pytest.approx(
        np.r_[np.arange(0, 40), np.arange(80, 120)].mean())
    assert out.loc["transition", "n"] == 42 and out.loc["transition", "a"] == pytest.approx(np.r_[np.arange(30, 51), np.arange(70, 91)].mean())
    assert (out["b"] == 2.0).all()


def test_a_crossing_in_the_first_forecasts_keeps_a_window_from_the_start():
    """PRESPEC 8.5: a crossing at forecast 3 still flags the forecasts 0 to 13; the window does not wrap around or vanish at the start of the sample."""
    idx = pd.bdate_range("2023-01-02", periods=40, name="target_day")
    p = pd.Series(0.1, index=idx)
    p.iloc[3:] = 0.9
    masks = E.regime_masks(p)
    assert list(np.flatnonzero(masks["transition"])) == list(range(0, 14))
    assert masks["transition"].sum() == 14 and masks["stress"].sum() == 37
    out = E.loss_splits(pd.DataFrame({"a": np.arange(40.0)}, index=idx), p)
    assert out.loc["transition", "n"] == 14 and out.loc["transition", "a"] == pytest.approx(6.5)
    q = pd.Series(0.9, index=idx)                                       # a crossing in the last forecast gets a window too
    q.iloc[-1] = 0.1
    assert list(np.flatnonzero(E.regime_masks(q)["transition"])) == list(range(29, 40))


def test_the_regime_summary_counts_stress_origins_with_the_strict_half_threshold():
    """PRESPEC 8.5 and 6: share_stress is the share of origins with prob_stress > 0.5, so exactly 0.5 is not stress and 0.5000001 is."""
    def result(values):
        regime = pd.DataFrame({"prob_stress": values, "k": 2, "refit": [True] + [False] * (len(values) - 1),
                               "hmm_converged": True, "hmm_warnings": 0, "hmm_failed": False})
        return WFResult(pd.DataFrame(), regime, pd.DataFrame())
    out = S.hmm_summary({"X": result([0.2, 0.5, 0.5, 0.5000001, 0.55, 0.9, 0.6, 0.65]), "Y": result([0.5, 0.5, 0.5, 0.5])}).set_index("instrument")
    assert out.loc["X", "share_stress"] == pytest.approx(5 / 8) and out.loc["Y", "share_stress"] == 0.0
