"""Tests for the study pipeline, the run guards and the output writers."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import rvf.study as S
from rvf.walkforward import WFConfig

from conftest import make_daily
from test_realized import brownian

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_study as R  # noqa: E402


def small_cfg(**kw) -> S.StudyConfig:
    panel = [S.InstrumentSpec(s, asset_class=c) for s, c in (("AAA", "FX"), ("BBB", "FX"), ("CCC", "Metals"), ("DDD", "Metals"))]
    base = dict(prespec_version="t", instruments=panel, window=120,
                refit_every=40, horizons=(1, 5), min_ess=20, ms_starts=1, mcs_reps=100, mcs_block=5)
    base.update(kw)
    return S.StudyConfig(**base)


def synth(seed, n=330):
    return make_daily(n, seed=seed)


def write_m1(path: Path, m1: pd.DataFrame) -> None:
    ts = m1["ts"]
    out = pd.DataFrame({"<DATE>": ts.dt.strftime("%Y.%m.%d"), "<TIME>": ts.dt.strftime("%H:%M:%S"), "<OPEN>": m1["open"],
                        "<HIGH>": m1["high"], "<LOW>": m1["low"], "<CLOSE>": m1["close"], "<TICKVOL>": 1, "<VOL>": 0,
                        "<SPREAD>": 10})
    out.to_csv(path, sep="\t", index=False, float_format="%.6f")


@pytest.fixture(scope="module")
def primary():
    cfg = small_cfg()
    daily = {"AAA": synth(1), "BBB": synth(2)}
    res = S.run_variant("primary", cfg, daily, progress=lambda m: None)
    return cfg, res


def test_evaluation_tables(primary):
    cfg, res = primary
    t = S.evaluate_run(res, cfg)
    assert set(t) == {"mean_loss", "counters", "regime_summary", "dm_by_instrument", "dm_pooled", "mcs", "gw_conditional", "loss_splits",
                      "effect_sizes"}
    dm = t["dm_by_instrument"]
    assert set(dm["family"]) == {"H1", "H2", "M4_vs_M2c", "H3"} and set(dm["h"]) == {1, 5}
    assert (dm["pvalue_holm"] >= dm["pvalue"] - 1e-12).all() and (dm["pvalue_holm"] <= 1).all()
    h3 = dm[(dm["family"] == "H3") & (dm["h"] == 1)]
    assert len(h3) == 4 and set(h3["alternative"]) == {"two-sided"}                  # two comparisons, two instruments, one Holm family
    h1 = dm[(dm["family"] == "H1") & (dm["h"] == 1)]
    assert len(h1) == 2 and np.allclose(h1["pvalue_holm"], E_holm(h1["pvalue"]))
    # the per-instrument mean difference is the difference of the mean losses on the common sample
    ml = t["mean_loss"].query("h == 1 and loss == 'qlike'").set_index(["instrument", "model"])["mean_loss"]
    row = h1[h1["instrument"] == "AAA"].iloc[0]
    assert row["mean_diff"] == pytest.approx(ml[("AAA", "m5")] - ml[("AAA", "m3")])
    pooled = t["dm_pooled"]
    assert set(pooled["family"]) == {"H1", "H2", "M4_vs_M2c", "H3"} and (pooled["instruments_per_day"] == 2).all()
    assert (t["mcs"].query("model == 'm0'").empty) and t["mcs"].groupby(["h", "instrument"])["included"].sum().min() >= 1
    assert set(t["gw_conditional"]["comparison"]) == {"m5 vs m3", "m6 vs m3"}
    sp = t["loss_splits"]
    assert {"all", "stress", "calm", "transition"} <= set(sp["split"]) and (sp.query("split == 'all'")["n"] > 100).all()
    assert t["counters"].groupby(["instrument", "model", "h"]).size().max() == 1
    assert set(t["regime_summary"]["instrument"]) == {"AAA", "BBB"} and (t["regime_summary"]["hmm_not_converged"] == 0).all()
    mse = S.evaluate_run(res, cfg, loss="mse", pooled=False, extras=False)
    assert set(mse) == {"mean_loss", "counters", "regime_summary", "dm_by_instrument"} and (mse["mean_loss"]["loss"] == "mse").all()
    one = S.evaluate_run({"AAA": res["AAA"]}, cfg)                                     # a single instrument has no pooled test
    assert "dm_pooled" not in one


def E_holm(p):
    from rvf.evaluate import holm
    return holm(p.to_numpy())


def test_figures_tables_and_latex(primary, tmp_path):
    cfg, res = primary
    tables = S.evaluate_run(res, cfg)
    paths = S.write_tables(tables, tmp_path / "t", "primary")
    assert {p.suffix for p in paths} == {".csv", ".tex"} and all(p.stat().st_size > 0 for p in paths)
    back = pd.read_csv(tmp_path / "t" / "primary__dm_pooled.csv")
    assert len(back) == len(tables["dm_pooled"])
    tex = (tmp_path / "t" / "primary__mean_loss.tex").read_text()
    assert tex.startswith("\\begin{tabular}") and "\\toprule" in tex and "mean\\_loss" in tex and tex.rstrip().endswith("\\end{tabular}")
    figs = S.make_figures(res, tmp_path / "f", "primary")
    assert [p.suffix for p in figs] == [".pdf", ".pdf"] and all(p.read_bytes()[:4] == b"%PDF" for p in figs)
    fc = S.save_forecasts(res, tmp_path / "fc", "primary")
    again = pd.read_parquet(fc[0])
    assert len(again) == len(res["AAA"].forecasts) and {"target_day", "model", "h", "forecast"} <= set(again.columns)
    assert S.latex_escape("a_b & 100% #1") == r"a\_b \& 100\% \#1"


def test_latex_escape_writes_out_every_special_character():
    """The descriptive-statistics table has a variable called 'RV (%^2)': a bare ^ in text mode is a LaTeX error."""
    import re
    s = "a\\b & 50% _x_ #1 $5 ^2 ~ {c} RV (%^2)"
    out = S.latex_escape(s)
    assert out == r"a\textbackslash{}b \& 50\% \_x\_ \#1 \$5 \textasciicircum{}2 \textasciitilde{} \{c\} RV (\%\textasciicircum{}2)"
    written_out = re.sub(r"\\(textbackslash|textasciicircum|textasciitilde)\{\}|\\[&%_#${}]", "", out)
    assert not re.search(r"[\\&%_#$^~{}]", written_out)                                   # nothing special is left bare
    assert S.latex_escape(5) == "5" and S.latex_escape("") == ""
    table = S.to_latex(pd.DataFrame({"variable": ["RV (%^2)", "x~y"], "mean_loss": [1.5, 2.0]}))
    assert r"RV (\%\textasciicircum{}2) & 1.5" in table and r"x\textasciitilde{}y & 2" in table and r"mean\_loss" in table



def test_the_variants_run_and_name_their_models():
    cfg = small_cfg(window=100, horizons=(1,))
    d = {"AAA": synth(3, 300)}
    expected = {"k_bic": {"m0", "m1", "m2", "m2c", "m3", "m4", "m5", "m6", "m7"},
                "hmm_log_rv": {"m1", "m2", "m2c", "m3", "m4", "m5", "m6", "m7"},
                "log_har": {"m3_log", "m5_log", "m6_log", "m7_log"}, "harq": {"m3_q", "m5_q"},
                "vintage": {"m3", "m2c", "m4", "m5", "m6"}, "hmm_5bar": {"m0", "m1", "m2", "m2c", "m3", "m4", "m5", "m6", "m7"}}
    for name, models in expected.items():
        res = S.run_variant(name, cfg, d, progress=lambda m: None)["AAA"]
        got = set(res.forecasts.index.get_level_values("model"))
        assert got == models, name
        assert np.isfinite(res.forecasts["forecast"]).all(), name
        setup = S.variant_setup(name, cfg)
        t = S.evaluate_run({"AAA": res}, cfg, families=setup["families"], pooled=False)
        assert "dm_by_instrument" in t, name
        if name == "k_bic":
            assert set(res.regime["k"]) <= {1, 2, 3}
        if name == "vintage":
            assert res.forecasts.index.get_level_values("target_day")[0] > d["AAA"].index[2 * 100 - 2]
    assert S.variant_setup("window_1500", cfg)["wf"].window == 1500 and S.variant_setup("window_750", cfg)["wf"].window == 750
    with pytest.raises(ValueError):
        S.variant_setup("nope", cfg)


def test_prepare_daily_target_variants_and_development_cut(tmp_path):
    m1, _ = brownian(120, 1.0, seed=21)
    m1 = m1.assign(ts=m1["ts"] + pd.Timedelta(days=0))
    write_m1(tmp_path / "SYN_M1_1_2.csv", m1)
    spec = S.InstrumentSpec("SYN", rv={"excluded_windows": [[1380, 1440]], "median_day_length": 287.0})
    base = S.prepare_daily(spec, raw_dir=tmp_path)
    pk = S.prepare_daily(spec, "parkinson", raw_dir=tmp_path)
    assert np.allclose(pk["rv"], base["parkinson"]) and pk.index.equals(base.index) and np.allclose(pk["r_cc"], base["r_cc"])
    nb = S.prepare_daily(spec, "rv_no_break", raw_dir=tmp_path)
    assert (nb["rv"] <= base["rv"] + 1e-12).all() and nb.index.equals(base.index)
    one = S.prepare_daily(spec, "rv_1min", raw_dir=tmp_path)
    assert one["rv"].notna().all() and not np.allclose(one["rv"], base["rv"])
    nr = S.prepare_daily(spec, "rv_no_rollover", raw_dir=tmp_path)
    assert not np.allclose(nr["rv"], base["rv"])
    # development cut: first N retained days, and data after the cut cannot change them
    dev = S.prepare_daily(spec, raw_dir=tmp_path, max_days=40)
    pd.testing.assert_frame_equal(dev, base.iloc[:40])
    changed = m1.copy()
    late = changed["ts"] >= changed["ts"].iloc[0].normalize() + pd.Timedelta(days=int(40 * 1.6))
    changed.loc[late, ["open", "high", "low", "close"]] *= 3.0
    write_m1(tmp_path / "SYN_M1_1_2.csv", changed)
    pd.testing.assert_frame_equal(S.prepare_daily(spec, raw_dir=tmp_path, max_days=40), dev)
    with pytest.raises(ValueError, match="development slice"):
        S.prepare_daily(spec, raw_dir=tmp_path, max_days=500)
    spec_none = S.InstrumentSpec("SYN")                                              # no fixed median: the slice's own median
    pd.testing.assert_frame_equal(S.prepare_daily(spec_none, raw_dir=tmp_path, max_days=40),
                                  S.prepare_daily(spec_none, raw_dir=tmp_path, max_days=40))


def test_config_round_trip(tmp_path):
    panel = [S.InstrumentSpec(s, "2008-11-10", "2026-10-31", {"excluded_windows": [[1380, 60]], "median_day_length": 264.0}, c)
             for s, c in (("XAUUSD", "Precious metals"), ("XAGUSD", "Precious metals"), ("EURUSD", "FX"), ("USDJPY", "FX"))]
    cfg = small_cfg(instruments=panel, long_history={"XAUUSD": "2004-06-14"}, variants=("primary", "harq"))
    p = tmp_path / "c.json"
    p.write_text(cfg.to_json())
    back = S.StudyConfig.from_json(p)
    assert back.instruments[0].rv_config().excluded_windows == ((1380, 60),) and back.variants == ("primary", "harq")
    assert back.horizons == (1, 5) and back.long_history == {"XAUUSD": "2004-06-14"} and back.burn_in == 22
    p.write_text(cfg.to_json().replace('"harq"', '"nonsense"'))
    with pytest.raises(ValueError, match="unknown variants"):
        S.StudyConfig.from_json(p)


# ------------------------------------------------------------------------------------------------ guards

PRESPEC = "# Pre-specification\n\n- **Study:** 1 of 3\n- **Version:** {v} (text)\n- **Status:** {s}\n"


def write_prespec(path, v, s):
    path.write_text(PRESPEC.format(v=v, s=s))
    return path


def test_prespec_must_be_approved_and_match_the_configuration(tmp_path):
    p = tmp_path / "PRESPEC.md"
    assert R.prespec_status(write_prespec(p, "0.2", "DRAFT. Becomes binding")) == ("0.2", "DRAFT. Becomes binding")
    with pytest.raises(SystemExit, match="0.3 or later"):
        R.check_prespec(write_prespec(p, "0.2", "APPROVED"), "0.2")
    with pytest.raises(SystemExit, match="DRAFT"):
        R.check_prespec(write_prespec(p, "0.3", "DRAFT (after audit)"), "0.3")
    with pytest.raises(SystemExit, match="configuration is for PRESPEC 0.2"):
        R.check_prespec(write_prespec(p, "0.3", "Approved by Saye"), "0.2")
    assert R.check_prespec(write_prespec(p, "0.3", "Approved by Saye"), "0.3") == "0.3"
    assert R.check_prespec(write_prespec(p, "1.0", "Approved"), "1.0") == "1.0"
    with pytest.raises(SystemExit, match="no PRESPEC.md"):
        R.check_prespec(tmp_path / "missing.md", "0.3")
    p.write_text("no header here")
    with pytest.raises(ValueError):
        R.prespec_status(p)
    assert R.version_tuple("0.10") > R.version_tuple("0.3")


def test_raw_data_hashes_are_verified(tmp_path):
    f = tmp_path / "AAA_M1_1_2.csv"
    f.write_bytes(b"abc")
    table = tmp_path / "h.tsv"
    table.write_text(f"file\tbytes\tsha256\n{f.name}\t3\t{R.sha256(f)}\n")
    assert R.verify_raw_hashes(["AAA"], tmp_path, table) == {f.name: R.sha256(f)}
    f.write_bytes(b"abd")
    with pytest.raises(SystemExit, match="differs from the hash"):
        R.verify_raw_hashes(["AAA"], tmp_path, table)
    with pytest.raises(SystemExit, match="expected one raw file"):
        R.verify_raw_hashes(["ZZZ"], tmp_path, table)


def test_full_run_refuses_without_tag_dirty_tree_second_run_and_draft(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "results").mkdir(parents=True)
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)
    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    (repo / "tracked.txt").write_text("x")
    write_prespec(repo / "PRESPEC.md", "1.0", "Approved")
    run("add", "tracked.txt", "PRESPEC.md")
    run("commit", "-q", "-m", "x")
    cfg = small_cfg(prespec_version="1.0")
    cfg_path = repo / "cfg.json"
    cfg_path.write_text(cfg.to_json())
    monkeypatch.setattr(R, "ROOT", repo)
    monkeypatch.setattr(R, "RESULTS", repo / "results")
    commit, dirty = R.git_state()
    assert len(commit) == 40 and dirty == []
    with pytest.raises(SystemExit, match="no tag prespec-v1.0"):                     # approved text, but not tagged
        R.run_full(cfg_path)
    run("tag", "prespec-v1.0")
    R.check_tag()
    write_prespec(repo / "PRESPEC.md", "1.0", "Approved, edited after the tag")
    run("commit", "-qam", "edit after the tag")
    with pytest.raises(SystemExit, match="differs from PRESPEC.md at tag"):
        R.run_full(cfg_path)
    run("tag", "-f", "prespec-v1.0")                                                  # re-tag: the edit is now the tagged text
    (repo / "tracked.txt").write_text("changed")
    assert R.git_state()[1] == [" M tracked.txt"]
    with pytest.raises(SystemExit, match="uncommitted changes"):
        R.run_full(cfg_path)
    run("checkout", "tracked.txt")
    (repo / "untracked.txt").write_text("not tracked: does not block")
    assert R.git_state()[1] == []
    (repo / "results" / "RUN_STARTED.lock").write_text("earlier")
    with pytest.raises(SystemExit, match="started before"):
        R.run_full(cfg_path)
    (repo / "results" / "RUN_STARTED.lock").unlink()
    (repo / "results" / "RUN_INFO.md").write_text("earlier")
    with pytest.raises(SystemExit, match="started before"):
        R.run_full(cfg_path)
    (repo / "results" / "RUN_INFO.md").unlink()
    for version, status, message in (("0.4", "Approved", "needs 1.0 or later"), ("1.0", "DRAFT", "DRAFT")):
        write_prespec(repo / "PRESPEC.md", version, status)
        cfg_path.write_text(small_cfg(prespec_version=version).to_json())
        with pytest.raises(SystemExit, match=message):
            R.run_full(cfg_path)
    write_prespec(repo / "PRESPEC.md", "1.0", "Approved")
    cfg_path.write_text(small_cfg(prespec_version="0.4").to_json())
    with pytest.raises(SystemExit, match="configuration is for PRESPEC 0.4"):
        R.run_full(cfg_path)
    assert not (repo / "results" / "RUN_STARTED.lock").exists()                      # a refused run leaves nothing behind


def approved_repo(tmp_path, monkeypatch):
    """A git repo with an approved PRESPEC.md (version 1.0) carrying the tag prespec-v1.0, a configuration and the run_study paths."""
    repo = tmp_path / "repo"
    (repo / "results").mkdir(parents=True)
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)
    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    write_prespec(repo / "PRESPEC.md", "1.0", "Approved")
    run("add", "PRESPEC.md")
    run("commit", "-q", "-m", "x")
    run("tag", "prespec-v1.0")
    cfg = small_cfg(prespec_version="1.0")
    cfg_path = repo / "cfg.json"
    cfg_path.write_text(cfg.to_json())
    monkeypatch.setattr(R, "ROOT", repo)
    monkeypatch.setattr(R, "RESULTS", repo / "results")
    return repo, cfg, cfg_path


def test_a_new_run_refuses_to_start_on_results_of_earlier_code(tmp_path, monkeypatch):
    """OP-1: with no lock, results left in forecasts, tables or figures could be mixed into the new run, so it does not start."""
    repo, cfg, cfg_path = approved_repo(tmp_path, monkeypatch)
    for folder, name in (("forecasts", "primary__AAA.done.json"), ("tables", "primary__mean_loss.csv"), ("figures", "primary__prob_stress.pdf")):
        (repo / "results" / folder).mkdir()
        (repo / "results" / folder / ".gitkeep").write_text("")                          # a hidden placeholder does not count
        assert R.leftovers(repo / "results" / folder) == []
        (repo / "results" / folder / name).write_text("x")
        with pytest.raises(SystemExit, match=rf"earlier results are in the way \({folder}/{name}\)"):
            R.run_full(cfg_path)
        (repo / "results" / folder / name).unlink()
    (repo / "results" / "forecasts" / "sub").mkdir()
    for i in range(5):
        (repo / "results" / "forecasts" / "sub" / f"f{i}.parquet").write_text("x")
    with pytest.raises(SystemExit, match=r"earlier results are in the way \(sub/f0.parquet, sub/f1.parquet, sub/f2.parquet and 2 more\)"):
        R.run_full(cfg_path)
    assert not (repo / "results" / "RUN_STARTED.lock").exists()                        # a refused run leaves no lock behind
    with pytest.raises(SystemExit, match="nothing to resume"):                         # resuming needs a lock, whatever lies around
        R.run_full(cfg_path, resume=True)


def test_resume_refuses_daily_rows_that_differ_from_those_of_the_started_run(tmp_path, monkeypatch):
    """The finished tasks were computed from the daily rows named in the lock; other rows void them."""
    import json
    repo, cfg, cfg_path = approved_repo(tmp_path, monkeypatch)
    commit = R.git_state()[0]
    symbols = sorted(s.symbol for s in cfg.instruments)
    stored = {f"daily__{s}__main.parquet": "a" * 64 for s in symbols}
    (repo / "results" / "RUN_STARTED.lock").write_text(json.dumps({
        "started": "2026-12-01T00:00:00+00:00", "commit": commit, "config_sha256": R.sha256(cfg_path),
        "prespec_sha256": R.sha256(repo / "PRESPEC.md"), "bundle_hashes": stored, "run_key": "x"}))
    monkeypatch.setattr(R, "verify_raw_hashes", lambda *a, **k: {})
    rebuilt = dict(stored, **{f"daily__{symbols[1]}__main.parquet": "b" * 64})
    monkeypatch.setattr(S, "prepare_all", lambda *a, **k: (rebuilt, []))
    with pytest.raises(SystemExit, match=rf"REFUSED: the daily rows daily__{symbols[1]}__main.parquet are not the ones"):
        R.run_full(cfg_path, workers=1, resume=True)


def test_run_info_records_commit_versions_and_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RESULTS", tmp_path)
    (tmp_path / "RUN_STARTED.lock").write_text("x")
    cfg_path = tmp_path / "cfg.json"
    cfg = small_cfg(prespec_version="0.3")
    cfg_path.write_text(cfg.to_json())
    from datetime import datetime, timezone
    R.write_run_info(cfg, "0.3", "a" * 40, {"AAA_M1_1_2.csv": "f" * 64}, datetime(2026, 12, 1, tzinfo=timezone.utc), 61.0,
                     [("primary", 2, 60)], cfg_path)
    text = (tmp_path / "RUN_INFO.md").read_text()
    assert "a" * 40 in text and "hmm-regime 0.2.0" in text and "f" * 64 in text and "PRESPEC version: 0.3" in text
    assert "2026-12-01" in text and "primary: 2 instruments" in text and '"window": 120' in text
    assert not (tmp_path / "RUN_STARTED.lock").exists()                              # the lock goes only after a complete run


def test_run_info_is_written_through_write_atomic_and_a_failed_write_leaves_nothing(tmp_path, monkeypatch):
    """A kill while RUN_INFO.md is being written must not leave a truncated file that blocks every later start."""
    f = tmp_path / "x.txt"
    f.write_text("old")

    def broken(p):
        p.write_text("par")
        raise OSError("disk full")

    with pytest.raises(OSError):
        S.write_atomic(f, broken)
    assert f.read_text() == "old"                                                    # the old file is untouched
    g = tmp_path / "new.txt"
    with pytest.raises(OSError):
        S.write_atomic(g, broken)
    assert not g.exists()                                                            # and a new file does not appear half written
    monkeypatch.setattr(R, "RESULTS", tmp_path)
    seen, real = [], S.write_atomic
    monkeypatch.setattr(S, "write_atomic", lambda path, w: (seen.append(Path(path).name), real(path, w))[1])
    cfg = small_cfg(prespec_version="1.0")
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(cfg.to_json())
    from datetime import datetime, timezone
    R.write_run_info(cfg, "1.0", "a" * 40, {}, datetime(2026, 12, 1, tzinfo=timezone.utc), 1.0, [], cfg_path, extra={"run_key": "k" * 64})
    assert seen == ["RUN_INFO.md"] and "k" * 64 in (tmp_path / "RUN_INFO.md").read_text()


def test_the_development_window_is_the_first_window_of_us500_of_prespec_12_8():
    """The minute data of US500 start on 2017-03-21; that day has no return, so the retained days run from 2017-03-22 to 2021-02-04."""
    assert (R.DEV_START, R.DEV_END) == ("2017-03-21", "2021-02-04")
    spec = R.dev_config().instruments[0]
    assert (spec.symbol, spec.start, spec.end) == ("US500", "2017-03-21", "2021-02-04")


def test_development_mode_writes_only_below_results_dev():
    import inspect
    src = inspect.getsource(R.run_dev)
    assert 'RESULTS / "dev"' in src and "RUN_INFO" not in src
    cfg = R.dev_config()
    assert [s.symbol for s in cfg.instruments] == ["US500"] and cfg.variants == ("primary",) and cfg.window == 500
