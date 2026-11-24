"""
The study pipeline: configuration, one run per instrument and variant, evaluation tables and figures.

Nothing here decides a value of the pre-specification. Every number that PRESPEC fixes (instruments, sample
period, rollover windows, short-day median, ...) comes from the configuration file.

Variants (PRESPEC section 10), each a full walk-forward run on every instrument it applies to:
  primary          M0-M7 and M2c, K = 2, per-bar log-range feature, window 1000, refit every 21 days
  rv_no_break, rv_no_rollover, rv_1min, parkinson, garman_klass     other targets, same days as primary (check 1)
  hmm_rolling, hmm_5bar                                              other HMM feature (check 2)
  k_bic                                                              K chosen by BIC from 1, 2, 3 (check 3)
  log_har                                                            log forms of M3, M5, M6, M7 (check 4)
  window_750, window_1500                                            other training windows (check 5)
  long_history                                                       the listed instruments from an earlier start (check 6)
  hmm_log_rv                                                         HMM fed log RV (check 8)
  vintage                                                            M4-M6 trained on real-time probabilities (check 9)
  harq                                                               HARQ against HARQ plus prob_stress (check 10)
Check 7 (MSE loss) is the same forecasts evaluated with the MSE loss, per instrument.

Running. The work is split into independent tasks, one walk-forward run per (variant, instrument). The daily rows of an
instrument are prepared once and stored (`build_bundle`, one file per instrument and sample); a task reads that file,
runs, and writes its forecasts, regime series and counters atomically with a completion marker. Tasks do not depend on
each other and use fixed seeds, so the result does not depend on the number of workers or on the order, and an
interrupted run continues where it stopped. Evaluation reads the stored task results.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from rvf import evaluate as E
from rvf import models as M
from rvf.data import RAW, CleanConfig, clean_m1, load_m1, raw_path
from rvf.realized import RVConfig, daily_measures, retained
from rvf.regime import RegimeConfig
from rvf.walkforward import HmmFirstFitError, WFConfig, WFResult, run_walk_forward

VARIANTS = ["primary", "rv_no_break", "rv_no_rollover", "rv_1min", "parkinson", "garman_klass", "hmm_rolling", "hmm_5bar",
            "k_bic", "log_har", "window_750", "window_1500", "long_history", "hmm_log_rv", "vintage", "harq"]
TARGET_VARIANTS = {"rv_no_break", "rv_no_rollover", "rv_1min", "parkinson", "garman_klass"}

# comparisons per family: (model a, model b, alternative); d = loss(a) - loss(b), "less" tests that a is better
FAMILIES = {
    "H1": [("m5", "m3", "less")],
    "H2": [("m6", "m3", "less")],
    "M4_vs_M2c": [("m4", "m2c", "less")],
    "H3": [("m5", "m7", "two-sided"), ("m6", "m7", "two-sided")],
}
LOG_FAMILIES = {"H1": [("m5_log", "m3_log", "less")], "H2": [("m6_log", "m3_log", "less")],
                "H3": [("m5_log", "m7_log", "two-sided"), ("m6_log", "m7_log", "two-sided")]}
HARQ_FAMILIES = {"HARQ": [("m5_q", "m3_q", "less")]}
GW_PAIRS = [("m5", "m3"), ("m6", "m3")]
EQUITY_INDICES = {"US500", "USTEC", "DE40"}
HMM_FAILURE_FLAG_SHARE = 0.10                       # PRESPEC 6: flag an instrument with more failures than this share of refits


# ------------------------------------------------------------------------------------------------ configuration

@dataclass
class InstrumentSpec:
    symbol: str
    start: str | None = None                  # first server day of the common sample
    end: str | None = None                    # last server day of the sample
    rv: dict = field(default_factory=dict)    # RVConfig fields (excluded_windows as a list of [start, end])
    asset_class: str = ""
    clean: dict = field(default_factory=dict)  # CleanConfig fields other than start and end (bounce_sd, ...)

    def rv_config(self, **override) -> RVConfig:
        kw = dict(self.rv)
        if "excluded_windows" in kw:
            kw["excluded_windows"] = tuple(tuple(w) for w in kw["excluded_windows"])
        kw.update(override)
        return RVConfig(**kw)

    def clean_config(self, start: str | None = None) -> CleanConfig:
        return CleanConfig(start=start or self.start, end=self.end, **self.clean)


@dataclass
class StudyConfig:
    prespec_version: str
    instruments: list[InstrumentSpec]
    window: int = 1000
    refit_every: int = 21
    burn_in: int = 22
    alt_windows: tuple[int, int] = (750, 1500)             # training windows of robustness check 5
    horizons: tuple[int, ...] = (1, 5, 22)
    min_ess: float = 100.0
    ms_starts: int = 10
    mcs_reps: int = 10_000
    mcs_block: int = 10
    mcs_size: float = 0.10
    variants: tuple[str, ...] = tuple(VARIANTS)
    long_history: dict = field(default_factory=dict)       # symbol -> first day of its own minute data (check 6)
    seed: int = 0

    @classmethod
    def from_json(cls, path: Path) -> "StudyConfig":
        d = json.loads(Path(path).read_text())
        d["instruments"] = [InstrumentSpec(**i) for i in d["instruments"]]
        for k in ("horizons", "variants", "alt_windows"):
            if k in d:
                d[k] = tuple(d[k])
        cfg = cls(**d)
        unknown = [v for v in cfg.variants if v not in VARIANTS]
        if unknown:
            raise ValueError(f"unknown variants: {unknown}")
        cfg.validate_panel()
        return cfg

    def validate_panel(self) -> None:
        """PRESPEC 3 and 12.2: no equity index, at least four instruments from at least two asset classes."""
        symbols = [i.symbol for i in self.instruments]
        indices = sorted(set(symbols) & EQUITY_INDICES)
        if indices:
            raise ValueError(f"equity indices are not part of the study: {indices}")
        if len(set(symbols)) != len(symbols):
            raise ValueError("an instrument is listed twice")
        classes = {i.asset_class for i in self.instruments}
        if len(symbols) < 4 or len(classes) < 2:
            raise ValueError(f"the panel needs at least four instruments from at least two asset classes "
                             f"(got {len(symbols)} from {len(classes)})")
        unknown = sorted(set(self.long_history) - set(symbols))
        if unknown:
            raise ValueError(f"long_history names instruments outside the panel: {unknown}")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=list)


def variant_setup(name: str, cfg: StudyConfig) -> dict:
    """Models factory, walk-forward and regime configuration of one variant."""
    wf = WFConfig(window=cfg.window, refit_every=cfg.refit_every, horizons=cfg.horizons, burn_in=cfg.burn_in)
    regime = RegimeConfig()
    factory: Callable[[], list] = lambda: M.all_models(cfg.min_ess, cfg.ms_starts)
    families = FAMILIES
    if name == "hmm_rolling":
        regime = RegimeConfig(vol_feature="rolling", vol_window=21)
    elif name == "hmm_5bar":
        regime = RegimeConfig(vol_feature="rolling", vol_window=5)
    elif name == "k_bic":
        regime = RegimeConfig(n_components=(1, 2, 3))
    elif name == "log_har":
        # the log models need M3_log as the fallback of M7_log; M2c is not part of this check
        factory, families = (lambda: M.log_models(cfg.min_ess, cfg.ms_starts)), LOG_FAMILIES
    elif name == "window_750":
        wf = replace(wf, window=cfg.alt_windows[0])
    elif name == "window_1500":
        wf = replace(wf, window=cfg.alt_windows[1])
    elif name == "hmm_log_rv":
        regime = RegimeConfig(feature_cols=("log_rv",), sort_by="mean:0")
        factory = lambda: [m for m in M.all_models(cfg.min_ess, cfg.ms_starts) if m.name != "m0"]   # no return variance
    elif name == "vintage":
        wf = replace(wf, vintage_training=True)
        factory = lambda: [M.Har(), M.CalibratedGarch(), M.CalibratedGarch(with_regime=True), M.Har(with_regime=True),
                           M.RegimeWeightedHar(cfg.min_ess)]
    elif name == "harq":
        factory, families = M.harq_models, HARQ_FAMILIES
    elif name not in VARIANTS:
        raise ValueError(f"unknown variant {name}")
    if name == "vintage":
        families = {k: v for k, v in FAMILIES.items() if k != "H3"}
    return dict(wf=wf, regime=regime, factory=factory, families=families)


# ------------------------------------------------------------------------------------------------ data

ALT_TARGETS = {"rv_no_break": {"include_session_break": False}, "rv_no_rollover": {"excluded_windows": ()},
               "rv_1min": {"interval_min": 1}}
RANGE_TARGETS = ("parkinson", "garman_klass")


def build_bundle(spec: InstrumentSpec, raw_dir: Path = RAW, start: str | None = None, max_days: int | None = None,
                 raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Retained daily rows of one instrument: the primary measures plus one column per alternative target of robustness
    check 1 (rv_no_break, rv_no_rollover, rv_1min; Parkinson and Garman-Klass are columns of the primary measures).

    The alternative targets are computed with the main run's day flags (short, stale, anchor), so every alternative
    has exactly the main run's days and a day cannot be lost because a changed window or interval would flag it.
    An alternative that is missing or not positive on a retained day raises: it is a data problem to be seen before
    any forecast is made.

    max_days (development mode): the M1 data is cut by calendar date, about 1.6 calendar days per wanted day, before
    anything is computed from it, and the result is the first max_days retained days. If the specification fixes no
    median day length, the median comes from that slice, which holds a few weeks more than max_days.
    raw: the loaded M1 file, if the caller has it already.
    """
    raw = load_m1(raw_path(spec.symbol, raw_dir)) if raw is None else raw
    m1, _ = clean_m1(raw, spec.clean_config(start))
    if max_days is not None:
        m1 = m1[m1["ts"] < m1["ts"].iloc[0].normalize() + pd.Timedelta(days=int(max_days * 1.6))]
    primary = daily_measures(m1, spec.rv_config())
    base = retained(primary)
    if max_days is not None:
        if len(base) < max_days:
            raise ValueError(f"{spec.symbol}: only {len(base)} retained days in the development slice")
        base = base.iloc[:max_days]
        m1 = m1[m1["ts"] < base.index[-1] + pd.Timedelta(days=1)]
    base = base.copy()
    flags = primary[["short", "stale", "anchor"]]
    for name, override in ALT_TARGETS.items():
        alt = daily_measures(m1, spec.rv_config(**override), flags=flags)
        if not alt["kept"].reindex(base.index).fillna(False).all():
            raise ValueError(f"{spec.symbol}/{name}: the alternative run does not keep the main run's days")
        base[name] = alt["rv"].reindex(base.index)
    bad = ~(base[[*ALT_TARGETS, *RANGE_TARGETS]] > 0).all(axis=1)
    if bad.any():
        raise ValueError(f"{spec.symbol}: an alternative target is missing or not positive on {int(bad.sum())} retained "
                         f"days (first: {base.index[bad][0].date()})")
    return base


def variant_daily(bundle: pd.DataFrame, variant: str) -> pd.DataFrame:
    """The daily rows one variant runs on: for a target variant the column 'rv' is replaced by that target."""
    if variant in ALT_TARGETS or variant in RANGE_TARGETS:
        bundle = bundle.assign(rv=bundle[variant])
    return bundle.drop(columns=list(ALT_TARGETS))


def prepare_daily(spec: InstrumentSpec, variant: str = "primary", raw_dir: Path = RAW, start: str | None = None,
                  max_days: int | None = None) -> pd.DataFrame:
    """Retained daily rows of one instrument for one variant (see `build_bundle`)."""
    return variant_daily(build_bundle(spec, raw_dir, start, max_days), variant)


# ------------------------------------------------------------------------------------------------ runs

def run_variant(name: str, cfg: StudyConfig, daily: dict[str, pd.DataFrame], progress: Callable[[str], None] = print,
                stopped: dict | None = None) -> dict[str, WFResult]:
    """
    One variant on every instrument in `daily` (symbol -> retained daily rows, prepared for this variant).
    An instrument whose first HMM fit fails is stopped (PRESPEC 6): it is left out of the result and its reason is
    written to `stopped` (without that dictionary the error is raised).
    """
    setup = variant_setup(name, cfg)
    out = {}
    for sym, d in daily.items():
        progress(f"{name}: {sym} ({len(d)} days)")
        try:
            out[sym] = run_walk_forward(d, setup["factory"](), setup["wf"], setup["regime"])
        except HmmFirstFitError as e:
            if stopped is None:
                raise
            stopped[sym] = str(e)
    return out


# ------------------------------------------------------------------------------------------------ tasks

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def limit_threads() -> None:
    """One BLAS thread per process: parallel workers then do not fight for cores, and each run is reproducible."""
    for v in THREAD_VARS:
        os.environ[v] = "1"


def single_thread():
    """A context in which the BLAS libraries of this process use one thread (so every run computes the same numbers)."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:                                    # threadpoolctl comes with scikit-learn, which hmmlearn needs
        return contextlib.nullcontext()
    return threadpool_limits(limits=1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(path: Path, write: Callable[[Path], None]) -> None:
    """Write through a temporary file and rename, so that a file that exists is a complete file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def sample_of(variant: str) -> str:
    return "long" if variant == "long_history" else "main"


def bundle_file(clean_dir: Path, symbol: str, sample: str) -> Path:
    return Path(clean_dir) / f"daily__{symbol}__{sample}.parquet"


def required_days(cfg: StudyConfig, sample: str) -> tuple[int, str]:
    """The fewest retained days that every variant on this sample needs (burn-in, window and at least one forecast)."""
    need, why = 0, ""
    for v in cfg.variants:
        if sample_of(v) != sample:
            continue
        wf = variant_setup(v, cfg)["wf"]
        n = wf.burn_in + (2 * wf.window + wf.refit_every if wf.vintage_training else wf.window) + 2
        if n > need:
            need, why = n, f"variant {v}"
    return need, why


def prepare_instrument(config_path: str, symbol: str, clean_dir: str, raw_dir: str) -> dict[str, str]:
    """Stage 0 of the full run: the daily bundles of one instrument (its sample, and for check 6 its own history)."""
    cfg = StudyConfig.from_json(Path(config_path))
    spec = next(i for i in cfg.instruments if i.symbol == symbol)
    raw = load_m1(raw_path(symbol, Path(raw_dir)))
    out = {}
    samples = [("main", None)]
    if "long_history" in cfg.variants and symbol in cfg.long_history:
        samples.append(("long", cfg.long_history[symbol]))
    for sample, start in samples:
        bundle = build_bundle(spec, Path(raw_dir), start, raw=raw)
        need, why = required_days(cfg, sample)
        if len(bundle) < need:                                # caught here, before the lock: no task could run
            raise ValueError(f"{symbol} ({sample} sample): {len(bundle)} retained days, but {why} needs {need}")
        path = bundle_file(Path(clean_dir), symbol, sample)
        write_atomic(path, lambda p, b=bundle: b.to_parquet(p))
        out[path.name] = sha256_file(path)
    return out


def task_marker(out_dir: Path, variant: str, symbol: str) -> Path:
    return Path(out_dir) / f"{variant}__{symbol}.done.json"


def task_done(out_dir: Path, variant: str, symbol: str) -> dict | None:
    """The completion marker of a task if the task finished and its files are intact, else None."""
    m = task_marker(out_dir, variant, symbol)
    if not m.exists():
        return None
    info = json.loads(m.read_text())
    if info["status"] == "ok":
        for name, digest in info["files"].items():
            f = Path(out_dir) / name
            if not f.exists() or sha256_file(f) != digest:
                return None
    return info


def save_task(res: WFResult, out_dir: Path, variant: str, symbol: str, seconds: float) -> dict:
    out_dir = Path(out_dir)
    files = {}
    for name, df in (("forecasts", res.forecasts.reset_index()), ("regime", res.regime.reset_index()), ("counters", res.counters)):
        p = out_dir / f"{variant}__{symbol}__{name}.parquet"
        write_atomic(p, lambda q, d=df: d.to_parquet(q, index=False))
        files[p.name] = sha256_file(p)
    info = {"variant": variant, "symbol": symbol, "status": "ok", "files": files, "seconds": round(seconds, 1)}
    write_atomic(task_marker(out_dir, variant, symbol), lambda q: q.write_text(json.dumps(info, indent=1)))
    return info


def save_stopped(out_dir: Path, variant: str, symbol: str, reason: str, seconds: float) -> dict:
    info = {"variant": variant, "symbol": symbol, "status": "stopped", "reason": reason, "seconds": round(seconds, 1)}
    write_atomic(task_marker(Path(out_dir), variant, symbol), lambda q: q.write_text(json.dumps(info, indent=1)))
    return info


def run_task(config_path: str, variant: str, symbol: str, clean_dir: str, out_dir: str) -> dict:
    """One walk-forward run (one variant, one instrument), written to disk. Top-level so that a worker can import it."""
    cfg = StudyConfig.from_json(Path(config_path))
    daily = variant_daily(pd.read_parquet(bundle_file(Path(clean_dir), symbol, sample_of(variant))), variant)
    setup = variant_setup(variant, cfg)
    t0 = time.time()
    try:
        with single_thread():
            res = run_walk_forward(daily, setup["factory"](), setup["wf"], setup["regime"])
    except HmmFirstFitError as e:                          # PRESPEC 6: the instrument is stopped and reported
        return save_stopped(Path(out_dir), variant, symbol, str(e), time.time() - t0)
    return save_task(res, Path(out_dir), variant, symbol, time.time() - t0)


def _pool(workers: int, fn: Callable, jobs: list[tuple], progress: Callable[[str], None], describe: Callable
          ) -> tuple[list, list[tuple]]:
    """Run fn(*job) for every job, in this process (workers <= 1) or in a pool. Returns (results, failures): the results of
    the jobs that finished and the (job, error) pairs of those that failed."""
    failures, results = [], []

    def finished(job, result):
        results.append(result)
        extra = f" ({result['seconds']} s)" if isinstance(result, dict) and "seconds" in result else ""
        progress(f"done: {describe(job)}{extra}")

    def failed(job, e):
        failures.append((job, repr(e)))
        progress(f"FAILED: {describe(job)}: {e!r}")

    if workers <= 1:
        for job in jobs:
            try:
                finished(job, fn(*job))
            except Exception as e:
                failed(job, e)
        return results, failures
    limit_threads()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        futures = {ex.submit(fn, *job): job for job in jobs}
        for f in as_completed(futures):
            try:
                finished(futures[f], f.result())
            except Exception as e:
                failed(futures[f], e)
    return results, failures


def prepare_all(cfg: StudyConfig, config_path: Path, clean_dir: Path, raw_dir: Path = RAW, workers: int = 1,
                progress: Callable[[str], None] = print, only: list[str] | None = None) -> tuple[dict[str, str], list[tuple]]:
    """
    Stage 0: the daily bundles of every instrument (or of those named in `only`). Returns the SHA-256 of every bundle
    file and the failures. Each job loads one whole raw file, so at most two run at the same time.
    """
    jobs = [(str(config_path), i.symbol, str(clean_dir), str(raw_dir)) for i in cfg.instruments if only is None or i.symbol in only]
    results, failures = _pool(min(workers, 2), prepare_instrument, jobs, progress, lambda job: f"daily rows of {job[1]}")
    hashes = {}
    for r in results:
        hashes.update(r)
    return hashes, failures


def execute_tasks(tasks: list[tuple[str, str]], config_path: Path, clean_dir: Path, out_dir: Path, workers: int = 1,
                  progress: Callable[[str], None] = print) -> list[tuple]:
    """Run the tasks that are not finished yet; returns the failures. Tasks already finished are skipped (resume)."""
    todo = [(str(config_path), v, s, str(clean_dir), str(out_dir)) for v, s in tasks if task_done(out_dir, v, s) is None]
    progress(f"{len(tasks) - len(todo)} of {len(tasks)} tasks are already finished; {len(todo)} to run, {max(workers, 1)} worker(s)")
    return _pool(workers, run_task, todo, progress, lambda job: f"task {job[1]} / {job[2]}")[1]


def load_variant_results(out_dir: Path, variant: str, symbols: list[str]) -> tuple[dict[str, WFResult], dict[str, str]]:
    """The stored results of a variant: (results of the instruments that ran, reasons of those that were stopped)."""
    results, stopped = {}, {}
    for s in symbols:
        info = task_done(out_dir, variant, s)
        if info is None:
            raise FileNotFoundError(f"task {variant} / {s} has no complete result in {out_dir}")
        if info["status"] == "stopped":
            stopped[s] = info["reason"]
            continue
        f = {k.split("__")[-1].replace(".parquet", ""): Path(out_dir) / k for k in info["files"]}
        fc = pd.read_parquet(f["forecasts"]).set_index(["target_day", "model", "h"]).sort_index()
        results[s] = WFResult(fc, pd.read_parquet(f["regime"]).set_index("origin"), pd.read_parquet(f["counters"]))
    return results, stopped


# ------------------------------------------------------------------------------------------------ evaluation

def _models(losses: pd.DataFrame) -> list[str]:
    return [m for m in losses.columns if m != "m0"]


def hmm_summary(results: dict[str, WFResult]) -> pd.DataFrame:
    """Per instrument: origins, refits, HMM failures (PRESPEC 6), the failure share and the flag for more than 10%."""
    rows = []
    for sym, r in results.items():
        g = r.regime
        refits = int(g["refit"].sum())
        fails = int(g["hmm_failed"].sum())
        share = fails / refits if refits else np.nan
        rows.append(dict(instrument=sym, origins=len(g), refits=refits, hmm_failures=fails, hmm_failure_share=share,
                         flag_failures_over_10pct=bool(share > HMM_FAILURE_FLAG_SHARE), share_stress=float((g["prob_stress"] > 0.5).mean()),
                         mean_k=float(g["k"].mean()), hmm_not_converged=int((~g["hmm_converged"]).sum()),
                         hmm_warnings=int(g["hmm_warnings"].sum())))
    return pd.DataFrame(rows)


def evaluate_run(results: dict[str, WFResult], cfg: StudyConfig, families: dict | None = None, loss: str = "qlike",
                 pooled: bool = True, extras: bool = True, effects: bool = True) -> dict[str, pd.DataFrame]:
    """
    Evaluation tables of one variant. Every table is long-format. Tests use the loss differential
    loss(model a) - loss(model b) on the days where all models of an instrument have a forecast.
    """
    families = families or FAMILIES
    horizons = sorted({h for r in results.values() for h in r.forecasts.index.get_level_values("h").unique()})
    mean_rows, dm_rows, pooled_rows, mcs_rows, gw_rows, split_rows, effect_rows = [], [], [], [], [], [], []
    for h in horizons:
        tabs = {s: E.loss_table(r.forecasts, h, loss) for s, r in results.items()}
        for s, t in tabs.items():
            mse_t = E.loss_table(results[s].forecasts, h, "mse") if loss == "qlike" else t
            for m in t.columns:
                mean_rows.append(dict(h=h, instrument=s, model=m, n=len(t), mean_loss=t[m].mean(),
                                      mean_other=mse_t[m].mean(), loss=loss))
        for fam, comps in families.items():
            fam_rows = []
            for a, b, alt in comps:
                diffs = {s: t[a] - t[b] for s, t in tabs.items() if a in t and b in t}
                if not diffs:
                    continue
                for s, d in diffs.items():
                    r = E.dm_test(d.to_numpy(), h=h, alternative=alt)
                    fam_rows.append(dict(h=h, family=fam, comparison=f"{a} vs {b}", alternative=alt, instrument=s,
                                         mean_diff=r["mean"], stat=r["stat"], pvalue=r["pvalue"], n=r["n"], lag=r["lag"]))
                if effects and loss == "qlike":                      # PRESPEC 8.6: model a against reference b
                    base = dict(h=h, family=fam, model=a, reference=b)
                    for s_, t in tabs.items():
                        if a in t and b in t:
                            effect_rows.append(dict(base, scope=s_, **E.effect_size(t[b], t[a], h=h)))
                    if len(tabs) > 1:
                        effect_rows.append(dict(base, scope="pooled", **E.pooled_effect_size(
                            pd.DataFrame({s_: t[b] for s_, t in tabs.items() if a in t and b in t}),
                            pd.DataFrame({s_: t[a] for s_, t in tabs.items() if a in t and b in t}), h=h)))
                if pooled and len(diffs) > 1:
                    r = E.pooled_dm(pd.DataFrame(diffs), h=h, alternative=alt)
                    pooled_rows.append(dict(h=h, family=fam, comparison=f"{a} vs {b}", alternative=alt, mean_diff=r["mean"],
                                            stat=r["stat"], pvalue=r["pvalue"], n=r["n"], lag=r["lag"],
                                            instruments_per_day=r["instruments_per_day"]))
            if fam_rows:                                         # Holm across instruments (and comparisons of a family)
                f = pd.DataFrame(fam_rows)
                f["pvalue_holm"] = E.holm(f["pvalue"].fillna(1.0).to_numpy())
                dm_rows.append(f)
        if not extras:
            continue
        for s, t in tabs.items():
            r = results[s]
            mcs = E.model_confidence_set(t[_models(t)], size=cfg.mcs_size, reps=cfg.mcs_reps, block=cfg.mcs_block, seed=cfg.seed)
            mcs_rows += [dict(h=h, instrument=s, model=m, mcs_pvalue=row.mcs_pvalue, included=row.included, status=row.status)
                         for m, row in mcs.iterrows()]
            if r.regime is None or r.regime.empty:
                continue
            p = E.at_origin(r.forecasts, r.regime["prob_stress"], h).reindex(t.index)
            for a, b in GW_PAIRS:
                if a in t and b in t:
                    g = E.gw_test((t[a] - t[b]).to_numpy(), np.column_stack([np.ones(len(t)), p.to_numpy()]), h=h)
                    gw_rows.append(dict(h=h, instrument=s, comparison=f"{a} vs {b}", stat=g["stat"], pvalue=g["pvalue"], n=g["n"]))
            sp = E.loss_splits(t, p).reset_index()
            sp.insert(0, "instrument", s)
            sp.insert(0, "h", h)
            split_rows.append(sp)
    counters = pd.concat([r.counters.assign(instrument=s) for s, r in results.items()], ignore_index=True)
    regime_summary = hmm_summary(results)
    tables = {"mean_loss": pd.DataFrame(mean_rows), "counters": counters, "regime_summary": regime_summary}
    if dm_rows:
        tables["dm_by_instrument"] = pd.concat(dm_rows, ignore_index=True)
    if pooled_rows:
        tables["dm_pooled"] = pd.DataFrame(pooled_rows)
    if mcs_rows:
        tables["mcs"] = pd.DataFrame(mcs_rows)
    if gw_rows:
        tables["gw_conditional"] = pd.DataFrame(gw_rows)
    if split_rows:
        tables["loss_splits"] = pd.concat(split_rows, ignore_index=True)
    if effect_rows:
        tables["effect_sizes"] = pd.DataFrame(effect_rows)
    return tables


# ------------------------------------------------------------------------------------------------ output

def latex_escape(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#").replace("$", r"\$"))


def to_latex(df: pd.DataFrame, float_fmt: str = "{:.4g}") -> str:
    """Plain booktabs table without extra dependencies."""
    def cell(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else float_fmt.format(v)
        if isinstance(v, pd.Timestamp):
            return str(v.date())
        return latex_escape(v)
    cols = "l" * len(df.columns)
    lines = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule", " & ".join(latex_escape(c) for c in df.columns) + r" \\", "\\midrule"]
    lines += [" & ".join(cell(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def write_tables(tables: dict[str, pd.DataFrame], outdir: Path, prefix: str) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in tables.items():
        for ext, text in (("csv", None), ("tex", to_latex(df))):
            p = outdir / f"{prefix}__{name}.{ext}"
            if ext == "csv":
                df.to_csv(p, index=False, float_format="%.8g")
            else:
                p.write_text(text)
            paths.append(p)
    return paths


def save_forecasts(results: dict[str, WFResult], outdir: Path, prefix: str) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for s, r in results.items():
        p = outdir / f"{prefix}__{s}__forecasts.parquet"
        r.forecasts.reset_index().to_parquet(p, index=False)
        q = outdir / f"{prefix}__{s}__regime.parquet"
        r.regime.reset_index().to_parquet(q, index=False)
        paths += [p, q]
    return paths


def make_figures(results: dict[str, WFResult], outdir: Path, prefix: str, a: str = "m5", b: str = "m3") -> list[Path]:
    """Two PDFs: real-time prob_stress per instrument and the cumulative QLIKE difference a - b (h = 1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    n = len(results)
    paths = []
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.8 * n + 0.6), sharex=False, squeeze=False)
    for ax, (s, r) in zip(axes[:, 0], results.items()):
        ax.plot(r.regime.index, r.regime["prob_stress"], lw=0.6, color="0.2")
        ax.axhline(0.5, lw=0.5, color="tab:red", ls="--")
        ax.set_ylim(0, 1)
        ax.set_ylabel(s)
    axes[0, 0].set_title("Real-time stress probability (forward filter)")
    fig.tight_layout()
    paths.append(outdir / f"{prefix}__prob_stress.pdf")
    fig.savefig(paths[-1])
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for s, r in results.items():
        t = E.loss_table(r.forecasts, 1, "qlike")
        if a in t and b in t:
            ax.plot(t.index, (t[a] - t[b]).cumsum(), lw=0.9, label=s)
    ax.axhline(0, lw=0.5, color="0.4")
    ax.set_title(f"Cumulative QLIKE difference {a} - {b} (below zero favours {a})")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    paths.append(outdir / f"{prefix}__cum_qlike_diff.pdf")
    fig.savefig(paths[-1])
    plt.close(fig)
    return paths


# ------------------------------------------------------------------------------------------------ evaluation of stored runs

FIGURE_PAIRS = {"log_har": ("m5_log", "m3_log"), "harq": ("m5_q", "m3_q")}


def variant_symbols(cfg: StudyConfig, variant: str) -> list[str]:
    """The instruments a variant runs on: all of the panel, and for check 6 those with an own, longer history."""
    return [i.symbol for i in cfg.instruments if variant != "long_history" or i.symbol in cfg.long_history]


def descriptive_tables(bundles: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Descriptive statistics of the daily series of each instrument (the retained days of its sample, burn-in included)
    and the correlation matrix of daily log RV across the instruments on the days they all have.

    Per instrument and variable (daily return in percent, RV in percent squared, log RV): number of days, first and last
    day, mean, standard deviation, minimum, median, maximum, skewness and excess kurtosis (pandas definitions, bias
    corrected) and the autocorrelations at lags 1, 5 and 22.
    """
    rows = []
    for sym, d in bundles.items():
        for name, x in (("return (%)", d["r_cc"]), ("RV (%^2)", d["rv"]), ("log RV", np.log(d["rv"]))):
            rows.append(dict(instrument=sym, variable=name, n=int(x.notna().sum()), first=d.index[0], last=d.index[-1],
                             mean=x.mean(), sd=x.std(), min=x.min(), median=x.median(), max=x.max(), skewness=x.skew(),
                             excess_kurtosis=x.kurt(), acf1=x.autocorr(1), acf5=x.autocorr(5), acf22=x.autocorr(22)))
    log_rv = pd.DataFrame({s: np.log(d["rv"]) for s, d in bundles.items()})
    common = log_rv.dropna()
    corr = common.corr().rename_axis("instrument").reset_index()
    sample = pd.DataFrame([dict(instruments=", ".join(log_rv.columns), common_days=len(common),
                                first=common.index[0] if len(common) else pd.NaT, last=common.index[-1] if len(common) else pd.NaT)])
    return {"descriptive_statistics": pd.DataFrame(rows), "correlation_log_rv": corr, "correlation_log_rv_sample": sample}


def evaluate_all(cfg: StudyConfig, out_dir: Path, tables_dir: Path, figures_dir: Path,
                 progress: Callable[[str], None] = print, clean_dir: Path | None = None) -> list[tuple[str, int, float]]:
    """
    Evaluate every variant from its stored task results: tables (CSV and LaTeX), figures, the instruments that were
    stopped, and for the primary run the MSE-loss tables of robustness check 7. With clean_dir (the daily rows of stage 0)
    also the descriptive statistics and the log-RV correlation matrix of the panel. Returns (variant, instruments, seconds).
    """
    summary = []
    if clean_dir is not None:
        bundles = {i.symbol: variant_daily(pd.read_parquet(bundle_file(clean_dir, i.symbol, "main")), "primary") for i in cfg.instruments}
        write_tables(descriptive_tables(bundles), tables_dir, "data")
        progress("wrote the descriptive statistics and the log-RV correlation matrix")
    for name in cfg.variants:
        results, stopped = load_variant_results(out_dir, name, variant_symbols(cfg, name))
        if stopped:                                          # PRESPEC 6: first HMM fit failed, instrument stopped
            Path(tables_dir).mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"instrument": list(stopped), "reason": list(stopped.values())}).to_csv(
                Path(tables_dir) / f"{name}__stopped_instruments.csv", index=False)
        task_seconds = sum(task_done(out_dir, name, s)["seconds"] for s in variant_symbols(cfg, name))
        if results:
            setup = variant_setup(name, cfg)
            write_tables(evaluate_run(results, cfg, families=setup["families"]), tables_dir, name)
            make_figures(results, figures_dir, name, *FIGURE_PAIRS.get(name, ("m5", "m3")))
            if name == "primary":                            # robustness check 7: the same forecasts under the MSE loss
                write_tables(evaluate_run(results, cfg, loss="mse", pooled=False, extras=False, effects=False), tables_dir, "mse_loss")
        summary.append((name, len(results), task_seconds))
        progress(f"evaluated: {name} ({len(results)} instruments, {len(stopped)} stopped)")
    return summary
