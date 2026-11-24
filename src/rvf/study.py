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
"""
from __future__ import annotations

import json
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
from rvf.walkforward import WFConfig, WFResult, run_walk_forward

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


# ------------------------------------------------------------------------------------------------ configuration

@dataclass
class InstrumentSpec:
    symbol: str
    start: str | None = None                  # first server day of the common sample
    end: str | None = None                    # last server day of the sample
    rv: dict = field(default_factory=dict)    # RVConfig fields (excluded_windows as a list of [start, end])
    asset_class: str = ""

    def rv_config(self, **override) -> RVConfig:
        kw = dict(self.rv)
        if "excluded_windows" in kw:
            kw["excluded_windows"] = tuple(tuple(w) for w in kw["excluded_windows"])
        kw.update(override)
        return RVConfig(**kw)

    def clean_config(self, start: str | None = None) -> CleanConfig:
        return CleanConfig(start=start or self.start, end=self.end)


@dataclass
class StudyConfig:
    prespec_version: str
    instruments: list[InstrumentSpec]
    window: int = 1000
    refit_every: int = 21
    horizons: tuple[int, ...] = (1, 5, 22)
    min_ess: float = 100.0
    ms_starts: int = 10
    mcs_reps: int = 10_000
    mcs_block: int = 10
    mcs_size: float = 0.10
    variants: tuple[str, ...] = tuple(VARIANTS)
    long_history: dict = field(default_factory=dict)       # {"symbols": [...], "start": "YYYY-MM-DD"}
    seed: int = 0

    @classmethod
    def from_json(cls, path: Path) -> "StudyConfig":
        d = json.loads(Path(path).read_text())
        d["instruments"] = [InstrumentSpec(**i) for i in d["instruments"]]
        for k in ("horizons", "variants"):
            if k in d:
                d[k] = tuple(d[k])
        cfg = cls(**d)
        unknown = [v for v in cfg.variants if v not in VARIANTS]
        if unknown:
            raise ValueError(f"unknown variants: {unknown}")
        return cfg

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=list)


def variant_setup(name: str, cfg: StudyConfig) -> dict:
    """Models factory, walk-forward and regime configuration of one variant."""
    wf = WFConfig(window=cfg.window, refit_every=cfg.refit_every, horizons=cfg.horizons)
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
        wf = replace(wf, window=750)
    elif name == "window_1500":
        wf = replace(wf, window=1500)
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

def prepare_daily(spec: InstrumentSpec, variant: str = "primary", raw_dir: Path = RAW, start: str | None = None,
                  max_days: int | None = None) -> pd.DataFrame:
    """
    Retained daily rows of one instrument for one variant. For the target variants only the column 'rv' differs; the
    retained days, the close-to-close returns and the HMM inputs are those of the primary run.

    max_days (development mode): the M1 data is cut by calendar date, about 1.6 calendar days per wanted day, before
    anything is computed from it, and the result is the first max_days retained days. If the specification fixes no
    median day length, the median comes from that slice, which holds a few weeks more than max_days.
    """
    m1, _ = clean_m1(load_m1(raw_path(spec.symbol, raw_dir)), spec.clean_config(start))
    if max_days is not None:
        m1 = m1[m1["ts"] < m1["ts"].iloc[0].normalize() + pd.Timedelta(days=int(max_days * 1.6))]
    base = retained(daily_measures(m1, spec.rv_config()))
    if max_days is not None:
        if len(base) < max_days:
            raise ValueError(f"{spec.symbol}: only {len(base)} retained days in the development slice")
        base = base.iloc[:max_days]
        m1 = m1[m1["ts"] < base.index[-1] + pd.Timedelta(days=1)]
    if variant in ("parkinson", "garman_klass"):
        base = base.assign(rv=base[variant])
    elif variant == "rv_no_break":
        base = base.assign(rv=retained(daily_measures(m1, spec.rv_config(include_session_break=False)))["rv"].reindex(base.index))
    elif variant == "rv_no_rollover":
        base = base.assign(rv=retained(daily_measures(m1, spec.rv_config(excluded_windows=())))["rv"].reindex(base.index))
    elif variant == "rv_1min":
        base = base.assign(rv=retained(daily_measures(m1, spec.rv_config(interval_min=1)))["rv"].reindex(base.index))
    if variant in TARGET_VARIANTS and not (base["rv"] > 0).all():
        raise ValueError(f"{spec.symbol}/{variant}: the target is missing or not positive on "
                         f"{int((~(base['rv'] > 0)).sum())} retained days")
    return base


# ------------------------------------------------------------------------------------------------ runs

def run_variant(name: str, cfg: StudyConfig, daily: dict[str, pd.DataFrame], progress: Callable[[str], None] = print
                ) -> dict[str, WFResult]:
    """One variant on every instrument in `daily` (symbol -> retained daily rows, prepared for this variant)."""
    setup = variant_setup(name, cfg)
    out = {}
    for sym, d in daily.items():
        progress(f"{name}: {sym} ({len(d)} days)")
        out[sym] = run_walk_forward(d, setup["factory"](), setup["wf"], setup["regime"])
    return out


# ------------------------------------------------------------------------------------------------ evaluation

def _models(losses: pd.DataFrame) -> list[str]:
    return [m for m in losses.columns if m != "m0"]


def evaluate_run(results: dict[str, WFResult], cfg: StudyConfig, families: dict | None = None, loss: str = "qlike",
                 pooled: bool = True, extras: bool = True) -> dict[str, pd.DataFrame]:
    """
    Evaluation tables of one variant. Every table is long-format. Tests use the loss differential
    loss(model a) - loss(model b) on the days where all models of an instrument have a forecast.
    """
    families = families or FAMILIES
    horizons = sorted({h for r in results.values() for h in r.forecasts.index.get_level_values("h").unique()})
    mean_rows, dm_rows, pooled_rows, mcs_rows, gw_rows, split_rows = [], [], [], [], [], []
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
            mcs_rows += [dict(h=h, instrument=s, model=m, mcs_pvalue=row.mcs_pvalue, included=bool(row.included))
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
    regime = pd.concat([r.regime.assign(instrument=s) for s, r in results.items() if r.regime is not None and len(r.regime)])
    regime_summary = regime.groupby("instrument").agg(
        origins=("prob_stress", "size"), share_stress=("prob_stress", lambda x: float((x > 0.5).mean())), refits=("refit", "sum"),
        mean_k=("k", "mean"), hmm_not_converged=("hmm_converged", lambda x: int((~x).sum())),
        hmm_warnings=("hmm_warnings", "sum")).reset_index() if len(regime) else pd.DataFrame()
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
