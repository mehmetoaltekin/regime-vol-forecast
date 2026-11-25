# regime-vol-forecast

Study 1 of 3 in a series on hidden-Markov-model (HMM) regimes: do causal, real-time regime probabilities improve
out-of-sample daily realised-variance forecasts of standard models (EWMA, GARCH(1,1), HAR-RV)?

The design is pre-specified in [`PRESPEC.md`](PRESPEC.md). It is a draft (version 0.4) until it is approved and tagged
`prespec-v1.0`, which has to happen before the full run. This repository holds the data pipeline, the estimation and
evaluation code, and the unit tests. It holds no data and, until the full run, no forecasts or forecast results;
`results/` holds data-quality reports only.

## Panel and data

- Five instruments: XAUUSD, XAGUSD (precious metals), EURUSD, USDJPY, GBPUSD (FX). Sample 2008-11-10 to 2026-10-31.
- Source: 1-minute bars exported from the MetaQuotes-Demo server of MetaTrader 5, bid-based, server time. The raw
  files stay out of git (`data/`); `results/raw_data_sha256.tsv` holds their SHA-256 hashes.
- US500, USTEC and DE40 were exported and audited but are not part of the study. Only the first training window of
  US500 is used, for development: minute data from 2017-03-21 to 2021-02-04, whose first retained day is 2017-03-22
  (PRESPEC 12.8), so 990 retained days.
- The audit is in [`results/data_audit.md`](results/data_audit.md), the effect of every cleaning rule in
  [`results/sample_definition.md`](results/sample_definition.md).

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/). The package versions are pinned in `pyproject.toml` and `uv.lock`:

| package | version | package | version |
|---|---|---|---|
| hmm-regime | 0.2.0 | statsmodels | 0.15.0 |
| hmmlearn | 0.3.3 | scipy | 1.18.1 |
| arch | 9.0.0 | numpy | 2.5.5 |
| pandas | 3.1.1 | pyarrow | 26.0.1 |
| matplotlib | 3.11.2 | pytest | 9.2.0 |

```bash
uv sync
```

## From `data/raw/` to `results/`

Run the steps in this order; steps 2, 3 and 5 and the full run read the raw files of step 1.

```bash
# 1. raw data (macOS, MetaTrader 5 installed and logged in to MetaQuotes-Demo, terminal closed; about 75 minutes)
python3 src/mt5_export.py
#    -> data/raw/<SYMBOL>_M1_<first>_<last>.csv, data/raw/export_meta.tsv, results/raw_data_sha256.tsv

# 2. data audit (data quality only; about 40 seconds)
uv run python src/audit_data.py
#    -> results/data_audit.md, results/data_audit/

# 3. sample definition: what each rule of PRESPEC 12.5-12.7 removes, per instrument
uv run python src/sample_definition.py
#    -> results/sample_definition.md, results/sample_definition/<SYMBOL>_days.csv

# 4. unit tests (including a no-look-ahead test for every model and for the regime series; a few minutes, depending on the machine)
uv run pytest
#    dress rehearsal of the whole full run (all variants, worker processes, an interruption and a resume) on
#    synthetic data; several minutes
uv run pytest -m slow

# 5. development run: US500, first training window only, window 500; writes to results/dev/ (git-ignored)
uv run python src/run_study.py --dev
```

### The full run

The full run is made once, on frozen code, and only when Saye says so:

```bash
caffeinate -i uv run python src/run_study.py --full --config config/study_config.json --workers 8
```

`caffeinate -i` keeps the Mac awake. The work is split into one walk-forward task per (variant, instrument), run in
`--workers` processes (default: CPU count minus two) with one BLAS thread each, so the numbers do not depend on the number
of workers. The Markov-switching HAR (M7) dominates the cost: about 2.5 to 7 seconds per refit and horizon on a
1000-day window (measured on synthetic data on a development Mac), roughly 40 minutes per instrument and variant. Added
up over the five instruments and the variants that contain M7 this is on the order of 55 hours of single-core time, so
about 7 to 8 hours with 8 workers. This is an estimate and depends on the machine and its load; `results/run_progress.log`
shows the real progress.

If the run is interrupted (machine asleep, process killed), continue it:

```bash
caffeinate -i uv run python src/run_study.py --full --config config/study_config.json --workers 8 --resume
```

`--resume` needs `results/RUN_STARTED.lock`, the same commit, the same configuration, the same `PRESPEC.md` and the same
daily rows (`data/clean/`), and skips the tasks that are finished. Every stored task carries a run key, a hash of those
four, and only tasks with the key of the run are used, by `--resume` and by the evaluation. If the code has to change,
the finished tasks are void: move `results/forecasts/`, `results/tables/` and `results/figures/` away, delete the lock and
start again; a run without a lock refuses to start while those folders hold files.

If something goes wrong:

- `REFUSED: earlier results are in the way`: a new run starts from empty `results/forecasts/`, `results/tables/` and
  `results/figures/`. Move them out of `results/`.
- Both `results/RUN_INFO.md` and `results/RUN_STARTED.lock` exist: the run finished, and the process was stopped between
  writing `RUN_INFO.md` and removing the lock. Nothing is left to do; delete the lock.
- Stage 2 stopped (an error or a kill), the lock exists and `RUN_INFO.md` does not: if the cause is outside the code (a
  kill, a full disk), run `--resume`; the tasks are finished, so only stage 2 runs again. A fix in the code is a new
  commit, and `--resume` then refuses ("the commit is not the one the run was started with"): the finished tasks belong
  to the old commit, so the run starts again as described above. `RUN_INFO.md` and the lock are written through a
  temporary file, so a kill cannot leave a half-written one.
- A task marker (`results/forecasts/*.done.json`) that is empty or damaged counts as an unfinished task: `--resume`
  computes that task again.

The run has three stages. Stage 0 prepares the daily rows of every instrument (`data/clean/`, git-ignored); a data problem
stops the run here, before the lock is written. Stage 1 runs the tasks. Stage 2 evaluates the stored results.

It refuses to start unless `PRESPEC.md` is version 1.0 or later and no longer says DRAFT, the repo carries the tag
`prespec-v1.0` on a commit whose `PRESPEC.md` equals the current one, the configuration names the same version, tracked
files are committed, the raw files match their stored hashes, neither `results/RUN_INFO.md` nor
`results/RUN_STARTED.lock` exists, and `results/forecasts/`, `results/tables/` and `results/figures/` are empty.

It writes the tables of the primary run and of the ten robustness checks to `results/tables/` (CSV and LaTeX), the
figures to `results/figures/` (PDF), the forecasts to `results/forecasts/` (parquet, git-ignored) and `results/RUN_INFO.md`
(commit hash, run key, package versions, hashes of the raw data and of the prepared daily rows, run date, configuration).

Every stored task has a regime table (`<variant>__<symbol>__regime.parquet`): `prob_stress` at each forecast origin, K, whether
the HMM was refitted, converged or failed at that origin, and the diagnostics of the calibration in force (number and
occupancy of the stress states, the log-likelihood and, per feature, how far the stress states' mean lies from the others',
in standard deviations over the window). `<variant>__regime_summary` counts the failures, the non-converged refits and their
warnings per refit and reports the share of refits in which the stress state differs more in the return than in the range
(a split on the sign of the return rather than on volatility).

Robustness check 6 (each instrument on its own longer minute-data history) is run for all five instruments (`long_history`
in the configuration: XAUUSD, XAGUSD, EURUSD, USDJPY, GBPUSD). XAGUSD has no longer history than the study sample, so its
check 6 run repeats the main run (a test checks that its two daily bundles are identical); the pooled test and the Holm
family of check 6 cover five instruments.

## Layout

| path | content |
|---|---|
| `config/study_config.json` | the study design as the code reads it from a file: instruments, sample, RV and cleaning rules, window, refit, burn-in, horizons, MCS and variant settings (constants such as the EWMA decay 0.94, K = 2 and the 0.5 stress threshold are in the code) |
| `pyproject.toml`, `uv.lock` | the pinned package versions |
| `src/rvf/data.py` | M1 loader, bar-level cleaning, bounce-back filter |
| `src/rvf/realized.py` | realised variance and quarticity, range estimators, daily returns, day-level rules |
| `src/rvf/regime.py` | causal HMM regime series (`hmm-regime`): `prob_stress`, h-step probabilities |
| `src/rvf/walkforward.py` | rolling window, shared refit calendar, burn-in, vintage probabilities, HMM failure rule |
| `src/rvf/models.py` | M0-M7, M2c and the robustness variants (log-HAR, HARQ) |
| `src/rvf/evaluate.py` | QLIKE, MSE, Diebold-Mariano, pooled test, Holm, MCS, Giacomini-White, effect sizes |
| `src/rvf/study.py` | variants, daily bundles, tasks and workers, evaluation tables, LaTeX and figure writers |
| `src/run_study.py`, `src/sample_definition.py`, `src/audit_data.py`, `src/mt5_export.py` | command-line scripts |
| `src/mt5/ExportM1.mq5` | the MetaTrader 5 script that `mt5_export.py` runs to export the M1 bars |
| `results/` | the data audit, the sample definition and the raw-file hashes; the full run writes its tables, figures and `RUN_INFO.md` here |
| `tests/` | unit tests |

The code is written to be reused by the later studies in the series (`regime-factor-premia`, `regime-option-hedging`).
