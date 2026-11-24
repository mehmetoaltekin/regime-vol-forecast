# Handoff: Study 2 — Data processing and code

Version 0.3, 2026-11-24. Aligned with `PRESPEC.md` version 0.4.

You handle **data processing and coding only** for an SSRN study by Saye
(business administration student, Marmara University). The research design, the
pre-specification (`PRESPEC.md`) and the paper are worked out in a separate chat.
Do not write or change them; implement what they say.

Talk to Saye in **Turkish**. Write all code, comments and docs in **English**.

`PRESPEC.md` is the authority on every definition. Where this file and `PRESPEC.md`
disagree, `PRESPEC.md` wins; tell Saye about the disagreement.

## The study, in one line

Do causal (real-time, filtered) HMM regime probabilities improve out-of-sample daily
volatility forecasts of standard models (EWMA, GARCH(1,1), HAR)?

## Status

| Task | State |
|---|---|
| 1. Data export and audit | Done. Report: `results/data_audit.md`. |
| 2. Infrastructure | Built. Task 2b below brings it in line with PRESPEC 0.4. |
| 3. Full run | Not started. Waits for the approved PRESPEC (hard rule 1). |

## Hard rules

1. **No experiment on the out-of-sample period runs before the approved `PRESPEC.md`
   (version 1.0) is in the repo and tagged `prespec-v1.0`.** Saye brings the approved file
   from the design chat. Task 3 starts only when Saye tells you to. Until then, only the
   data pipeline, the infrastructure and the unit tests are built.
2. Develop and debug only on the first training window of US500 (2017-03-22 to
   2021-02-04; PRESPEC §12.8). US500 is not in the study panel, so no panel data are used
   in development. The full run happens once, on frozen code. Record the commit hash in
   `results/RUN_INFO.md`.
3. **No look-ahead.** A forecast made at the close of day t may use information up
   to t only. Prove this with unit tests (e.g. change data after t; the forecast must
   not change).
4. Don't interpret results or write paper text. Produce numbers, tables and figures
   from scripts, never by hand.
5. If something in `PRESPEC.md` cannot be implemented as written, stop and tell
   Saye. Don't pick an alternative yourself.

## Repo and library

- Repo: `regime-vol-forecast` (GitHub owner `mehmetoaltekin`).
- Pin `hmm-regime==0.2.0` (PyPI; `import hmm_regime`). Source and `AUDIT.md` are in
  `mehmetoaltekin/hmm-regime-detection`. Use `prob_stress`, not
  `1 − prob_regime_0`, because K can be 3.
- Layout: `data/raw/`, `data/clean/`, `src/`, `tests/`, `results/`.
  `data/` stays out of git; store SHA-256 hashes of the raw files instead.

## Data

- Source: Saye's MT5 terminal, MetaQuotes-Demo server (build 6249), **M1 bars**, bid-based,
  stamped in server time (GMT+2 in winter, GMT+3 in summer). The MT5 CSV export is
  tab-separated with columns `<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL>
  <SPREAD>`.
- Exported and audited, in `data/raw/`: US500, USTEC, DE40, XAUUSD, XAGUSD, EURUSD,
  USDJPY, GBPUSD.
- **Study panel, five instruments:** XAUUSD, XAGUSD, EURUSD, USDJPY, GBPUSD
  (PRESPEC §3, §12.2). Sample 2008-11-10 to 2026-10-31.
- US500, USTEC and DE40 are **not part of the study** and are not analysed anywhere,
  supplementary analyses included. The raw files stay in `data/raw/`. Only the first 1000
  minute-data days of US500 are used, for development.
- The `<SPREAD>` column is a placeholder in the early years. Do not use it for any rule.

## Task 1 — Data export and audit (done)

`src/audit_data.py` and `results/data_audit.md` stay as they are. They document the data
for the paper.

## Task 2b — Bring the infrastructure in line with PRESPEC 0.4 → report back

Implement the following. Give each item a unit test. Keep the existing tests green.

1. **Panel and sample** (PRESPEC §3, §12.2, §12.3). Configure the five panel instruments
   and the sample 2008-11-10 to 2026-10-31. Bars after the end date are dropped. The equity
   indices are not part of any study run.
2. **Rollover window** (§12.5). Exclude bars stamped 23:00–00:59 server time, for all
   instruments and all years. Excluded bars are not used for the 5-minute grid, for the
   daily close (the last bar before 23:00) or for the daily high and low. The price change
   across the window goes into the session-break return. This replaces the old 3×
   median-spread rule; the study configuration no longer uses that rule.
3. **Day-level rules** (§12.6).
   - Nominal day: 264 five-minute intervals (01:00–22:59).
   - Short day: fewer than 132 intervals contain at least one bar.
   - Stale day: more than 50% of the day's one-minute returns are exactly zero.
   - A removed day passes its price change to the session-break return of the next
     retained day.
   - Anchor rule: if the previous retained day is more than 5 calendar days earlier, the
     day is used only as a price anchor. It gets no RV and no return and is not part of the
     daily series.
4. **Bar-level cleaning** (§12.7).
   - Duplicate timestamps: keep the last row. Drop non-positive prices and `high < low`.
   - Bounce-back filter: remove a bar if its one-minute return exceeds 20 robust standard
     deviations in absolute value and at least 75% of it is reversed in the next minute.
     The robust standard deviation is 1.4826 × the median absolute deviation of one-minute
     returns over the previous 20 trading days (past data only).
   - Large moves that are not reversed stay in the data.
5. **`results/sample_definition.md`.** Written by a script. For each instrument: first and
   last retained day, number of retained days, days removed by each rule in item 3, bars
   removed by each rule in item 4. Produce it before any full run.
6. **Burn-in** (§7). The first 22 retained days of each instrument's sample are used only
   to build features and lags, identically in every run, so that the main run and the
   robustness runs share their evaluation days.
7. **HMM failure rule** (§6). If the HMM cannot be estimated at a refit, keep the previous
   refit's parameters and continue filtering. If the first refit of an instrument fails,
   stop that instrument and report it. Count failures per instrument; flag an instrument
   with failures in more than 10% of its refits.
8. **Implementation notes** (§16). Follow them exactly. Where the code differs, change the
   code.
9. **Effect sizes** (§8.6). For H1, H2, H3 and M4 vs M2c: the QLIKE differential
   d = L(reference) − L(model); the mean of d divided by the reference's mean QLIKE, in
   percent; a 95% confidence interval from the same HAC variance as the Diebold–Mariano
   test. Compute it pooled (on the cross-instrument average d̄_t) and per instrument.
10. **Tests.** Cover each new rule: rollover window, day rules, anchor rule, bounce-back
    filter, burn-in, HMM failure path, effect sizes. Keep the no-look-ahead tests for every
    model and for the regime series.
11. **README.** Keep `README.md` current with the exact commands that go from `data/raw/`
    to `results/`, and pin the package versions.

When done, **stop and report to Saye** (in Turkish):

- a summary of `results/sample_definition.md`
- the unit-test results
- any place where `PRESPEC.md` could not be implemented as written

Do not start Task 3.

## Task 3 — Full run (only after the approved `PRESPEC.md` is in the repo and tagged)

Saye tells you when to start. Implement exactly what PRESPEC says and run it once on
frozen code. Then:

- write the outputs to `results/` (tables as CSV and LaTeX, figures as PDF), including
  descriptive statistics per instrument and the correlation matrix of daily log RV across
  the five instruments
- record the commit hash, the package versions and the run date in `RUN_INFO.md`

Run the robustness checks PRESPEC §10 lists. Check 8 is feasible: `hmm-regime` 0.2.0
accepts a user-supplied feature matrix. Do not modify the library.

**Stop and hand the results to Saye** without interpreting them.

## Context

This is study 1 of 3. The later studies (`regime-factor-premia`, `regime-option-hedging`)
will reuse the regime-series and walk-forward infrastructure, so keep it reusable and
documented.
