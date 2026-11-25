# Pre-specification: Does HMM regime information improve daily volatility forecasts?

- **Study:** 1 of 3 in the HMM regime series (repo `regime-vol-forecast`)
- **Version:** 0.4 (draft, panel decided), 2026-11-24
- **Status:** DRAFT. This becomes binding once author approves it and the approved version
  (1.0) is committed to the repo and tagged `prespec-v1.0` before any out-of-sample run.
  All design decisions are closed; what remains is Saye's approval.
- **Changes from earlier versions:** listed in §15.

---

## 1. Research question

Do causal (real-time, filtered) regime probabilities from a Gaussian hidden Markov model,
estimated in a separate first step with the open-source library `hmm-regime`, improve
out-of-sample daily forecasts of realised variance? The comparison is against:

- standard benchmarks (EWMA, GARCH(1,1), HAR-RV), and
- a jointly estimated Markov-switching HAR model (MS-HAR).

The study is a multi-asset, strictly causal extension of existing two-step HMM–HAR work
(Blake, Gandhi & Jakkula, 2025). It is not intended as a methodological breakthrough. A
positive, partial or null result are all reportable outcomes.

## 2. Hypotheses

All hypotheses are evaluated at the primary horizon h = 1 trading day, using the QLIKE
loss.

| | Hypothesis | Models | Test direction |
|---|---|---|---|
| **H1** (primary, approach A) | HAR-X has lower expected QLIKE than HAR. | M5 vs M3 | One-sided |
| **H2** (secondary, approach B) | The regime-weighted HAR has lower expected QLIKE than HAR. | M6 vs M3 | One-sided |
| **H3** (two-step vs joint estimation) | Each two-step model and MS-HAR have equal expected QLIKE. | M5 vs M7, and M6 vs M7 | Two-sided |
| **H4** (where gains arise) | If H1 or H2 shows a gain, the gain is concentrated on stress days and around regime transitions (§8.5). | M5, M6 vs M3 | Descriptive |

Notes:

- H1 is the single primary hypothesis. H2 is secondary, so that "H1 or H2" does not double
  the chance of a positive headline result.
- H3 is tested for both two-step models. The better of the two is not picked after seeing
  the results.
- **Secondary comparison, reported but not a hypothesis:** GARCH-X (M4) against the
  *calibrated* GARCH benchmark (M2c), not against raw GARCH (M2). See §5 for why.

## 3. Data

**Source.** 1-minute (M1) bars exported from the MetaQuotes-Demo server of MetaTrader 5
(details in §12.1).

- Timestamps are in server time: GMT+2 in winter, GMT+3 in summer.
- All symbols are built from **bid** prices. Spread widening therefore shows up as price
  movement. §4 and §12.5 handle this.

**Instruments planned (shortlist, fixed before the audit).** Six instruments in three asset
classes:

| Asset class | Instruments | Pre-named alternate |
|---|---|---|
| Equity indices | US500; NAS100/USTEC | GER40/DE40 |
| Precious metals | XAUUSD; XAGUSD | none |
| FX | EURUSD; USDJPY | GBPUSD |

**Eligibility rule (fixed now, applied at the audit).**

- An instrument whose M1 history starts after 2014-12-31 is excluded.
- **Panel rule.** Every shortlisted instrument that passes the rule enters the panel,
  alternates included. (Version 0.3 admitted an alternate only as the replacement of a failed
  instrument of its class. Leaving out an eligible, audited instrument would be a selection
  made by hand, so version 0.4 admits every instrument that passes.) No instrument outside
  the shortlist is added.
- The study needs at least four instruments from at least two asset classes. Otherwise it
  stops and the design is revisited before any forecast is run.

**Outcome of the rule (§12.2).** On this server no equity index has minute data before 2017,
so the equity class is dropped. It is not analysed in a supplementary tier either (decision
of 2026-11-24). The study panel is XAUUSD, XAGUSD, EURUSD, USDJPY and GBPUSD: five
instruments in two asset classes, FX (EURUSD, USDJPY, GBPUSD) and precious metals (XAUUSD,
XAGUSD). The paper's scope is therefore FX and precious metals.

**Sample.** 2008-11-10 to 2026-10-31. The start is the first server day on which all five
instruments have minute data (§12.2, §12.3).

## 4. Target variable

**Trading day.** One server-time calendar day. The server has no Sunday bars. Saturday bars
are dropped. There is no DST adjustment (§12.4).

**Sampling.** A regular 5-minute grid with previous-tick sampling: the price at a grid point
is the last M1 close at or before it.

**Primary measure.** Daily realised variance, the sum of squared 5-minute log returns
within the day:

RV_t = Σᵢ r²_{t,i}

- **Units.** Returns are in percent, so RV and all variance forecasts are in percent².
- **Session-break return.** The return from the last price of day t−1 to the first grid
  point of day t is included in RV_t as one additional squared return. On Mondays this is
  the weekend gap. RV_t is therefore a whole-day measure, and its returns add up to the
  close-to-close return that EWMA and GARCH model.
- **Rollover window.** Around the daily rollover, spreads widen and bid prices move without
  a true price change. Bars stamped 23:00–00:59 server time are excluded (§12.5). The price
  change across the excluded minutes is part of the session-break return, so nothing is
  dropped from the day's total return.
- **Short and stale days** (holidays, half-days, frozen feeds) are removed from an
  instrument's daily series (§12.6).

**Justification.** Liu, Patton & Sheppard (2015) compare over 400 realised measures across
31 assets and find that 5-minute RV is difficult to beat significantly. A 5-minute rather
than 1-minute interval limits market-microstructure noise.

**Alternatives used only as robustness checks:**

- RV excluding the session-break return
- RV without the rollover-window exclusion
- 1-minute RV
- Parkinson (1980) range estimator
- Garman–Klass (1980) range estimator

Range estimators are not used as the primary measure, for three reasons:

- they use four prices per day instead of hundreds of intraday returns, so they are much
  noisier proxies of latent variance;
- they assume continuous trading without drift;
- Parkinson ignores the overnight return.

**Daily returns.** r_t = 100 · log(C_t / C_{t−1}), where C is the close of the last M1 bar
of the server day outside the excluded window (§12.5).

## 5. Models

| # | Model | Group | Specification |
|---|---|---|---|
| M1 | EWMA | Benchmark | σ²_{t+1} = λσ²_t + (1 − λ)r²_t, with λ = 0.94 fixed (J.P. Morgan/Reuters, 1996). Initialised with the sample variance of the training window. |
| M2 | GARCH(1,1) | Benchmark | r_t = μ + ε_t, σ²_t = ω + αε²_{t−1} + βσ²_{t−1}. Normal QMLE with the `arch` package (Bollerslev, 1986). |
| M2c | Calibrated GARCH | Benchmark | RV_{t+1} = a + b σ̂²_{GARCH,t+1∣t} + e_{t+1}, estimated by OLS on the training window. |
| M3 | HAR-RV | Benchmark | RV_{t+1} = β₀ + β_d RV_t + β_w RV_t^(w) + β_m RV_t^(m) + e_{t+1}, where RV^(w) and RV^(m) are 5- and 22-day averages. OLS in levels (Corsi, 2009). |
| M4 | GARCH-X | Approach A | M2c plus c · p_t. |
| M5 | HAR-X | Approach A | M3 plus γ · p_t. |
| M6 | Regime-weighted HAR | Approach B | One HAR per regime k, estimated by weighted least squares. Forecast = Σ_k π_{k,t} · HAR_k forecast. Follows Blake et al. (2025), but strictly causal. |
| M7 | MS-HAR | Competing regime model | Two-regime Markov-switching regression of RV_{t+1} on the HAR regressors, with switching coefficients and switching variance (`statsmodels` `MarkovRegression`). Forecast = Σ_k P(s_{t+1} = k ∣ F_t) · x_t′β_k, from filtered probabilities only. |
| M0 | Pure HMM | Reference only | Σ_k P(s_{t+h} = k ∣ F_t) · σ²_k |

Here p_t is `prob_stress` at the close of day t, and π_{k,t} = P(s_{t+1} = k ∣ F_t) is the
one-step-ahead regime probability (§6).

**Why M2c exists.** GARCH forecasts the variance of close-to-close returns, while the target
is RV. The two can differ in level. M4 contains an affine recalibration (a, b) that raw
GARCH lacks, so M4 could beat M2 through recalibration alone. Comparing M4 with M2c isolates
the contribution of p_t.

**M6 details.**

- The same quantity is used in training and in forecasting: the WLS weight of observation
  (x_t, RV_{t+1}) for regime k is π_{k,t}, and the forecast mixes with π_{k,t}.
- Effective sample size of regime k in the window: ESS_k = (Σ_t π_{k,t})² / Σ_t π²_{k,t}.
  If ESS_k < 100, the regime's HAR is replaced by the pooled HAR (M3) for that refit. The
  number of such refits is reported.

**M7 details.**

- Each refit uses 10 random starting values plus the previous refit's estimate. The fit
  with the highest likelihood is kept.
- The regime with the larger error variance is labelled "stress".

**Estimation failures (M2, M2c, M4, M7).** If a refit does not converge, the previous
refit's parameters are carried forward. If there is no previous refit, the model's forecast
is replaced by the M3 forecast (M1 for M2). Failures are counted and reported per model and
asset.

**Insanity filter (all models, M0–M7).** A forecast below the minimum or above the maximum
RV of its training window is replaced by the mean RV of that window. The window is the one
used at the most recent refit. For h > 1 the bounds and the mean are taken from the h-day
sums inside that window. The number of replacements is reported per model and asset.

## 6. HMM regime series

- **Library:** `hmm-regime==0.2.0` (PyPI, `import hmm_regime`), applied to daily bars
  aggregated from the M1 data.
- **Primary settings:** K = 2 and `vol_feature="bar"`, using the per-bar log range. These are
  fixed before seeing any real-data result. The choice rests on the library's synthetic
  audit (accuracy 0.96 against 0.77 for the default `"rolling"` setting). That audit is
  synthetic and is not cited as real-data evidence.
- **Calibration:** At each refit date the model is calibrated on the training window only.
  Between refits the parameters stay fixed and the forward filter is updated daily.
- **Outputs:**
  - `prob_stress_t` = P(stress regime ∣ F_t)
  - h-step regime probabilities: P(s_{t+h} ∣ F_t) = filtered probabilities × Aʰ

  `prob_stress` is used rather than 1 − prob_regime_0, because K can exceed 2 in the
  robustness runs.
- **Whipsaw filter:** not used, because the models use probabilities, not labels.
- **Estimation failure:** if the HMM cannot be estimated at a refit, the previous refit's
  parameters are kept and filtering continues. If the first refit of an instrument fails,
  the run stops for that instrument and the failure is reported. Failures are counted per
  instrument; an instrument with failures in more than 10% of its refits is flagged in the
  paper.

**Two properties of this design, stated up front.**

- **Generated regressor.** Inside a training window, p_s is produced by the forward filter
  with parameters estimated on the whole window. Out of sample, p_t uses parameters from the
  past only. The regressions for M4–M6 are therefore trained on a slightly "cleaner"
  regressor than they are applied to. No out-of-sample information leaks, but coefficients
  may be mis-scaled. Robustness check 9 measures this.
- **Information sets differ.** The HMM sees daily bars (returns and range). MS-HAR sees
  5-minute RV. H3 therefore compares two things at once: two-step versus joint estimation,
  and range-based versus RV-based regime inference. Robustness check 8 separates them.

## 7. Estimation design

- **Burn-in:** the first 22 retained days of each instrument's sample are used only to build
  features and lags. This is the same in every run, so the main run and the robustness runs
  share their evaluation days.
- **Walk-forward:** a rolling training window of 1000 trading days after the burn-in. The
  first forecast is for the day after the first window.
- **Refits:** all models (M1–M7 and the HMM) are re-estimated every 21 trading days on one
  shared calendar per instrument.
- **Timing:** a forecast for day t+1 uses information up to the close of day t only. This is
  enforced by unit tests that perturb data after t.
- **Common evaluation sample:** within an instrument, all models are evaluated on the
  identical set of days.
- **Horizons:**
  - h = 1 is primary.
  - h = 5 and h = 22 are exploratory. Their target is cumulative variance Σ_{j=1..h} RV_{t+j}.
  - HAR-type models (M3–M7) and M2c use direct regressions on the cumulative target.
  - EWMA, GARCH and the pure HMM model (M0) use their own iterated forecasts.

## 8. Evaluation

### 8.1 Loss functions

- **Primary, QLIKE:** L(RV, f) = RV/f − log(RV/f) − 1
- **Secondary, MSE:** (RV − f)²

Both are robust to noise in the volatility proxy (Patton, 2011).

### 8.2 Pooled test (primary inference for H1)

- For each day t, the QLIKE loss differential is averaged over the instruments that have a
  forecast that day: d̄_t = (1/N_t) Σ_i d_{i,t}.
- A Diebold–Mariano test is applied to the series d̄_t, with Newey–West HAC variance and lag
  max(h − 1, 5). Averaging across instruments first makes the test valid under
  cross-sectional dependence.
- Pooling is done for QLIKE only. QLIKE is scale-free, so instruments are comparable. MSE is
  not pooled, because the most volatile instrument would dominate.
- The pooled H1 test is the single primary test, at the 5% level. The pooled H2 test is
  reported as secondary.
- **Framework.** Qu, Timmermann & Zhu (2024) study tests of equal forecasting performance in
  panels of forecasts, averaged over all time periods and units. The panel here is small
  (five instruments, about 3,500 forecast days each), so the test is the simpler
  time-series test on the cross-sectional average. Its validity rests on the number of days,
  not on the number of instruments. Its power depends on how strongly the instruments' loss
  differentials move together (§13).

### 8.3 Per-instrument tests (supporting)

- Diebold–Mariano (1995) statistic on loss differentials, with Newey–West HAC variance, lag
  max(h − 1, 5), and the small-sample correction of Harvey, Leybourne & Newbold (1997).
- Because models are estimated on a rolling window of fixed width, the test is read within
  the Giacomini–White (2006) framework, which allows nested comparisons such as HAR against
  HAR-X.
- The critique of Zhu & Timmermann (2020) is acknowledged in the paper. Conclusions do not
  rest on a single test.
- Significance level 5%. The Holm correction is applied across the five instruments,
  separately for H1, for H2, for the M4–M2c comparison, and for H3 (both comparisons
  together, ten tests).

### 8.4 Model Confidence Set

- Hansen, Lunde & Nason (2011), applied per instrument to M1–M7 and M2c with QLIKE loss.
- Confidence level 90%, range statistic.
- Stationary bootstrap with 10,000 replications and an average block length of 10.
- Implementation: `arch.bootstrap.MCS`.

### 8.5 Conditional analysis (H4)

Losses are reported separately for:

- **Stress days:** p_t > 0.5, known at t.
- **Calm days:** p_t ≤ 0.5.
- **Transition windows:** ±10 trading days around each crossing of p_t through 0.5.
- **Calendar years.**

As a formal counterpart, the Giacomini–White conditional predictive ability test is run for
M5 vs M3 and M6 vs M3 with instruments (1, p_t). It is exploratory.

### 8.6 Effect sizes and confidence intervals

For every comparison in §2 (H1, H2, H3) and for M4 against M2c, the paper reports the size
of the difference, whatever the outcome of the tests.

- **Sign convention.** The QLIKE loss differential is d = L(reference) − L(model), so
  positive values favour the model. The reference is M3 for H1 and H2 (model M5 or M6), M2c
  for the GARCH pair (model M4), and M7 for H3 (model M5 or M6).
- **Relative improvement.** The mean of d divided by the mean QLIKE of the reference, in
  percent, with a 95% confidence interval from the same HAC variance as the
  Diebold–Mariano test (the reference's mean loss is treated as fixed). It is reported
  pooled (using the cross-instrument average d̄_t of §8.2) and per instrument.
- **Purpose.** A null result is informative only if it says which improvements can be
  excluded. The interval does this. No further test is added.

## 9. What counts as a result

"Two thirds" below means 4 of the 5 instruments in the study panel (two thirds of 5,
rounded up; §12.2).

| Outcome | Condition |
|---|---|
| **Strong** | The pooled H1 test rejects at 5%, **and** the per-instrument H1 test rejects after the Holm correction in at least two thirds of the instruments, **and** in most of those instruments M5 is in the 90% MCS while HAR (M3) is excluded. |
| **Partial** | Any one of: the pooled H1 test rejects but the other "strong" conditions fail; only H2 is supported; only the GARCH pair improves (M4 vs M2c); a gain appears only in stress or transition periods. |
| **Null** | The pooled H1 test does not reject and no per-instrument H1 or H2 test rejects after the Holm correction. |

**H3** is reported in every case, as one of three outcomes: the two-step approach is as good
as joint estimation, it is better, or joint estimation is better. A directional H3
conclusion is drawn only if the levels form and the log form (robustness check 4) agree.
Otherwise H3 is reported as "form-dependent". The reason: a Gaussian MS-HAR in levels is a
weaker specification than in logs, and H3 should not be decided by that alone.

**Wording of conclusions.** The panel covers two asset classes (FX and precious metals),
and the instruments share most stress episodes, so the number of distinct regime episodes
in 2012–2026 is small (§13). A null result is therefore written as "no detectable
improvement in this panel and specification", with the intervals of §8.6, and not as a
general statement that regimes do not help volatility forecasting. A positive result is
likewise stated for FX and precious metals only. The title and the abstract name this scope.

## 10. Robustness checks (reported, not used to select the main specification)

1. Target measure: RV without the session-break return, RV without the rollover exclusion,
   1-minute RV, Parkinson, Garman–Klass.
2. HMM feature: `vol_feature="rolling"` and the 5-bar feature.
3. Number of regimes: K selected by BIC from {1, 2, 3} at each refit.
4. HAR form: log-HAR in place of levels HAR for M3, M5, M6 and M7. Forecasts are converted
   back to variance with the usual log-normal correction.
5. Training window: 750 and 1500 days.
6. Longer history: each instrument on its own full minute-data history (EURUSD, USDJPY and
   GBPUSD from 2000-01-03, XAUUSD from 2004-06-14; XAGUSD is unchanged), reported per
   instrument. The pooled test then averages over the instruments available on each day
   (§8.2). This check adds the 2008 and euro-crisis periods to the evaluation sample.
7. MSE in place of QLIKE as the loss (per instrument only).
8. **Information-matched HMM:** the HMM is fed log RV as its feature, so that it sees the
   same information as MS-HAR. Version 0.2.0 accepts a user-supplied feature matrix, so the
   check is run. States are ordered by the mean of log RV. M0 is not part of this run.
9. **Vintage probabilities:** the regressions for M4–M6 are trained on real-time vintage
   probabilities (each p_s as it was computed on day s). This needs 1000 further days of
   burn-in, so it is run on the shorter remaining sample.
10. **Stronger benchmark:** HARQ (Bollerslev, Patton & Quaedvlieg, 2016) against HARQ plus
    p_t. This checks whether the regime probability only proxies for a known nonlinearity in
    RV dynamics.

## 11. Commitments

1. The paper is written whatever the outcome.
2. Nothing in §§2–10 changes after out-of-sample results have been seen. Any change is
   reported in a separate section labelled "post-hoc", with the reason.
3. The instrument panel (§3, §12.2) is fixed at approval. No instrument is added, replaced
   or removed afterwards.
4. Code is developed and debugged only on the first training window of US500, which is not
   part of the panel (§12.8). The full run is executed once, on frozen code. The commit
   hash, package versions and run date are recorded in `results/RUN_INFO.md`.
5. The results are handed to Saye uninterpreted before any writing starts.

## 12. Decisions fixed after the data audit

These rest on `results/data_audit.md` (generated 2026-11-23). They depend on data quality
only. No forecast had been computed when they were made.

### 12.1 Source

- Server: MetaQuotes-Demo (MetaQuotes Ltd.), terminal build 6249. Server time is GMT+2 in
  winter and GMT+3 in summer. All eight exported symbols are bid-based; five are used in the
  study (§12.2).
- Server symbol names: US500, USTEC, DE40, XAUUSD, XAGUSD, EURUSD, USDJPY, GBPUSD.
- This is MetaQuotes' own demo server, not a broker's trading server. That is a limitation
  of the study (§13). **Decision (2026-11-24):** the study keeps this source. No other
  source is used.

### 12.2 Minute-data start and eligibility

- **Minute-data start** of a symbol: the first of 20 consecutive trading days that each
  have at least half the usual number of 5-minute intervals containing a bar (the audit's
  "minute data from" column).
- The eligibility rule of §3 is applied to the minute-data start, not to the first bar in
  the file. The rule refers to M1 history, and the index files hold one bar per day for
  2012–2015 and one bar per hour for 2016.

| Symbol | Minute data from | Outcome |
|---|---|---|
| US500 | 2017-03-21 | fails (after 2014-12-31) |
| USTEC | 2022-07-07 | fails |
| DE40 (alternate) | 2017-04-18 | fails; the equity class is dropped |
| XAUUSD | 2004-06-14 | eligible |
| XAGUSD | 2008-11-10 | eligible |
| EURUSD | 2000-01-03 | eligible |
| USDJPY | 2000-01-03 | eligible |
| GBPUSD (alternate) | 2000-01-03 | eligible; enters the panel under the panel rule of §3 |

- **Study panel:** XAUUSD, XAGUSD, EURUSD, USDJPY, GBPUSD. Five instruments in two classes
  (FX: EURUSD, USDJPY, GBPUSD; precious metals: XAUUSD, XAGUSD). The minimum of §3 (four
  instruments, two classes) is met with one instrument to spare.
- **Equity indices are not analysed**, not even as a supplementary tier (decision of
  2026-11-24). Their minute data start in 2017 or 2022, they contain frozen-feed and outage
  periods, and a separate shorter-sample analysis would need its own rules and would not be
  comparable with the panel. US500 is used only for development (§12.8).
- The index data have two further defects on this server: a frozen feed in October–December
  2024 (46 to 58 stale days per index) and no bars at all from mid-July to 8 September 2025.

### 12.3 Sample

- Start: 2008-11-10, the latest minute-data start in the panel (XAGUSD). End: 2026-10-31.
  Bars after the end date are dropped.
- With the 22-day burn-in and the 1000-day window (§7), the first forecast falls in late
  2012.

### 12.4 Trading day and DST

- A trading day is a server calendar day. There are no Sunday bars. Saturday bars are
  dropped.
- No DST adjustment is made. The server follows the European clock change. For one to three
  weeks in March and one week in October or November, US market events therefore fall one
  hour earlier in server time. The excluded window in §12.5 is wide enough for both cases.

### 12.5 Rollover window (replaces the 3× median-spread rule of version 0.2)

- **Why the old rule is replaced.** Applied to the whole sample, it flags no minute in any
  panel instrument. The reason is the `<SPREAD>` column: it is a constant placeholder before
  2009 (zero for the metals until 2010) and its level drifts from year to year, so a
  whole-sample median says little. Applied year by year, the same rule flags the first
  server hour in the years with informative spreads (2018, 2020, 2023–2026) and parts of
  the last hour.
- **Rule.** Bars stamped 23:00–00:59 server time are excluded, for all five instruments and
  all years. This covers the New York 17:00 rollover hour in both DST configurations.
- Excluded bars are not used for any price: not for the 5-minute grid, not for the daily
  close (the last bar before 23:00) and not for the daily high and low (HMM feature,
  Parkinson, Garman–Klass).
- The price change across the window is part of the session-break return. RV therefore
  covers 22 trading hours plus one squared return for the rest of the day.

### 12.6 Day-level rules

- **Nominal day length:** 264 five-minute intervals (01:00–22:59). This is a fixed number,
  so no day's status depends on later data.
- **Short day:** fewer than 132 intervals contain at least one bar. Intervals with a bar are
  counted, not the span from first to last bar, so a day with a long hole inside also fails.
- **Stale day:** more than 50% of the day's one-minute returns are exactly zero.
- Short and stale days are removed from the instrument's daily series. A removed day's price
  change goes into the session-break return of the next retained day.
- **Anchor rule:** if the previous retained day is more than 5 calendar days earlier, the
  day is used only as a price anchor. It gets no RV and no return, and it is not part of the
  daily series. This keeps multi-day returns out of RV.
- The number of days removed by each rule is written per instrument to
  `results/sample_definition.md` before the full run.

### 12.7 Bar-level cleaning

- Duplicate timestamps: keep the last row. Rows with a non-positive price or with
  `high < low`: drop. None of these occur in the files; the rules are stated for
  completeness.
- **Bounce-back filter.** A bar is removed if its one-minute return exceeds 20 robust
  standard deviations in absolute value and at least 75% of it is reversed in the next
  minute. The robust standard deviation is 1.4826 times the median absolute deviation of
  one-minute returns over the previous 20 trading days, so only past data are used.
- Large moves that are not reversed are kept. They are treated as real price jumps, and RV
  includes jumps.
- The number of bars removed is reported per instrument.

### 12.8 Development window

Code was developed on the first 1000 minute-data days of US500 (2017-03-22 to 2021-02-04).
US500 is no longer in the study, so no data of a panel instrument were used in development.

## 13. Known limitations (to be stated in the paper)

- The data come from MetaQuotes' demo server, not from a broker or an exchange. They are
  built from bid prices and stamped in server time.
- No equity index is covered. The results apply to FX and precious metals only, and the
  title and abstract say so.
- The panel is small: five instruments in two asset classes. Their volatilities move
  together and they share most stress episodes, so per-instrument results are not
  independent evidence. The number of distinct regime episodes in 2012–2026 is small, which
  limits the power to detect modest gains from regime information. The confidence intervals
  of §8.6 show which improvements can be excluded.
- Minute data are sparser in the early years (2008–2012 for silver). Those years are used
  for training, not for evaluation.
- No implied-volatility benchmark, because it is not available for these instruments.
- Evaluation is statistical only. Economic value (for example Value-at-Risk backtests) is
  left for later work.
- The generated-regressor and information-set properties described in §6.
- K is fixed at 2 in the primary specification.

## 14. References

- Andersen, T. G. & Bollerslev, T. (1998). Answering the skeptics: Yes, standard volatility
  models do provide accurate forecasts. *International Economic Review*, 39(4), 885–905.
- Ardia, D., Bluteau, K., Boudt, K. & Catania, L. (2018). Forecasting risk with
  Markov-switching GARCH models: A large-scale performance study. *International Journal of
  Forecasting*, 34(4), 733–747.
- Blake, A. C., Gandhi, N. A. & Jakkula, A. R. (2025). Improving S&P 500 volatility
  forecasting through regime-switching methods. arXiv:2510.03236.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity.
  *Journal of Econometrics*, 31(3), 307–327.
- Bollerslev, T., Patton, A. J. & Quaedvlieg, R. (2016). Exploiting the errors: A simple
  approach for improved volatility forecasting. *Journal of Econometrics*, 192(1), 1–18.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility.
  *Journal of Financial Econometrics*, 7(2), 174–196.
- Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business
  & Economic Statistics*, 13(3), 253–263.
- Diebold, F. X. (2015). Comparing predictive accuracy, twenty years later: A personal
  perspective on the use and abuse of Diebold–Mariano tests. *Journal of Business & Economic
  Statistics*, 33(1).
- Ding, Y., Kambouroudis, D. & McMillan, D. G. (2025). Forecasting realised volatility using
  regime-switching models. *International Review of Economics & Finance*, 101, 104171.
- Garman, M. B. & Klass, M. J. (1980). On the estimation of security price volatilities from
  historical data. *Journal of Business*, 53(1).
- Giacomini, R. & White, H. (2006). Tests of conditional predictive ability. *Econometrica*,
  74(6), 1545–1578.
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time
  series and the business cycle. *Econometrica*, 57(2).
- Hamilton, J. D. & Susmel, R. (1994). Autoregressive conditional heteroskedasticity and
  changes in regime. *Journal of Econometrics*, 64(1–2), 307–333.
- Hansen, P. R. & Lunde, A. (2005). A forecast comparison of volatility models: Does anything
  beat a GARCH(1,1)? *Journal of Applied Econometrics*, 20(7), 873–889.
- Hansen, P. R., Lunde, A. & Nason, J. M. (2011). The model confidence set. *Econometrica*,
  79(2), 453–497.
- Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of prediction mean
  squared errors. *International Journal of Forecasting*, 13(2), 281–291.
- J.P. Morgan/Reuters (1996). *RiskMetrics — Technical Document*, 4th edition.
- Kılıç, R. (2026). Linear and nonlinear econometric models versus machine-learning models:
  Evidence from realized-volatility forecasting. FEDS Working Paper 2025-061r1, Federal
  Reserve Board.
- Klaassen, F. (2002). Improving GARCH volatility forecasts with regime-switching GARCH.
  *Empirical Economics*, 27(2), 363–394.
- Liu, L. Y., Patton, A. J. & Sheppard, K. (2015). Does anything beat 5-minute RV? A
  comparison of realized measures across multiple asset classes. *Journal of Econometrics*,
  187(1), 293–311.
- Ma, F., Wahab, M. I. M., Huang, D. & Xu, W. (2017). Forecasting the realized volatility of
  the oil futures market: A regime switching approach. *Energy Economics*, 67, 136–145.
- Marcucci, J. (2005). Forecasting stock market volatility with regime-switching GARCH
  models. *Studies in Nonlinear Dynamics & Econometrics*, 9(4).
- Nystrup, P., Madsen, H. & Lindström, E. (2017). Long memory of financial time series and
  hidden Markov models with time-varying parameters. *Journal of Forecasting*, 36(8),
  989–1002.
- Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of
  return. *Journal of Business*, 53(1).
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies.
  *Journal of Econometrics*, 160(1), 246–256.
- Qu, R., Timmermann, A. & Zhu, Y. (2024). Comparing forecasting performance with panel
  data. *International Journal of Forecasting*, 40(3), 918–941.
- Zhu, Y. & Timmermann, A. (2020). Can two forecasts have the same conditional expected
  accuracy? arXiv:2006.03238.

**To verify before the paper is submitted:**

- the page ranges of Hamilton (1989), Garman & Klass (1980) and Parkinson (1980), and the
  issue details of Diebold (2015);
- a published precedent to cite for the insanity filter in §5 and for the bounce-back filter
  in §12.7. The rules themselves are fixed here; only the citations are open;
- how the pooled test of §8.2 relates to the panel tests of Qu, Timmermann & Zhu (2024).
  Only the abstract has been read so far; the paper has to be read before the relationship
  is described;
- the bibliographic details of Andersen & Bollerslev (1998), Nystrup, Madsen & Lindström
  (2017) and Kılıç (2026), which were not confirmed against the publishers' pages.

## 15. Changes from earlier versions

### From 0.1 to 0.2

| # | Change | Reason |
|---|---|---|
| 1 | Added calibrated GARCH (M2c); M4 is now compared with M2c. | M4 vs raw GARCH mixed the regime effect with a level recalibration. |
| 2 | H3 is tested for both two-step models. | Picking the better one after the fact biases the test. |
| 3 | H1 is the single primary hypothesis; a pooled QLIKE test is the primary inference. | "H1 or H2" doubled the chance of a headline result; six separate tests have low power. |
| 4 | M6 uses the same probabilities (one-step-ahead) in training and forecasting, with an effective-sample-size fallback. | Version 0.1 trained on one quantity and forecast with another; the stress-regime HAR can rest on very few days. |
| 5 | Rollover-window exclusion, short-day rule, previous-tick sampling, units, Sunday bars. | MT5 bars are bid-based; spread spikes and half-days would distort RV. |
| 6 | Estimation-failure rules and an insanity filter for all models. | Version 0.1 only handled negative forecasts, and only for linear models. |
| 7 | H3 conclusion requires agreement between levels and log forms. | Gaussian MS-HAR in levels is a weak specification. |
| 8 | Instrument eligibility rule with pre-named alternates. | Otherwise the instrument set could be chosen after seeing the data. |
| 9 | Robustness checks 8–10 and the GW conditional test for H4. | To address the information-set confound, the generated-regressor issue and a stronger benchmark. |
| 10 | HLN correction; common evaluation sample; limitations section. | Completeness. |

### From 0.2 to 0.3 (after the data audit, before any forecast)

| # | Change | Reason |
|---|---|---|
| 1 | Eligibility is judged on the minute-data start. The equity class is dropped; the panel is four instruments. | The index files hold daily and hourly bars before 2017, which the first-bar reading of the rule missed. |
| 2 | The 3× median-spread rule is replaced by a fixed excluded window, 23:00–00:59 server time. | The spread column is a placeholder before 2009 and drifts afterwards, so the old rule flagged nothing. |
| 3 | Daily close, high and low are taken from bars outside the excluded window. | Otherwise the bid dip at rollover stays in the close and in the range. |
| 4 | Short-day rule with a fixed nominal length; stale-day rule; anchor rule. | The audit found frozen feeds and missing days; a sample median would make early days depend on later data. |
| 5 | Bounce-back filter for bad ticks. | A small share of extreme one-minute returns reverse at once. |
| 6 | HMM estimation-failure rule; 22-day burn-in shared by all runs. | Version 0.2 had no rule; feature settings otherwise shift the evaluation days. |
| 7 | Robustness check 6 redefined; check 8 confirmed as feasible. | US500 has no early minute data; the library accepts a custom feature matrix. |
| 8 | "Two thirds" now means 3 of 4 instruments. | Follows from change 1. |

### From 0.3 to 0.4 (panel decision, before any forecast)

| # | Change | Reason |
|---|---|---|
| 1 | GBPUSD enters the panel, which now has five instruments. The panel rule admits every shortlisted instrument that passes the eligibility rule, alternates included. | Leaving out an eligible, audited instrument would be a selection made by hand. |
| 2 | Equity indices are excluded and get no supplementary analysis. | Short and defective data; a separate analysis would need its own rules and would not be comparable. |
| 3 | The data source is kept (MetaQuotes-Demo); the open decision of §12.1 is closed. | Decision of 2026-11-24. No other source is used. |
| 4 | "Two thirds" now means 4 of 5 instruments; the Holm correction runs across five instruments. | Follows from change 1. |
| 5 | Effect sizes with 95% confidence intervals are reported for every comparison (§8.6). | A null result is informative only if it says which improvements can be excluded. |
| 6 | Rule for the wording of conclusions (§9) and a longer limitations section (§13). | Two asset classes and few distinct regime episodes limit what the study can claim. |
| 7 | The pooled test is placed in the framework of Qu, Timmermann & Zhu (2024) (§8.2). | Gives the primary inference a published reference. |
| 8 | Robustness check 6 also covers GBPUSD. | Follows from change 1. |
| 9 | New commitment: the panel is fixed at approval (§11). | Prevents panel changes after results are seen. |

## 16. Implementation notes confirmed

These points were left open by version 0.2 and settled as implemented in the code.

**Realised variance**

- An M1 bar stamped HH:MM closes at HH:MM+1. A grid point takes the last close up to that
  instant.
- Session-break return = close-to-close return minus the sum of the grid returns. Overnight
  movement and movement inside the excluded window are one return, squared once.
- Realised quarticity uses the intraday grid returns only.

**Models**

- HAR weekly and monthly averages may reach back before the start of the training window.
  They use past data only.
- M2c and M4, for h > 1: the h-day RV sum is regressed on the summed GARCH forecast. If
  GARCH fails, the previous GARCH parameters and regression coefficients are kept together.
- M6 uses the one-step-ahead regime probabilities as weights at every horizon.
- M7: the target and regressors are divided by their window mean for numerical stability;
  random starts are drawn around the default starting values; for h > 1 the regime
  probabilities are propagated h steps with the transition matrix.
- M0: σ²_k is the state's return variance from the HMM covariance, in percent²; for h > 1
  the forecasts for steps 1 to h are summed.

**Tests**

- The pooled Diebold–Mariano test uses normal critical values. The per-instrument tests use
  the HLN correction with Student-t critical values.
- A transition window is the 10 forecasts on each side of the first forecast at which
  p_t crosses 0.5.

**Robustness runs**

- Log-HAR: weekly and monthly terms are averages of daily log RV. The log-normal correction
  is exp(forecast + s²/2), with s² the residual variance of the training window (per regime
  for M6 and M7). If M7 fails in logs, it falls back to M3 in logs.
- HARQ: the daily term is RV_t · (b_d + b_q · √RQ_t), without demeaning RQ, with the same
  regressors at every horizon.
- Vintage probabilities (check 9): K is fixed, and forecasts start at the first refit after
  about 2000 days.
- Target variants (check 1) keep the main run's days, returns and HMM inputs; only the
  target changes.
- In check 3, K = 1 is a candidate; `prob_stress` is then zero.
