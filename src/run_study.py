"""
Run the study.

    python src/run_study.py --dev                                   # development: US500, first training window only
    python src/run_study.py --full --config FILE.json [--workers N]  # the one full run, on frozen code
    python src/run_study.py --full --config FILE.json --resume       # continue the same run after an interruption

The full run refuses to start unless
  - PRESPEC.md is approved: its version is 1.0 or later, its status line no longer says DRAFT, the configuration
    file names the same version, and the repo has the tag prespec-v1.0 on a commit whose PRESPEC.md equals the
    current one,
  - the working tree has no uncommitted change in tracked files (the commit hash goes into results/RUN_INFO.md),
  - the SHA-256 of every raw data file equals the one stored in results/raw_data_sha256.tsv,
  - neither results/RUN_INFO.md nor results/RUN_STARTED.lock exists: the run happens once,
  - results/forecasts, results/tables and results/figures are empty (no result of earlier code can be mixed in).

The run has three stages. Stage 0 prepares the daily rows of every instrument (data/clean/); a data problem stops the
run here, before the lock is written. Stage 1 runs one walk-forward task per (variant, instrument) in worker processes;
every finished task is stored with a completion marker that carries the run key (a hash of the commit, the configuration,
PRESPEC.md and the daily rows). Stage 2 evaluates the stored results of this run and writes tables, figures and
RUN_INFO.md. The result does not depend on the number of workers.

If the run is interrupted (machine asleep, process killed), `--resume` continues it: it needs the lock, the same commit,
the same configuration, the same PRESPEC.md and the same daily rows, and it skips the tasks that are finished and carry the
run key. If the code has to change, the finished tasks are void: move results/forecasts, results/tables and results/figures
away, delete the lock and start again.
Development mode writes to results/dev/ only, never touches data after the first training window and writes
no RUN_INFO.md.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rvf import study as S  # noqa: E402
from rvf.data import RAW  # noqa: E402

RESULTS = ROOT / "results"
CONFIG = ROOT / "config" / "study_config.json"
TAG = "prespec-v1.0"
MINIMUM_VERSION = (1, 0)
PACKAGES = ["numpy", "pandas", "scipy", "statsmodels", "arch", "hmm-regime", "hmmlearn", "matplotlib", "pyarrow"]


def prespec_status(path: Path) -> tuple[str, str]:
    """(version, status line) from the header of PRESPEC.md."""
    text = path.read_text()
    v = re.search(r"^- \*\*Version:\*\*\s*([0-9]+(?:\.[0-9]+)*)", text, re.M)
    s = re.search(r"^- \*\*Status:\*\*\s*(.*)$", text, re.M)
    if not v or not s:
        raise ValueError("PRESPEC.md has no Version and Status lines in its header")
    return v.group(1), s.group(1)


def version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def check_prespec(path: Path, config_version: str, minimum: tuple[int, ...] = (0, 3)) -> str:
    if not path.exists():
        raise SystemExit("REFUSED: no PRESPEC.md")
    version, status = prespec_status(path)
    if version_tuple(version) < minimum:
        raise SystemExit(f"REFUSED: PRESPEC.md is version {version}; the full run needs "
                         f"{'.'.join(map(str, minimum))} or later")
    if "DRAFT" in status.upper():
        raise SystemExit(f"REFUSED: PRESPEC.md still says DRAFT ({status})")
    if config_version != version:
        raise SystemExit(f"REFUSED: the configuration is for PRESPEC {config_version}, PRESPEC.md is {version}")
    return version


def check_tag(tag: str = TAG) -> None:
    """The approved PRESPEC.md must be committed and tagged, and unchanged since the tag."""
    if subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"], cwd=ROOT, capture_output=True).returncode:
        raise SystemExit(f"REFUSED: the repo has no tag {tag}")
    if subprocess.run(["git", "diff", "--quiet", tag, "HEAD", "--", "PRESPEC.md"], cwd=ROOT).returncode:
        raise SystemExit(f"REFUSED: PRESPEC.md at HEAD differs from PRESPEC.md at tag {tag}")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.rstrip("\n")


def git_state() -> tuple[str, list[str]]:
    dirty = [l for l in git("status", "--porcelain", "--untracked-files=no").splitlines() if l.strip()]
    return git("rev-parse", "HEAD"), dirty


def sha256(path: Path) -> str:
    return S.sha256_file(path)


def verify_raw_hashes(symbols: list[str], raw_dir: Path = RAW, table: Path = RESULTS / "raw_data_sha256.tsv") -> dict[str, str]:
    stored = dict(l.split("\t")[::2] for l in table.read_text().splitlines()[1:] if l.strip())
    used = {}
    for s in symbols:
        files = sorted(raw_dir.glob(f"{s}_M1_*.csv"))
        if len(files) != 1:
            raise SystemExit(f"REFUSED: expected one raw file for {s}, found {len(files)}")
        h = sha256(files[0])
        if stored.get(files[0].name) != h:
            raise SystemExit(f"REFUSED: {files[0].name} differs from the hash stored in {table.name}")
        used[files[0].name] = h
    return used


# PRESPEC 12.8: the first training window of US500, 2017-03-22 to 2021-02-04. The minute data start on 2017-03-21, a day
# without a return, so the first retained day is 2017-03-22.
DEV_START, DEV_END = "2017-03-21", "2021-02-04"


def dev_config() -> S.StudyConfig:
    """
    Development set-up: US500 between DEV_START and DEV_END with the same rules as the panel (taken from the study
    configuration), a smaller window so that forecasts exist inside the window. US500 is not in the study panel.
    """
    rules = S.StudyConfig.from_json(CONFIG).instruments[0]
    spec = S.InstrumentSpec("US500", DEV_START, DEV_END, dict(rules.rv), "development only", dict(rules.clean))
    return S.StudyConfig("dev", [spec], window=500, refit_every=100, horizons=(1, 5), ms_starts=3, mcs_reps=300,
                         variants=("primary",))             # built directly: an index would fail the panel validation


def run_dev() -> None:
    cfg = dev_config()
    out = RESULTS / "dev"
    daily = {s.symbol: S.prepare_daily(s, "primary") for s in cfg.instruments}
    for s, d in daily.items():
        assert d.index[-1] <= pd.Timestamp(DEV_END)
        print(f"{s}: {len(d)} retained days {d.index[0].date()} .. {d.index[-1].date()} (development window only)")
    res = S.run_variant("primary", cfg, daily)
    tables = S.evaluate_run(res, cfg)
    S.write_tables(tables, out / "tables", "dev_primary")
    S.make_figures(res, out / "figures", "dev_primary")
    print(f"development outputs in {out.relative_to(ROOT)} (not results; nothing to interpret)")


def make_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}"
        print(line, flush=True)
        with open(path, "a") as f:
            f.write(line + "\n")
    return log


def leftovers(*dirs: Path) -> list[Path]:
    """The files that an earlier run left in these folders (hidden files such as .gitkeep do not count)."""
    return sorted(p for d in dirs if d.exists() for p in d.rglob("*") if p.is_file() and not p.name.startswith("."))


def default_workers(n_tasks: int) -> int:
    return max(1, min((os.cpu_count() or 2) - 2, n_tasks))


def read_lock(lock: Path) -> dict:
    """The contents of RUN_STARTED.lock; a lock that is not the JSON object of a started run (cut short, empty, a list ...) is refused."""
    try:
        prior = json.loads(lock.read_text())
        datetime.fromisoformat(prior["started"])
        if not isinstance(prior["bundle_hashes"], dict):
            raise TypeError("bundle_hashes is not an object")
        return prior
    except (ValueError, KeyError, TypeError, OSError):
        raise SystemExit("REFUSED: results/RUN_STARTED.lock cannot be read")


def run_full(config_path: Path, workers: int | None = None, resume: bool = False, raw_dir: Path = RAW,
             hash_table: Path | None = None) -> None:
    cfg = S.StudyConfig.from_json(config_path)
    version = check_prespec(ROOT / "PRESPEC.md", cfg.prespec_version, MINIMUM_VERSION)
    check_tag()
    lock, info = RESULTS / "RUN_STARTED.lock", RESULTS / "RUN_INFO.md"
    clean_dir, out_dir = ROOT / "data" / "clean", RESULTS / "forecasts"
    if info.exists() or (lock.exists() and not resume):
        raise SystemExit("REFUSED: the full run has been started before (results/RUN_INFO.md or RUN_STARTED.lock exists; "
                         "--resume continues an interrupted run)")
    if resume and not lock.exists():
        raise SystemExit("REFUSED: nothing to resume (no results/RUN_STARTED.lock)")
    if not resume:                                          # results of earlier code must never be mixed into this run
        old = leftovers(out_dir, RESULTS / "tables", RESULTS / "figures")
        if old:
            shown = ", ".join(f"{p.parent.name}/{p.name}" for p in old[:3]) + (f" and {len(old) - 3} more" if len(old) > 3 else "")
            raise SystemExit(f"REFUSED: earlier results are in the way ({shown}). A new run starts from empty results/forecasts, "
                             "results/tables and results/figures; move them away first")
    commit, dirty = git_state()
    if dirty:
        raise SystemExit("REFUSED: uncommitted changes in tracked files:\n  " + "\n  ".join(dirty))
    symbols = sorted({s.symbol for s in cfg.instruments})
    hashes = verify_raw_hashes(symbols, raw_dir, hash_table or RESULTS / "raw_data_sha256.tsv")
    config_sha, prespec_sha = sha256(config_path), sha256(ROOT / "PRESPEC.md")
    prior = {}
    if resume:
        prior = read_lock(lock)
        for key, now, label in (("commit", commit, "commit"), ("config_sha256", config_sha, "configuration"),
                                ("prespec_sha256", prespec_sha, "PRESPEC.md")):
            if prior.get(key) != now:
                raise SystemExit(f"REFUSED: the {label} is not the one the run was started with; a changed code base voids "
                                 "the finished tasks, so start again from an empty results/forecasts/ and no lock")
    log = make_logger(RESULTS / "run_progress.log")
    tasks = [(v, s) for v in cfg.variants for s in S.variant_symbols(cfg, v)]
    workers = workers if workers is not None else default_workers(len(tasks))
    started = datetime.fromisoformat(prior["started"]) if prior else datetime.now(timezone.utc)
    t0 = time.time()

    # stage 0: daily rows; a data problem stops the run here, before the lock exists
    stored = prior.get("bundle_hashes", {})
    have = {n: sha256(clean_dir / n) for n in stored if (clean_dir / n).exists()}
    todo = [s for s in symbols if not all(n in have and have[n] == h for n, h in stored.items() if n.split("__")[1] == s)] \
        if (resume and stored) else symbols
    log(f"stage 0: daily rows of {', '.join(todo) if todo else 'no instrument (all present)'}")
    new_hashes, failures = S.prepare_all(cfg, config_path, clean_dir, raw_dir, workers, log, only=todo) if todo else ({}, [])
    if failures:
        state = "the lock and the finished tasks are unchanged" if resume else "no lock written, nothing to undo"
        raise SystemExit(f"STOPPED in stage 0 ({state}): " + "; ".join(f"{j[1]}: {e}" for j, e in failures))
    changed = sorted(n for n, h in new_hashes.items() if n in stored and stored[n] != h)
    if changed:                                             # the finished tasks were made from other daily rows
        raise SystemExit(f"REFUSED: the daily rows {', '.join(changed)} are not the ones the run was started with; the finished "
                         "tasks are void, so start again from empty results folders and no lock")
    bundle_hashes = {**stored, **new_hashes}
    stamp = S.run_stamp(commit, config_sha, prespec_sha, bundle_hashes)
    if not resume:
        S.write_atomic(lock, lambda p: p.write_text(json.dumps(
            {"started": started.isoformat(), "commit": commit, "config_sha256": config_sha, "prespec_sha256": prespec_sha,
             "bundle_hashes": bundle_hashes, "run_key": stamp["run_key"]}, indent=1)))

    # stage 1: one walk-forward task per (variant, instrument)
    log(f"stage 1: {len(tasks)} tasks, {workers} worker(s)")
    failures = S.execute_tasks(tasks, config_path, clean_dir, out_dir, workers, log, stamp=stamp)
    if failures:
        raise SystemExit(f"STOPPED: {len(failures)} task(s) failed ({'; '.join(f'{j[1]}/{j[2]}: {e}' for j, e in failures[:5])}). "
                         "If the cause is outside the code (machine, memory), run again with --resume; the lock stays.")

    # stage 2: evaluation of the stored results
    log("stage 2: evaluation")
    summary = S.evaluate_all(cfg, out_dir, RESULTS / "tables", RESULTS / "figures", log, clean_dir=clean_dir, run_key=stamp["run_key"])
    write_run_info(cfg, version, commit, hashes, started, time.time() - t0, summary, config_path,
                   extra={"workers": workers, "bundle_hashes": bundle_hashes, "resumed": resume, "run_key": stamp["run_key"]})
    log("finished")


def write_run_info(cfg, version, commit, hashes, started, seconds, summary, config_path, extra: dict | None = None) -> None:
    versions = {p: md.version(p) for p in PACKAGES}
    extra = extra or {}
    lines = ["# RUN_INFO", "",
             f"- Run started (UTC): {started:%Y-%m-%d %H:%M:%S}",
             f"- Wall time of the last session: {seconds / 60:.1f} minutes"
             + (f" ({extra['workers']} worker(s); the run was resumed)" if extra.get("resumed") else
                f" ({extra['workers']} worker(s))" if "workers" in extra else ""),
             f"- Git commit of the frozen code: `{commit}`",
             *([f"- Run key (commit, configuration, PRESPEC.md and daily rows; every stored task result carries it): `{extra['run_key']}`"]
               if "run_key" in extra else []),
             f"- PRESPEC version: {version}",
             f"- Configuration file: `{config_path.name}` (SHA-256 {sha256(config_path)})",
             f"- Python {platform.python_version()} on {platform.platform()}", "",
             "## Package versions", "", *[f"- {k} {v}" for k, v in versions.items()], "",
             "## Raw data files (SHA-256, verified before the run)", "", *[f"- {k}  {v}" for k, v in hashes.items()], "",
             *(["## Daily rows prepared from them (data/clean, SHA-256)", "", *[f"- {k}  {v}" for k, v in sorted(extra["bundle_hashes"].items())], ""]
               if "bundle_hashes" in extra else []),
             "## Variants run", "", *[f"- {n}: {k} instruments, task time {s:.0f} s" for n, k, s in summary], "",
             "## Configuration", "", "```json", cfg.to_json(), "```", ""]
    S.write_atomic(RESULTS / "RUN_INFO.md", lambda p: p.write_text("\n".join(lines)))      # whole or not at all
    (RESULTS / "RUN_STARTED.lock").unlink(missing_ok=True)
    print("wrote results/RUN_INFO.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dev", action="store_true", help="development window of US500 only")
    g.add_argument("--full", action="store_true", help="the one full run (needs an approved PRESPEC.md)")
    ap.add_argument("--config", type=Path, help="study configuration (JSON), required with --full")
    ap.add_argument("--workers", type=int, help="worker processes for --full (default: CPU count minus two)")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted full run")
    a = ap.parse_args()
    if a.dev:
        run_dev()
    else:
        if a.config is None:
            ap.error("--full needs --config")
        run_full(a.config, a.workers, a.resume)


if __name__ == "__main__":
    main()
