"""
Run the study.

    python src/run_study.py --dev                      # development: US500, first 1000 retained days only
    python src/run_study.py --full --config FILE.json  # the one full run, on frozen code

The full run refuses to start unless
  - PRESPEC.md is approved: its version is at least 0.3 and its status line no longer says DRAFT, and the
    configuration file names the same version,
  - the working tree has no uncommitted change in tracked files (the commit hash goes into results/RUN_INFO.md),
  - the SHA-256 of every raw data file equals the one stored in results/raw_data_sha256.tsv,
  - neither results/RUN_INFO.md nor results/RUN_STARTED.lock exists: the run happens once. If a run stopped half
    way, look at what it wrote and remove the lock by hand.
Development mode writes to results/dev/ only, never touches data after the first training window and writes
no RUN_INFO.md.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
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


def check_prespec(path: Path, config_version: str) -> str:
    if not path.exists():
        raise SystemExit("REFUSED: no PRESPEC.md")
    version, status = prespec_status(path)
    if version_tuple(version) < (0, 3):
        raise SystemExit(f"REFUSED: PRESPEC.md is version {version}; the full run needs 0.3 or later")
    if "DRAFT" in status.upper():
        raise SystemExit(f"REFUSED: PRESPEC.md still says DRAFT ({status})")
    if config_version != version:
        raise SystemExit(f"REFUSED: the configuration is for PRESPEC {config_version}, PRESPEC.md is {version}")
    return version


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.rstrip("\n")


def git_state() -> tuple[str, list[str]]:
    dirty = [l for l in git("status", "--porcelain", "--untracked-files=no").splitlines() if l.strip()]
    return git("rev-parse", "HEAD"), dirty


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def dev_config() -> S.StudyConfig:
    """Development set-up: US500, a smaller window so that forecasts exist inside the first training window."""
    return S.StudyConfig(prespec_version="dev", instruments=[S.InstrumentSpec("US500", start="2017-03-21")],
                         window=500, refit_every=100, horizons=(1, 5), ms_starts=3, mcs_reps=300, variants=("primary",))


def run_dev(max_days: int = 1000) -> None:
    cfg = dev_config()
    out = RESULTS / "dev"
    daily = {s.symbol: S.prepare_daily(s, "primary", max_days=max_days) for s in cfg.instruments}
    for s, d in daily.items():
        print(f"{s}: {len(d)} retained days {d.index[0].date()} .. {d.index[-1].date()} (development window only)")
    res = S.run_variant("primary", cfg, daily)
    tables = S.evaluate_run(res, cfg)
    S.write_tables(tables, out / "tables", "dev_primary")
    S.make_figures(res, out / "figures", "dev_primary")
    print(f"development outputs in {out.relative_to(ROOT)} (not results; nothing to interpret)")


def run_full(config_path: Path) -> None:
    cfg = S.StudyConfig.from_json(config_path)
    version = check_prespec(ROOT / "PRESPEC.md", cfg.prespec_version)
    if (RESULTS / "RUN_INFO.md").exists() or (RESULTS / "RUN_STARTED.lock").exists():
        raise SystemExit("REFUSED: the full run has been started before (results/RUN_INFO.md or RUN_STARTED.lock exists)")
    commit, dirty = git_state()
    if dirty:
        raise SystemExit("REFUSED: uncommitted changes in tracked files:\n  " + "\n  ".join(dirty))
    symbols = sorted({s.symbol for s in cfg.instruments})
    hashes = verify_raw_hashes(symbols)
    started = datetime.now(timezone.utc)
    (RESULTS / "RUN_STARTED.lock").write_text(f"{started.isoformat()}\ncommit {commit}\n")
    t0 = time.time()
    summary = []
    for name in cfg.variants:
        specs = cfg.instruments
        start = None
        if name == "long_history":
            specs = [s for s in cfg.instruments if s.symbol in cfg.long_history["symbols"]]
            start = cfg.long_history["start"]
        daily = {s.symbol: S.prepare_daily(s, name, start=start) for s in specs}
        res = S.run_variant(name, cfg, daily)
        setup = S.variant_setup(name, cfg)
        tables = S.evaluate_run(res, cfg, families=setup["families"])
        S.write_tables(tables, RESULTS / "tables", name)
        S.save_forecasts(res, RESULTS / "forecasts", name)
        S.make_figures(res, RESULTS / "figures", name, *(("m5_log", "m3_log") if name == "log_har" else
                                                          ("m5_q", "m3_q") if name == "harq" else ("m5", "m3")))
        if name == "primary":                                   # robustness check 7: the same forecasts under the MSE loss
            S.write_tables(S.evaluate_run(res, cfg, loss="mse", pooled=False, extras=False), RESULTS / "tables", "mse_loss")
        summary.append((name, len(specs), round(time.time() - t0)))
        print(f"done: {name} ({time.time() - t0:.0f} s)", flush=True)
    write_run_info(cfg, version, commit, hashes, started, time.time() - t0, summary, config_path)


def write_run_info(cfg, version, commit, hashes, started, seconds, summary, config_path) -> None:
    versions = {p: md.version(p) for p in PACKAGES}
    lines = ["# RUN_INFO", "",
             f"- Run started (UTC): {started:%Y-%m-%d %H:%M:%S}",
             f"- Run time: {seconds / 60:.1f} minutes",
             f"- Git commit of the frozen code: `{commit}`",
             f"- PRESPEC version: {version}",
             f"- Configuration file: `{config_path.name}` (SHA-256 {sha256(config_path)})",
             f"- Python {platform.python_version()} on {platform.platform()}", "",
             "## Package versions", "", *[f"- {k} {v}" for k, v in versions.items()], "",
             "## Raw data files (SHA-256, verified before the run)", "", *[f"- {k}  {v}" for k, v in hashes.items()], "",
             "## Variants run", "", *[f"- {n}: {k} instruments, finished after {s} s" for n, k, s in summary], "",
             "## Configuration", "", "```json", cfg.to_json(), "```", ""]
    (RESULTS / "RUN_INFO.md").write_text("\n".join(lines))
    (RESULTS / "RUN_STARTED.lock").unlink(missing_ok=True)
    print("wrote results/RUN_INFO.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dev", action="store_true", help="development window of US500 only")
    g.add_argument("--full", action="store_true", help="the one full run (needs an approved PRESPEC.md)")
    ap.add_argument("--config", type=Path, help="study configuration (JSON), required with --full")
    a = ap.parse_args()
    if a.dev:
        run_dev()
    else:
        if a.config is None:
            ap.error("--full needs --config")
        run_full(a.config)


if __name__ == "__main__":
    main()
