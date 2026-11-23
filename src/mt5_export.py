"""
Export the M1 history of the study's symbols from the local MT5 terminal (macOS, Wine build).

    python src/mt5_export.py [--offline] [--collect-only] [SYMBOL ...]

--collect-only skips steps 1-2 and only verifies and collects files a previous run left in the terminal folder.

Without --offline the run stops at once if the terminal cannot connect to its trade server; with it, only the
terminal's local history cache is exported (may be incomplete and end early).

Steps:
  1. copy src/mt5/ExportM1.mq5 into the terminal and compile it with MetaEditor,
  2. run the terminal once with a start-up config that executes the (read-only) script and shuts down,
  3. verify every CSV against what the script reported (row count, first and last bar),
  4. move the files to data/raw/<SYM>_M1_<first>_<last>.csv and write
     data/raw/export_meta.tsv and results/raw_data_sha256.tsv.

Nothing is traded; algo trading stays disabled in the start-up config.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MT5 = Path.home() / "Library/Application Support/net.metaquotes.wine.metatrader5/drive_c/Program Files/MetaTrader 5"
WINE = "/Applications/MetaTrader 5.app/Contents/SharedSupport/wine/bin/wine"
OUT = MT5 / "MQL5/Files/rvf_export"
SCRIPT_DIR = MT5 / "MQL5/Scripts/rvf"
INI = MT5 / "config/rvf_export.ini"
RAW = ROOT / "data/raw"

# Names on the MetaQuotes-Demo server: six primary instruments, then the two alternates (pre-specified).
SYMBOLS = ["US500", "USTEC", "XAUUSD", "XAGUSD", "EURUSD", "USDJPY", "DE40", "GBPUSD"]

INI_TEXT = (
    "[Experts]\r\nAllowLiveTrading=0\r\nAllowDllImport=0\r\nEnabled=0\r\n"
    "[StartUp]\r\nScript=rvf\\ExportM1\r\nSymbol=EURUSD\r\nPeriod=M1\r\nShutdownTerminal=1\r\n"
)


def wine(args: list, timeout_s: int) -> None:
    env = dict(os.environ, WINEPREFIX=str(MT5.parents[2]), WINEDEBUG="-all")
    subprocess.run([WINE, *args], cwd=MT5, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   timeout=timeout_s)


def wait_exit(name: str, timeout_s: int) -> None:
    t0 = time.time()
    while subprocess.run(["pgrep", "-q", name]).returncode == 0:      # wine may return before the program exits
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"{name} did not exit within {timeout_s} s")
        time.sleep(3)


def compile_script() -> None:
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "src/mt5/ExportM1.mq5", SCRIPT_DIR / "ExportM1.mq5")
    for ext in ("ex5", "log"):
        (SCRIPT_DIR / f"ExportM1.{ext}").unlink(missing_ok=True)
    # path relative to the terminal folder: MetaEditor silently ignores the absolute path (space in "Program Files")
    wine(["metaeditor64.exe", r"/compile:MQL5\Scripts\rvf\ExportM1.mq5", "/log"], 300)
    wait_exit("metaeditor64", 300)
    log = SCRIPT_DIR / "ExportM1.log"
    text = log.read_bytes().decode("utf-16", errors="replace") if log.exists() else ""
    if not (SCRIPT_DIR / "ExportM1.ex5").exists() or "0 errors" not in text:
        sys.exit(f"compile failed:\n{text}")


def read_meta(path: Path) -> dict:
    meta: dict = {}
    for line in path.read_text(encoding="latin-1").splitlines()[1:]:
        parts = line.split("\t", 2)
        if len(parts) == 3:
            meta.setdefault(parts[0], {})[parts[1]] = parts[2]
    return meta


def stamp(s: str) -> str:
    return s.replace(".", "").replace(":", "").replace(" ", "")


def check_file(path: Path, m: dict) -> str:
    """Row count, first and last bar of the CSV must equal what the script reported."""
    rows, first, last = -1, "", ""
    with open(path, "rb") as f:
        for line in f:
            rows += 1
            if rows == 1:
                first = line
            last = line
    def ts(b: bytes) -> str:
        d, t = b.decode("ascii").split("\t")[:2]
        return f"{d} {t[:5]}"
    if rows != int(m.get("bars_written", -1)):
        return f"{rows} rows vs {m.get('bars_written')} bars written"
    if ts(first) != m.get("first_bar") or ts(last) != m.get("last_bar"):
        return "first/last bar differ from the script's report"
    return ""


def status_ok(m: dict) -> bool:
    """
    OK, or CHECK only because the market was open: bars that formed while the file was being written make the
    written count exceed the count reported before writing, by at most one bar per minute of export time.
    """
    if m.get("status") == "OK":
        return True
    try:
        extra = int(m["bars_written"]) - int(m["bars_reported"])
        return (m.get("status") == "CHECK" and m.get("copy_errors") == "0"
                and 0 <= extra <= float(m["seconds"]) / 60 + 2)
    except (KeyError, ValueError):
        return False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = sys.argv[1:]
    offline = "--offline" in args
    collect_only = "--collect-only" in args
    syms = [a for a in args if not a.startswith("--")] or SYMBOLS
    if subprocess.run(["pgrep", "-q", "terminal64"]).returncode == 0:
        sys.exit("MT5 terminal is already running; close it first (the export needs its own start-up config).")
    if not collect_only:
        compile_script()
        OUT.mkdir(parents=True, exist_ok=True)
        for p in OUT.glob("*"):
            p.unlink()
        (OUT / "symbols.txt").write_bytes("".join(s + "\r\n" for s in syms).encode("ascii"))
        if offline:
            (OUT / "offline_ok.txt").write_bytes(b"1")
        INI.write_bytes(INI_TEXT.encode("ascii"))

        t0 = time.time()
        timeout_s = 1200 * len(syms) + 600
        wine(["terminal64.exe", r"/config:C:\Program Files\MetaTrader 5\config\rvf_export.ini"], timeout_s)
        wait_exit("terminal64", timeout_s)
        print(f"terminal run: {time.time() - t0:.0f} s", flush=True)

    meta_path = OUT / "export_meta.tsv"
    if not meta_path.exists():
        sys.exit("no export_meta.tsv: the script did not run")
    meta = read_meta(meta_path)
    term = meta.get("_terminal", {})
    if term.get("aborted"):
        sys.exit("ABORTED: the terminal could not connect to its trade server (log in first, or use --offline).")
    print("terminal:", {k: term.get(k) for k in ("connected", "account_server", "server_minus_gmt_hours", "done")})

    RAW.mkdir(parents=True, exist_ok=True)
    ok_all = True
    for s in syms:
        m = meta.get(s, {})
        src = OUT / f"{s}_M1.csv"
        problem = "" if status_ok(m) else f"status {m.get('status', 'missing')}"
        if not problem and m.get("reached_server_first_date") != "1":
            problem = (f"history incomplete: terminal has data from {m.get('terminal_first_date')}, "
                       f"server from {m.get('server_first_date')}")
        if not problem and term.get("connected") == "1" and m.get("series_synchronized") != "1":
            problem = "series not synchronised"
        if not problem and not src.exists():
            problem = "file missing"
        if not problem:
            problem = check_file(src, m)
        if problem:
            ok_all = False
            print(f"{s:8s} NOT VERIFIED: {problem} | reported {m.get('bars_reported')} written {m.get('bars_written')} "
                  f"sync {m.get('series_synchronized')} deep {m.get('reached_server_first_date')}")
            continue
        for old in RAW.glob(f"{s}_M1_*.csv"):
            old.unlink()                                               # replaced by this verified export
        dst = RAW / f"{s}_M1_{stamp(m['first_bar'])}_{stamp(m['last_bar'])}.csv"
        shutil.move(str(src), dst)
        print(f"{s:8s} OK {m['bars_written']:>9s} bars {m['first_bar']} .. {m['last_bar']} "
              f"sync {m['series_synchronized']} deep {m['reached_server_first_date']} mode {m.get('chart_mode')}")
    # export_meta.tsv in data/raw: this run's lines, plus earlier lines of symbols this run did not export
    new_lines = meta_path.read_text(encoding="latin-1").splitlines()
    old_path = RAW / "export_meta.tsv"
    old_lines = old_path.read_text(encoding="latin-1").splitlines()[1:] if old_path.exists() else []
    keep = [l for l in old_lines if l.split("\t")[0] not in set(syms) | {"_terminal"}]
    old_path.write_text("\n".join(new_lines + keep) + "\n", encoding="latin-1")
    if term.get("connected") != "1":
        print("NOTE: the terminal was OFFLINE. The files hold only the local cache: they may end early, and 'deep 1' "
              "is not a completeness check (offline the server's first date is just the cache's first date).")

    files = sorted(RAW.glob("*_M1_*.csv")) + [RAW / "export_meta.tsv"]
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results/raw_data_sha256.tsv", "w") as f:
        f.write("file\tbytes\tsha256\n")
        for p in files:
            f.write(f"{p.name}\t{p.stat().st_size}\t{sha256(p)}\n")
    print("hashes: results/raw_data_sha256.tsv")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
