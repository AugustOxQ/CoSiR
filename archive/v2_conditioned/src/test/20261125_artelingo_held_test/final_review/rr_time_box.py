"""Scoped re-review (r6 fix wave), item 3: rule section 9's time-box ("The read has not started by Thu 2026-10-15:
no read starts"), agent default: a first read refuses when the Amsterdam date is after 2026-10-15.

The real clock function run_r6_held.amsterdam_today() is driven by a fake `datetime.now` at chosen UTC instants (the
process TZ is set to UTC and to America/New_York in turn), and the real entry point run_r6_held.main(["--mode",
"held", ...]) is called with R.RESULTS and LEDGER pointed at an empty temp folder and a ledger without row H5, so a
read that passes the time-box stops at the next refusal (nothing is written either way).

Mine: Amsterdam date = UTC instant + 2 h (CEST until 2026-10-25); boxed iff no flag and that date > 2026-10-15.

Usage: python rr_time_box.py <out_dir>   -> <out_dir>/rr_time_box.json
"""
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

F = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(F))
import r6_common as R  # noqa: E402
import run_r6_held as RH  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
REAL_DT = RH.datetime


def fake_at(utc):
    class Fake(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            return utc.astimezone(tz) if tz is not None else utc.astimezone().replace(tzinfo=None)
    return Fake


def mine_boxed(utc, flag):
    ams = (utc + timedelta(hours=2)).date()          # CEST (UTC+2) holds from 2026-03-29 to 2026-10-25
    return flag is None and ams > datetime(2026, 10, 15).date(), ams.isoformat()


def files(d):
    return sorted(str(p.relative_to(d)) for p in d.rglob("*"))


if __name__ == "__main__":
    tmp = OUT / "tb"
    res, folder = tmp / "results", tmp / "folder"
    res.mkdir(parents=True, exist_ok=True)
    folder.mkdir(parents=True, exist_ok=True)
    ledger = tmp / "held_ledger.md"
    ledger.write_text("# ledger\n\n| # | Date |\n|---|---|\n")
    instants = {
        "utc 2026-10-15 21:59 (ams 10-15 23:59)": datetime(2026, 10, 15, 21, 59, tzinfo=timezone.utc),
        "utc 2026-10-15 22:00 (ams 10-16 00:00)": datetime(2026, 10, 15, 22, 0, tzinfo=timezone.utc),
        "utc 2026-10-14 23:30 (ams 10-15 01:30)": datetime(2026, 10, 14, 23, 30, tzinfo=timezone.utc),
        "utc 2026-10-15 23:30 (ams 10-16 01:30)": datetime(2026, 10, 15, 23, 30, tzinfo=timezone.utc),
        "utc 2026-10-16 10:00": datetime(2026, 10, 16, 10, 0, tzinfo=timezone.utc),
    }
    out = {}
    for tz in ("UTC", "America/New_York"):
        os.environ["TZ"] = tz
        time.tzset()
        for label, utc in instants.items():
            RH.datetime = fake_at(utc)
            today = RH.amsterdam_today()
            for flag in (None, "--after-crash", "--fix", "--reserve"):
                argv = ["--mode", "held"] + ([flag, "1"] if flag == "--fix" else [flag] if flag else [])
                before = files(tmp)
                # the real entry point: main() -> run_held() with R.RESULTS and LEDGER pointed at the temp folder
                R.RESULTS, RH.LEDGER = res, ledger
                buf = io.StringIO()
                saved_here = RH.HERE
                RH.HERE = folder                       # the started file's committed copy is looked up here
                try:
                    with redirect_stdout(buf):
                        code = RH.main(argv)
                finally:
                    RH.HERE = saved_here
                text = buf.getvalue()
                boxed_code = "so no read starts" in text
                want, ams = mine_boxed(utc, flag)
                out[f"{tz} | {label} | {flag}"] = {
                    "amsterdam_today": today.isoformat(), "mine_date": ams, "exit": code, "boxed": boxed_code,
                    "mine_boxed": want, "wrote": files(tmp) != before,
                    "agree": today.isoformat() == ams and boxed_code == want and code == RH.EXIT_REFUSE
                    and files(tmp) == before,
                    "refusal": text.strip().splitlines()[0][:160] if text.strip() else ""}
    RH.datetime = REAL_DT
    # the smoke never reaches the time-box (run_smoke has no refuse_or_go)
    import inspect
    out["smoke_has_no_time_box"] = {"agree": "refuse_or_go" not in inspect.getsource(RH.run_smoke)}
    (OUT / "rr_time_box.json").write_text(json.dumps(out, indent=1))
    bad = [k for k, v in out.items() if not v["agree"]]
    print("cases", len(out), "disagree", len(bad))
    for k in bad:
        print(" ", k, out[k])
