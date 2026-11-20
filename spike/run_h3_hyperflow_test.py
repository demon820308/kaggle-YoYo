"""Push and monitor Nancy's h3_hyperflow_test.py verbatim on Kaggle.

Enforces Section 9 rules:
- Checks gpuQuota.timeReserved == 0 before pushing.
- Sends h3_hyperflow_test.py verbatim (no modifications).
- Polls status without blind re-pushes.
- Downloads output files & saves kernel log tail (last 100 lines) + quota delta.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "https://www.kaggle.com/api/v1"
OWNER = "demon820308"
SLUG = "h3-hyperflow-test"
ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = ROOT / "h3_hyperflow_test.py"
OUT_DIR = Path(__file__).resolve().parent / "_h3_hyperflow_output"


def load_token() -> str:
    env_tok = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if env_tok:
        return env_tok
    tok_file = ROOT / "data" / "kaggle_token"
    if tok_file.is_file():
        return tok_file.read_text(encoding="utf-8").strip()
    raise RuntimeError("Kaggle token not found.")


def parse_quota(q_json: dict) -> tuple[float, float, float]:
    gq = q_json.get("gpuQuota", {})
    used = float(gq.get("timeUsed", {}).get("seconds", 0) or 0) + float(gq.get("timeUsed", {}).get("nanos", 0) or 0) / 1e9
    res = float(gq.get("timeReserved", {}).get("seconds", 0) or 0) + float(gq.get("timeReserved", {}).get("nanos", 0) or 0) / 1e9
    allowed = float(gq.get("totalTimeAllowed", {}).get("seconds", 0) or 0)
    return used, res, allowed


def format_log(raw_log: str) -> list[str]:
    lines: list[str] = []
    for raw_line in raw_log.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            obj = json.loads(raw_line)
            if isinstance(obj, dict) and "data" in obj:
                lines.append(obj["data"].rstrip("\r\n"))
                continue
        except Exception:
            pass
        lines.append(raw_line)
    return lines


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    tok = load_token()
    headers = {"Authorization": f"Bearer {tok}", "Accept": "application/json"}

    with httpx.Client(base_url=BASE_URL, headers=headers, timeout=60, follow_redirects=True) as client:
        # 1. Preflight guardrail
        qr = client.get("/kernels/quota")
        qr.raise_for_status()
        used_before, res_before, allowed = parse_quota(qr.json())
        print(
            f"[Preflight] used={used_before:.2f}s / {allowed:.0f}s | reserved={res_before:.2f}s",
            flush=True,
        )
        if res_before != 0:
            print("BLOCKED: gpuQuota.timeReserved != 0, refusing to push.", flush=True)
            return 2

        # 2. Push verbatim script
        payload = {
            "slug": f"{OWNER}/{SLUG}",
            "newTitle": SLUG,
            "text": source,
            "language": "python",
            "kernelType": "script",
            "isPrivate": True,
            "enableGpu": True,
            "enableInternet": True,
            "machineShape": "NvidiaTeslaT4",
            "datasetDataSources": [],
            "competitionDataSources": [],
            "kernelDataSources": [],
        }
        pr = client.post("/kernels/push", json=payload)
        print(f"[Push] HTTP {pr.status_code}: {pr.text[:500]}", flush=True)
        pr.raise_for_status()

        # 3. Poll until terminal state (up to 12.2 hours, checking every 30s)
        t_start = time.time()
        deadline = t_start + 12.2 * 3600
        last_status = None
        while time.time() < deadline:
            time.sleep(20 if last_status is None else 30)
            try:
                sr = client.get("/kernels/status", params={"userName": OWNER, "kernelSlug": SLUG})
                sr.raise_for_status()
                st = sr.json()
            except Exception as exc:
                print(f"[{time.strftime('%H:%M:%S')}] status poll error (will retry): {exc}", flush=True)
                continue

            status = str(st.get("status", ""))
            fail_msg = st.get("failureMessage", "")
            elapsed_min = (time.time() - t_start) / 60.0
            if status != last_status or int(elapsed_min * 2) % 2 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] +{elapsed_min:.1f}m status={status} msg={fail_msg}", flush=True)
            last_status = status

            if status.lower() in ("complete", "error", "cancel_acknowledged"):
                break

        # 4. Fetch output & log
        out_r = client.get("/kernels/output", params={"userName": OWNER, "kernelSlug": SLUG})
        out_r.raise_for_status()
        out_data = out_r.json()
        (OUT_DIR / "output_meta.json").write_text(
            json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        raw_log = out_data.get("logNullable") or out_data.get("log") or ""
        log_lines = format_log(raw_log)
        (OUT_DIR / "kernel.log").write_text("\n".join(log_lines), encoding="utf-8")

        files = out_data.get("files", [])
        print(f"\n[Outputs] {len(files)} file(s):", flush=True)
        for f in files:
            fname = f.get("fileName") or f.get("name") or "unknown"
            furl = f.get("url")
            print(f"  - {fname}", flush=True)
            if furl:
                dest = OUT_DIR / fname
                dest.parent.mkdir(parents=True, exist_ok=True)
                with client.stream("GET", furl, timeout=300) as dl:
                    dl.raise_for_status()
                    with open(dest, "wb") as fp:
                        for chunk in dl.iter_bytes(chunk_size=1024 * 1024):
                            fp.write(chunk)
                print(f"    saved -> {dest} ({dest.stat().st_size} bytes)", flush=True)

        # 5. Wait briefly for timeReserved to return to 0 for accurate accounting
        used_after, res_after = used_before, 0.0
        for _ in range(12):
            qr2 = client.get("/kernels/quota")
            if qr2.status_code == 200:
                used_after, res_after, _ = parse_quota(qr2.json())
                if res_after == 0:
                    break
            time.sleep(5)

        summary = {
            "final_status": last_status,
            "quota_used_before_sec": round(used_before, 2),
            "quota_used_after_sec": round(used_after, 2),
            "quota_delta_sec": round(used_after - used_before, 2),
            "quota_reserved_after_sec": round(res_after, 2),
            "output_files": [f.get("fileName") for f in files],
        }
        (OUT_DIR / "run_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("\n=== RUN SUMMARY ===", flush=True)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

        rj = OUT_DIR / "result.json"
        if rj.is_file():
            print("\n=== result.json ===", flush=True)
            print(rj.read_text(encoding="utf-8"), flush=True)

        print("\n=== KERNEL LOG LAST 100 LINES ===", flush=True)
        for line in log_lines[-100:]:
            print(line, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
