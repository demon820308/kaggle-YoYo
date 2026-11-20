"""Push and monitor h3-model-mover (CPU-only) on Kaggle.

Usage:
  python spike/push_mover.py --placeholder   # creates kernel page for Secrets attachment
  python spike/push_mover.py                 # pushes verbatim h3_model_mover.py and polls
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
SLUG = "h3-model-mover"
ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = ROOT / "h3_model_mover.py"
OUT_DIR = Path(__file__).resolve().parent / "_h3_mover_output"


def load_token() -> str:
    env_tok = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if env_tok:
        return env_tok
    tok_file = ROOT / "data" / "kaggle_token"
    if tok_file.is_file():
        return tok_file.read_text(encoding="utf-8").strip()
    raise RuntimeError("Kaggle token not found.")


def clean_kaggle_log(raw: str) -> list[str]:
    out: list[str] = []
    for line in raw.splitlines():
        s = line.strip().lstrip(",").strip("[]")
        if not s:
            continue
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and "data" in obj:
                t = obj.get("time", 0.0)
                stream = obj.get("stream_name", "")
                out.append(f"[{t:7.1f}s {stream}] " + obj["data"].rstrip("\r\n"))
                continue
        except Exception:
            pass
        out.append(line)
    return out


def main() -> int:
    is_placeholder = "--placeholder" in sys.argv
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = load_token()
    headers = {"Authorization": f"Bearer {tok}", "Accept": "application/json"}

    text = (
        'print("h3-model-mover placeholder ready for Secret attachment.")\n'
        if is_placeholder
        else SCRIPT_PATH.read_text(encoding="utf-8")
    )

    with httpx.Client(base_url=BASE_URL, headers=headers, timeout=60, follow_redirects=True) as client:
        payload = {
            "slug": f"{OWNER}/{SLUG}",
            "newTitle": SLUG,
            "text": text,
            "language": "python",
            "kernelType": "script",
            "isPrivate": True,
            "enableGpu": False,
            "enableInternet": True,
            "datasetDataSources": [],
            "competitionDataSources": [],
            "kernelDataSources": [],
        }
        pr = client.post("/kernels/push", json=payload)
        print(f"[Push] HTTP {pr.status_code}: {pr.text[:500]}", flush=True)
        pr.raise_for_status()

        t0 = time.time()
        last_status = None
        max_wait = 180 if is_placeholder else 12 * 3600
        poll_sec = 5 if is_placeholder else 20
        while time.time() - t0 < max_wait:
            time.sleep(poll_sec)
            try:
                sr = client.get("/kernels/status", params={"userName": OWNER, "kernelSlug": SLUG})
                sr.raise_for_status()
                st = sr.json()
            except Exception as exc:
                print(f"[{time.strftime('%H:%M:%S')}] poll error: {exc}", flush=True)
                continue
            status = str(st.get("status", ""))
            msg = st.get("failureMessage", "")
            if status != last_status or not is_placeholder:
                print(f"[{time.strftime('%H:%M:%S')}] +{(time.time()-t0)/60:.1f}m status={status} msg={msg}", flush=True)
            last_status = status
            if status.lower() in ("complete", "error", "cancel_acknowledged"):
                break

        if not is_placeholder:
            out_r = client.get("/kernels/output", params={"userName": OWNER, "kernelSlug": SLUG})
            out_r.raise_for_status()
            out_data = out_r.json()
            (OUT_DIR / "output_meta.json").write_text(
                json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            raw_log = out_data.get("logNullable") or out_data.get("log") or ""
            lines = clean_kaggle_log(raw_log)
            (OUT_DIR / "kernel.log").write_text("\n".join(lines), encoding="utf-8")
            for f in out_data.get("files", []):
                fn = f.get("fileName")
                if f.get("url") and fn:
                    dl = client.get(f["url"], timeout=60)
                    dl.raise_for_status()
                    (OUT_DIR / fn).write_bytes(dl.content)
                    print(f"Downloaded {fn} ({len(dl.content)} bytes)", flush=True)
            mr = OUT_DIR / "mover_result.json"
            if mr.is_file():
                print("\n=== mover_result.json ===", flush=True)
                print(mr.read_text(encoding="utf-8"), flush=True)
            print("\n=== KERNEL LOG LAST 100 LINES ===", flush=True)
            for l in lines[-100:]:
                print(l, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
