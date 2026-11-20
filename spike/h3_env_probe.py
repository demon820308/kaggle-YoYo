"""Phase 0 Kaggle H3 Environment Probe v3 (demon820308/h3-env-probe).

Captures literal `nvidia-smi` and `free -g` / `free -m` / `/proc/meminfo`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx

BASE_URL = "https://www.kaggle.com/api/v1"
OWNER = "demon820308"
SLUG = "h3-env-probe"

PROBE_SCRIPT = r'''import json
import subprocess
from pathlib import Path

out = {
    "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
    "free_g": subprocess.check_output(["free", "-g"], text=True),
    "free_m": subprocess.check_output(["free", "-m"], text=True),
    "meminfo_head": "\n".join(Path("/proc/meminfo").read_text().splitlines()[:6]),
}
Path("/kaggle/working/probe_gpu_mem.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("=== nvidia-smi ===")
print(out["nvidia_smi"])
print("=== free -g ===")
print(out["free_g"])
print("=== free -m ===")
print(out["free_m"])
'''


def load_token() -> str:
    env_tok = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if env_tok:
        return env_tok
    tok_file = Path(__file__).resolve().parent.parent / "data" / "kaggle_token"
    if tok_file.is_file():
        return tok_file.read_text(encoding="utf-8").strip()
    raise RuntimeError("Kaggle token not found.")


def main() -> int:
    tok = load_token()
    headers = {"Authorization": f"Bearer {tok}", "Accept": "application/json"}
    with httpx.Client(base_url=BASE_URL, headers=headers, timeout=30, follow_redirects=True) as client:
        q = client.get("/kernels/quota")
        q.raise_for_status()
        reserved = int(q.json().get("gpuQuota", {}).get("timeReserved", {}).get("seconds", 0) or 0)
        if reserved != 0:
            raise RuntimeError(f"Guardrail blocked: gpuQuota.timeReserved={reserved}s != 0")

        payload = {
            "slug": f"{OWNER}/{SLUG}",
            "newTitle": SLUG,
            "text": PROBE_SCRIPT,
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
        push_r = client.post("/kernels/push", json=payload)
        push_r.raise_for_status()
        print(f"Push HTTP {push_r.status_code} version={push_r.json().get('versionNumber')}")

        deadline = time.time() + 600
        while time.time() < deadline:
            time.sleep(8)
            sr = client.get("/kernels/status", params={"userName": OWNER, "kernelSlug": SLUG})
            sr.raise_for_status()
            status = sr.json().get("status", "")
            print(f"[{time.strftime('%H:%M:%S')}] status={status}")
            if str(status).lower() in ("complete", "error", "cancel_acknowledged"):
                break

        out_r = client.get("/kernels/output", params={"userName": OWNER, "kernelSlug": SLUG})
        out_r.raise_for_status()
        for f in out_r.json().get("files", []):
            if f.get("fileName") == "probe_gpu_mem.json":
                dl = httpx.get(f["url"], timeout=30, follow_redirects=True)
                dl.raise_for_status()
                out_dir = Path(__file__).resolve().parent / "_probe_output"
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "probe_gpu_mem.json").write_text(dl.text, encoding="utf-8")
                data = dl.json()
                print("\n--- nvidia-smi ---")
                print(data["nvidia_smi"])
                print("--- free -g ---")
                print(data["free_g"])
                print("--- free -m ---")
                print(data["free_m"])
                print("--- /proc/meminfo ---")
                print(data["meminfo_head"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
