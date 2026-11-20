"""Phase 0 Kaggle Connectivity Spike (Items 1-3).

Read-only verification against existing kernels:
  - demon820308/yoyo-omnivoice (144856842)
  - demon820308/yoyo-omnivoice-clone (145372284)

Tests:
1. KGAT token auth & GET /api/v1/kernels/quota (108000s / time_reserved parsing)
2. Pull both existing kernels (code + metadata) and inspect parameter sections
   vs Spec Section 12.3. Does NOT push or modify anything.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

BASE_URL = "https://www.kaggle.com/api/v1"
EXPECTED_USER = "demon820308"
KERNELS = [
    ("yoyo-omnivoice", 144856842),
    ("yoyo-omnivoice-clone", 145372284),
]


def load_token() -> str:
    env_tok = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if env_tok:
        return env_tok
    tok_file = Path(__file__).resolve().parent.parent / "data" / "kaggle_token"
    if tok_file.is_file():
        return tok_file.read_text(encoding="utf-8").strip()
    raise RuntimeError(
        "Kaggle token not found. Set KAGGLE_API_TOKEN env var or write it to data/kaggle_token."
    )


def main() -> int:
    token = load_token()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    with httpx.Client(base_url=BASE_URL, headers=headers, timeout=30, follow_redirects=True) as client:
        print("=== [Kaggle Spike 1] Checking GPU quota ===")
        qr = client.get("/kernels/quota")
        print(f"HTTP {qr.status_code}")
        qr.raise_for_status()
        quota = qr.json()
        print(json.dumps(quota, indent=2, ensure_ascii=False))

        out_root = Path(__file__).resolve().parent / "_pulled_kernels"
        out_root.mkdir(parents=True, exist_ok=True)

        print("\n=== [Kaggle Spike 2] Pulling existing kernels (read-only) ===")
        for slug, expected_id in KERNELS:
            pr = client.get(
                "/kernels/pull",
                params={"userName": EXPECTED_USER, "kernelSlug": slug},
            )
            print(f"\nKernel {EXPECTED_USER}/{slug} (expected id {expected_id}): HTTP {pr.status_code}")
            pr.raise_for_status()
            data = pr.json()
            kdir = out_root / slug
            kdir.mkdir(parents=True, exist_ok=True)
            meta = data.get("metadata", {})
            blob = data.get("blob", {})
            (kdir / "metadata.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            source = blob.get("source", "")
            (kdir / "source_preview.txt").write_text(source, encoding="utf-8")
            print(
                f"  id={meta.get('id')} | kernelType={meta.get('kernelType')} | "
                f"enableGpu={meta.get('enableGpu')} | enableInternet={meta.get('enableInternet')} | "
                f"accelerator={meta.get('accelerator')}"
            )
            print("  First 35 lines of source:")
            for idx, line in enumerate(source.splitlines()[:35], 1):
                print(f"    {idx:02d}: {line}")

            sr = client.get(
                "/kernels/status",
                params={"userName": EXPECTED_USER, "kernelSlug": slug},
            )
            print(f"  Status endpoint HTTP {sr.status_code}: {sr.text[:300]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
