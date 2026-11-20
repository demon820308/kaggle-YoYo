"""Async Kaggle Public REST API client using KGAT Bearer Token."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from server.config import (
    KAGGLE_API_BASE,
    KAGGLE_GPU_WEEKLY_SEC,
    load_kaggle_owner,
    load_kaggle_token,
)


def _duration_sec(obj: dict[str, Any] | None) -> float:
    if not obj:
        return 0.0
    sec = float(obj.get("seconds", 0) or 0)
    nanos = float(obj.get("nanos", 0) or 0)
    return round(sec + nanos / 1e9, 3)


def clean_kaggle_log(raw: str) -> list[str]:
    out: list[str] = []
    for line in (raw or "").splitlines():
        s = line.strip().lstrip(",").strip("[]")
        if not s:
            continue
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and "data" in obj:
                t = float(obj.get("time", 0.0) or 0.0)
                stream = obj.get("stream_name", "")
                out.append(f"[{t:7.1f}s {stream}] " + str(obj["data"]).rstrip("\r\n"))
                continue
        except Exception:
            pass
        out.append(line)
    return out


class KaggleClient:
    def __init__(self, token: str | None = None, owner: str | None = None) -> None:
        self._explicit_token = token
        self._explicit_owner = owner

    def _resolve_owner(self, owner: str | None = None) -> str:
        resolved = (owner or self._explicit_owner or load_kaggle_owner() or "").strip()
        if not resolved:
            raise RuntimeError("Kaggle 账号未配置，请先填写 Kaggle 用户名与 KGAT_ API Key。")
        return resolved

    def _headers(self) -> dict[str, str]:
        tok = self._explicit_token or load_kaggle_token()
        if not tok:
            raise RuntimeError("Kaggle API Key 未配置，请先添加 Kaggle 账号或填写 KGAT_ Token。")
        return {
            "Authorization": f"Bearer {tok}",
            "Accept": "application/json",
        }

    async def get_quota(self, owner: str | None = None) -> dict[str, Any]:
        resolved_owner = (owner or self._explicit_owner or load_kaggle_owner() or "").strip()
        async with httpx.AsyncClient(
            base_url=KAGGLE_API_BASE, headers=self._headers(), timeout=20, follow_redirects=True
        ) as client:
            r = await client.get("/kernels/quota")
            r.raise_for_status()
            data = r.json()
        gq = data.get("gpuQuota", {})
        used = _duration_sec(gq.get("timeUsed"))
        reserved = _duration_sec(gq.get("timeReserved"))
        allowed = _duration_sec(gq.get("totalTimeAllowed")) or float(KAGGLE_GPU_WEEKLY_SEC)
        return {
            "owner": resolved_owner,
            "quota_refresh_time": data.get("quotaRefreshTime", ""),
            "used_sec": used,
            "reserved_sec": reserved,
            "allowed_sec": allowed,
            "remaining_sec": round(max(0.0, allowed - used - reserved), 3),
            "has_ever_run": bool(gq.get("hasEverRun", False)),
        }

    async def pull_kernel(self, slug: str, owner: str | None = None) -> dict[str, Any]:
        resolved_owner = self._resolve_owner(owner)
        async with httpx.AsyncClient(
            base_url=KAGGLE_API_BASE, headers=self._headers(), timeout=30, follow_redirects=True
        ) as client:
            r = await client.get("/kernels/pull", params={"userName": resolved_owner, "kernelSlug": slug})
            r.raise_for_status()
            return r.json()

    async def push_kernel(
        self,
        *,
        slug: str,
        title: str,
        code_text: str,
        enable_gpu: bool = True,
        enable_internet: bool = True,
        machine_shape: str = "NvidiaTeslaT4",
        docker_image: str | None = None,
        dataset_sources: list[str] | None = None,
        owner: str | None = None,
    ) -> dict[str, Any]:
        resolved_owner = self._resolve_owner(owner)
        payload: dict[str, Any] = {
            "slug": f"{resolved_owner}/{slug}",
            "newTitle": title,
            "text": code_text,
            "language": "python",
            "kernelType": "script",
            "isPrivate": True,
            "enableGpu": enable_gpu,
            "enableInternet": enable_internet,
            "machineShape": machine_shape,
            "datasetDataSources": dataset_sources or [],
            "competitionDataSources": [],
            "kernelDataSources": [],
        }
        if docker_image:
            payload["dockerImage"] = docker_image
        async with httpx.AsyncClient(
            base_url=KAGGLE_API_BASE, headers=self._headers(), timeout=60, follow_redirects=True
        ) as client:
            r = await client.post("/kernels/push", json=payload)
            r.raise_for_status()
            resp = r.json()
        if resp.get("error"):
            raise RuntimeError(f"Kaggle push error: {resp['error']}")
        return resp

    async def get_kernel_status(self, slug: str, owner: str | None = None) -> dict[str, Any]:
        resolved_owner = self._resolve_owner(owner)
        async with httpx.AsyncClient(
            base_url=KAGGLE_API_BASE, headers=self._headers(), timeout=20, follow_redirects=True
        ) as client:
            r = await client.get("/kernels/status", params={"userName": resolved_owner, "kernelSlug": slug})
            r.raise_for_status()
            return r.json()

    async def get_kernel_output(self, slug: str, owner: str | None = None) -> dict[str, Any]:
        resolved_owner = self._resolve_owner(owner)
        async with httpx.AsyncClient(
            base_url=KAGGLE_API_BASE, headers=self._headers(), timeout=60, follow_redirects=True
        ) as client:
            r = await client.get("/kernels/output", params={"userName": resolved_owner, "kernelSlug": slug})
            r.raise_for_status()
            return r.json()

    async def download_file(self, url: str, dest: Path) -> int:
        dest.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(headers=self._headers(), timeout=300, follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                total = 0
                with open(dest, "wb") as fp:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                        fp.write(chunk)
                        total += len(chunk)
        return total

    async def check_dataset_status(self, slug: str, owner: str | None = None) -> dict[str, Any]:
        resolved_owner = self._resolve_owner(owner)
        async with httpx.AsyncClient(
            base_url=KAGGLE_API_BASE, headers=self._headers(), timeout=20, follow_redirects=True
        ) as client:
            sr = await client.get(f"/datasets/status/{resolved_owner}/{slug}")
            if sr.status_code in (403, 404):
                return {"slug": slug, "exists": False, "status": "missing", "files": [], "total_bytes": 0}
            status_val = sr.text.strip().strip('"').lower() if sr.status_code == 200 else f"http_{sr.status_code}"
            lr = await client.get(f"/datasets/list/{resolved_owner}/{slug}")
            files = []
            if lr.status_code == 200:
                for it in lr.json().get("datasetFiles", []):
                    files.append(
                        {
                            "name": it.get("name") or it.get("nameNullable") or "",
                            "bytes": int(it.get("totalBytes", 0) or 0),
                        }
                    )
            total_bytes = sum(f["bytes"] for f in files)
            return {
                "slug": slug,
                "exists": True,
                "status": status_val,
                "files": files,
                "file_count": len(files),
                "total_bytes": total_bytes,
                "total_gb": round(total_bytes / (1024**3), 2),
            }
