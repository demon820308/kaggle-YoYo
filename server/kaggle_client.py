"""Async Kaggle Public REST API client using KGAT Bearer Token."""

from __future__ import annotations

import asyncio
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
        kernel_sources: list[str] | None = None,
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
            "machineShape": machine_shape if enable_gpu else "None",
            "datasetDataSources": dataset_sources or [],
            "competitionDataSources": [],
            "kernelDataSources": kernel_sources or [],
        }
        if docker_image:
            payload["dockerImage"] = docker_image
        async with httpx.AsyncClient(
            base_url=KAGGLE_API_BASE, headers=self._headers(), timeout=60, follow_redirects=True
        ) as client:
            r = await client.post("/kernels/push", json=payload)
            if r.status_code >= 400:
                raise RuntimeError(f"Kaggle push HTTP {r.status_code}: {r.text[:800]}")
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

    async def upload_file_to_dataset(
        self,
        *,
        slug: str,
        file_path: Path,
        title: str | None = None,
        owner: str | None = None,
    ) -> dict[str, Any]:
        """Upload a local file into a private Kaggle dataset (create or publish new version)
        and wait until the dataset status becomes 'ready'."""
        resolved_owner = self._resolve_owner(owner)
        if not file_path.is_file():
            raise FileNotFoundError(str(file_path))

        st = file_path.stat()
        size = int(st.st_size)
        mtime = int(st.st_mtime)
        headers = self._headers()

        async with httpx.AsyncClient(
            base_url=KAGGLE_API_BASE, headers=headers, timeout=60, follow_redirects=True
        ) as client:
            # 1) Request upload slot
            r_blob = await client.post(
                "/blobs/upload",
                json={
                    "type": "DATASET",
                    "name": file_path.name,
                    "contentType": "application/octet-stream",
                    "contentLength": size,
                    "lastModifiedEpochSeconds": mtime,
                },
            )
            if r_blob.status_code >= 400:
                raise RuntimeError(f"Kaggle blob upload init HTTP {r_blob.status_code}: {r_blob.text[:500]}")
            b_resp = r_blob.json()
            file_token = next((b_resp[k] for k in ("token", "fileToken", "uploadToken") if b_resp.get(k)), None)
            create_url = next((b_resp[k] for k in ("createUrl", "url", "uploadUrl", "signedUrl") if b_resp.get(k)), None)
            if not file_token or not create_url:
                raise RuntimeError(f"Kaggle blob upload missing token/url: {b_resp}")

        # 2) PUT file bytes directly to signed GCS URL
        async with httpx.AsyncClient(timeout=300, follow_redirects=True) as put_client:
            data = file_path.read_bytes()
            r_put = await put_client.put(
                create_url,
                content=data,
                headers={"Content-Type": "application/octet-stream", "Content-Length": str(len(data))},
            )
            if r_put.status_code not in (200, 201):
                raise RuntimeError(f"Kaggle blob PUT HTTP {r_put.status_code}: {r_put.text[:400]}")

        # 3) Create dataset or publish new version
        ds_info = await self.check_dataset_status(slug, owner=resolved_owner)
        async with httpx.AsyncClient(
            base_url=KAGGLE_API_BASE, headers=headers, timeout=60, follow_redirects=True
        ) as client:
            if not ds_info.get("exists"):
                r_ds = await client.post(
                    "/datasets/create/new",
                    json={
                        "ownerSlug": resolved_owner,
                        "slug": slug,
                        "title": title or slug,
                        "isPrivate": True,
                        "licenses": [{"name": "CC0-1.0"}],
                        "files": [{"token": file_token, "description": ""}],
                    },
                )
            else:
                r_ds = await client.post(
                    f"/datasets/create/version/{resolved_owner}/{slug}",
                    json={
                        "versionNotes": f"Update {file_path.name} ({size} bytes)",
                        "files": [{"token": file_token, "description": ""}],
                        "deleteOldVersions": True,
                    },
                )
            if r_ds.status_code >= 400:
                raise RuntimeError(f"Kaggle dataset publish HTTP {r_ds.status_code}: {r_ds.text[:500]}")
            ds_resp = r_ds.json()
            if ds_resp.get("error") or ds_resp.get("hasError"):
                raise RuntimeError(f"Kaggle dataset publish error: {ds_resp.get('error') or ds_resp}")

        # 4) Poll until dataset becomes ready (max ~60s)
        for _ in range(30):
            cur = await self.check_dataset_status(slug, owner=resolved_owner)
            if cur.get("status") == "ready":
                return cur
            if cur.get("status") == "error":
                raise RuntimeError(f"Kaggle 数据集 {resolved_owner}/{slug} 处理失败。")
            await asyncio.sleep(2.0)
        return await self.check_dataset_status(slug, owner=resolved_owner)
