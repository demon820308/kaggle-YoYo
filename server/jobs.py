"""Global Kaggle GPU lock, persistent job queue, and SSE event bus."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, Awaitable, Callable

from server.config import scrub_secrets
from server.kaggle_client import KaggleClient
from server.storage import load_jobs, save_jobs

TERMINAL_STATES = {"done", "failed"}


class JobManager:
    def __init__(self, client: KaggleClient) -> None:
        self.client = client
        self._gpu_lock = asyncio.Lock()
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._logs: deque[dict[str, Any]] = deque(maxlen=200)
        self._kaggle_logs: deque[dict[str, Any]] = deque(maxlen=500)
        self._kaggle_status: dict[str, Any] = {
            "slug": "",
            "status": "idle",
            "version": None,
            "elapsed_sec": 0,
            "failure_message": "",
        }
        self._jobs: list[dict[str, Any]] = load_jobs()
        self._active_task: asyncio.Task[None] | None = None

    def get_logs(self) -> list[dict[str, Any]]:
        return list(self._logs)

    def clear_logs(self) -> None:
        self._logs.clear()
        self._broadcast("logs_cleared", {})

    def get_kaggle_logs(self) -> list[dict[str, Any]]:
        return list(self._kaggle_logs)

    def get_kaggle_status(self) -> dict[str, Any]:
        return dict(self._kaggle_status)

    def clear_kaggle_logs(self) -> None:
        self._kaggle_logs.clear()
        self._broadcast("kaggle_logs_cleared", {})

    def set_kaggle_status(
        self,
        *,
        slug: str = "",
        status: str = "idle",
        version: int | None = None,
        elapsed_sec: float = 0.0,
        failure_message: str = "",
    ) -> None:
        self._kaggle_status = {
            "slug": slug,
            "status": status,
            "version": version if version is not None else self._kaggle_status.get("version"),
            "elapsed_sec": round(float(elapsed_sec), 1),
            "failure_message": failure_message,
            "updated_at": time.strftime("%H:%M:%S"),
        }
        self._broadcast("kaggle_status", self._kaggle_status)

    def kaggle_log(self, text: str, level: str = "info") -> None:
        clean = scrub_secrets(text)
        entry = {
            "ts": time.strftime("%H:%M:%S"),
            "level": level,
            "text": clean,
        }
        self._kaggle_logs.append(entry)
        self._broadcast("kaggle_log", entry)

    def log(self, text: str, level: str = "info") -> None:
        clean = scrub_secrets(text)
        entry = {
            "ts": time.strftime("%H:%M:%S"),
            "level": level,
            "text": clean,
        }
        self._logs.append(entry)
        self._broadcast("log", entry)

    def _broadcast(self, event_type: str, payload: Any) -> None:
        msg = {"type": event_type, "data": payload}
        dead: list[asyncio.Queue[dict[str, Any]]] = []
        for q in self._subscribers:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._subscribers.discard(q)

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(q)

    def get_jobs(self) -> list[dict[str, Any]]:
        return list(self._jobs)

    def get_active_job(self) -> dict[str, Any] | None:
        for job in reversed(self._jobs):
            if job.get("state") not in TERMINAL_STATES:
                return job
        return None

    def is_busy(self) -> bool:
        return self._gpu_lock.locked() or self.get_active_job() is not None

    def _save(self) -> None:
        save_jobs(self._jobs)
        self._broadcast("jobs_update", self.get_active_job())

    def update_job(self, job_id: str, **fields: Any) -> dict[str, Any]:
        for job in self._jobs:
            if job["job_id"] == job_id:
                job.update(fields)
                job["updated_at"] = time.time()
                self._save()
                return job
        raise KeyError(job_id)

    async def start_gpu_job(
        self,
        *,
        job_id: str,
        kind: str,
        slug: str,
        params_summary: dict[str, Any],
        runner: Callable[[dict[str, Any]], Awaitable[None]],
        owner: str | None = None,
        client_override: KaggleClient | None = None,
    ) -> dict[str, Any]:
        if self._gpu_lock.locked() or self.get_active_job() is not None:
            active = self.get_active_job()
            desc = active.get("kind", "未知任务") if active else "未知任务"
            raise RuntimeError(f"GPU 锁被占用（当前任务：{desc}），Kaggle 永不并发，请等待结束。")

        active_client = client_override or self.client
        resolved_owner = active_client._resolve_owner(owner)
        quota = await active_client.get_quota(owner=resolved_owner)
        if quota["reserved_sec"] > 0:
            raise RuntimeError(
                f"Kaggle 账号 {resolved_owner} 当前有任务在跑（time_reserved={quota['reserved_sec']}s），禁止并发提交。"
            )

        job = {
            "job_id": job_id,
            "owner": resolved_owner,
            "kind": kind,
            "slug": slug,
            "state": "precheck",
            "kaggle_status": "precheck",
            "version_number": None,
            "quota_before_sec": quota["used_sec"],
            "quota_after_sec": None,
            "gpu_seconds_used": None,
            "params": params_summary,
            "created_at": time.time(),
            "updated_at": time.time(),
            "error": None,
        }
        self._jobs.append(job)
        self._save()

        async def _guarded() -> None:
            async with self._gpu_lock:
                try:
                    await runner(job)
                except Exception as exc:
                    self.log(f"作业 {job_id} 异常：{exc}", level="error")
                    self.update_job(job_id, state="failed", error=scrub_secrets(str(exc)))

        self._active_task = asyncio.create_task(_guarded())
        return job

    async def poll_until_terminal(
        self,
        job_id: str,
        slug: str,
        *,
        poll_interval: int = 20,
        max_wait_sec: int = 3600,
        client_override: KaggleClient | None = None,
    ) -> str:
        active_client = client_override or self.client
        t0 = time.time()
        last_kstatus = ""
        err_streak = 0
        while time.time() - t0 < max_wait_sec:
            await asyncio.sleep(poll_interval)
            try:
                st = await active_client.get_kernel_status(slug)
                err_streak = 0
            except Exception as exc:
                err_streak += 1
                backoff = min(60, 5 * err_streak)
                self.log(f"[{slug}] 状态轮询网络波动（不重推，{backoff}s 后重试）：{exc}", level="warn")
                await asyncio.sleep(backoff)
                continue

            kstatus = str(st.get("status", "")).lower()
            fail_msg = st.get("failureMessage") or ""
            elapsed = round(time.time() - t0, 1)
            self.set_kaggle_status(
                slug=slug,
                status=kstatus or "unknown",
                elapsed_sec=elapsed,
                failure_message=fail_msg,
            )
            if kstatus != last_kstatus:
                self.kaggle_log(f"[{slug}] 状态变更 -> {kstatus} {fail_msg}".strip())
                last_kstatus = kstatus

            mapped_state = "running" if kstatus == "running" else ("queued" if kstatus == "queued" else kstatus)
            self.update_job(job_id, state=mapped_state, kaggle_status=kstatus)

            if kstatus in ("complete", "error", "cancel_acknowledged", "cancelacknowledged", "cancelled", "canceled"):
                return kstatus
        raise TimeoutError(f"[{slug}] 轮询超时（已禁止自动重推，请检查 Kaggle 状态）。")

    async def reconcile_quota(
        self,
        job_id: str,
        used_before: float,
        *,
        client_override: KaggleClient | None = None,
    ) -> dict[str, Any]:
        active_client = client_override or self.client
        self.update_job(job_id, state="reconcile")
        quota = await active_client.get_quota()
        for _ in range(12):
            if quota["reserved_sec"] == 0:
                break
            await asyncio.sleep(5)
            quota = await active_client.get_quota()
        delta = round(max(0.0, quota["used_sec"] - used_before), 2)
        self.update_job(
            job_id,
            quota_after_sec=quota["used_sec"],
            gpu_seconds_used=delta,
        )
        self._broadcast("quota_update", quota)
        return quota
