"""FastAPI application for H3 Kaggle Console (127.0.0.1:8760)."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from server.config import (
    H3_DATASETS,
    OMNIVOICE_CAPABILITIES,
    VOICE_LIBRARY_DIR,
    VOICE_OUTPUTS_DIR,
    WEB_DIR,
    delete_kaggle_account,
    ensure_dirs,
    get_kaggle_account,
    list_kaggle_accounts_public,
    load_kaggle_owner,
    load_kaggle_token,
    mask_token,
    save_kaggle_token,
    switch_kaggle_account,
    upsert_kaggle_account,
)
from server.jobs import JobManager, TERMINAL_STATES
from server.kaggle_client import KaggleClient
from server.storage import load_settings, load_voice_runs, save_settings, save_voice_runs
from server.voice_kaggle import (
    build_srt_content,
    check_instruct_warnings,
    delete_voice_from_library,
    deploy_omnivoice_for_account,
    execute_voice_generation,
    get_voice_library_item,
    list_voice_library,
    probe_audio_meta,
    probe_remote_omnivoice,
    save_voice_to_library,
    validate_instruct,
)

client = KaggleClient()
jobs = JobManager(client)
_UPLOADED_REFS: dict[str, Path] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_dirs()
    # Recover polling for any non-terminal job left in data/jobs.json (never re-push)
    active = jobs.get_active_job()
    if active and active.get("state") in ("queued", "running", "fetching", "reconcile"):
        slug = active["slug"]
        job_id = active["job_id"]
        job_owner = active.get("owner") or (active.get("params") or {}).get("owner")
        acct = get_kaggle_account(job_owner) if job_owner else get_kaggle_account()
        job_client = (
            KaggleClient(token=str(acct["token"]), owner=str(acct["username"]))
            if acct and acct.get("token") and acct.get("username")
            else client
        )
        jobs.log(f"检测到未完结作业 {job_id} ({job_owner or 'unknown'}/{slug})，恢复只读轮询（不重推）", level="warn")

        async def _resume() -> None:
            try:
                await jobs.poll_until_terminal(
                    job_id,
                    slug,
                    poll_interval=20,
                    max_wait_sec=3600,
                    client_override=job_client,
                )
                await jobs.reconcile_quota(
                    job_id,
                    active.get("quota_before_sec", 0.0),
                    client_override=job_client,
                )
                jobs.update_job(job_id, state="done")
            except Exception as exc:
                jobs.update_job(job_id, state="failed", error=str(exc))

        asyncio.create_task(_resume())
    elif active and active.get("state") not in TERMINAL_STATES:
        jobs.update_job(active["job_id"], state="failed", error="控制台重启前作业未进入运行队列，已标记终止")
    yield


app = FastAPI(title="H3 Kaggle Console", lifespan=lifespan)


class TokenPayload(BaseModel):
    token: str
    username: str | None = None
    label: str = ""
    use_pinned_image: bool = False


class AccountUpsertPayload(BaseModel):
    username: str
    token: str
    label: str = ""
    use_pinned_image: bool = False
    activate: bool = True


class AccountSwitchPayload(BaseModel):
    username: str


class AccountDeployPayload(BaseModel):
    force: bool = False


class SettingsPayload(BaseModel):
    colab_fallback_enabled: bool = False


class VoiceGeneratePayload(BaseModel):
    mode: str = "design"
    sentences: list[str]
    instruct: str | None = "女，青年，中音调"
    language: str | None = None
    speed: float | None = 0.85
    duration: float | None = None
    num_step: int = 32
    guidance_scale: float = 2.0
    denoise: bool = True
    normalize_text: bool = False
    postprocess_output: bool = True
    pause_sec: float | None = None
    ref_id: str | None = None
    ref_text: str | None = None
    voice_id: str | None = None
    allow_mixed_instruct: bool = False


class VoiceLibrarySavePayload(BaseModel):
    name: str
    run_id: str | None = None
    mode: str | None = None
    instruct: str | None = None
    language: str | None = None
    speed: float | None = None


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html_path = WEB_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/events")
async def sse_events() -> StreamingResponse:
    q = jobs.subscribe()

    async def gen():
        try:
            init_payload = {
                "logs": jobs.get_logs(),
                "active_job": jobs.get_active_job(),
            }
            yield f"event: init\ndata: {json.dumps(init_payload, ensure_ascii=False)}\n\n"
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"event: {msg['type']}\ndata: {json.dumps(msg['data'], ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield "event: ping\ndata: {}\n\n"
        finally:
            jobs.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


async def _build_state_payload() -> dict[str, Any]:
    acct = get_kaggle_account()
    owner = acct.get("username") if acct else load_kaggle_owner()
    tok = acct.get("token") if acct else load_kaggle_token()
    accounts = list_kaggle_accounts_public()
    configured = bool(owner and tok)
    quota = None
    quota_error = None
    if configured:
        try:
            acct_client = KaggleClient(token=str(tok), owner=str(owner))
            quota = await acct_client.get_quota(owner=str(owner))
        except Exception as exc:
            quota_error = str(exc)
    active_summary = next((a for a in accounts if a.get("active")), None)
    return {
        "owner": owner,
        "token_configured": configured,
        "token_masked": mask_token(tok),
        "active_account": active_summary,
        "accounts": accounts,
        "quota": quota,
        "quota_error": quota_error,
        "active_job": jobs.get_active_job(),
        "settings": load_settings(),
    }


@app.get("/api/state")
async def get_state() -> dict[str, Any]:
    return await _build_state_payload()


@app.get("/api/accounts")
async def get_accounts() -> dict[str, Any]:
    accounts = list_kaggle_accounts_public()
    active = next((a["username"] for a in accounts if a.get("active")), None)
    return {"active_username": active, "accounts": accounts}


@app.post("/api/accounts")
async def add_or_update_account(body: AccountUpsertPayload) -> dict[str, Any]:
    if body.activate and jobs.is_busy():
        active = jobs.get_active_job()
        cur_owner = active.get("owner") if active else None
        if cur_owner and cur_owner != body.username.strip():
            raise HTTPException(status_code=409, detail="当前有任务正在运行，请等待结束后再切换账号（或先取消勾选「添加后立即启用」）。")
    try:
        from server.config import validate_kaggle_token, validate_kaggle_username

        clean_user = validate_kaggle_username(body.username)
        clean_tok = validate_kaggle_token(body.token)
        probe_client = KaggleClient(token=clean_tok, owner=clean_user)
        try:
            quota = await probe_client.get_quota(owner=clean_user)
        except Exception as exc:
            raise ValueError(f"Kaggle API Key 验证失败（{exc}）。请核对 KGAT_ Token 是否完整且未过期。") from exc

        upsert_kaggle_account(
            username=clean_user,
            token=clean_tok,
            label=body.label,
            use_pinned_image=body.use_pinned_image,
            activate=body.activate,
        )
        probe = await probe_remote_omnivoice(clean_user)
        if probe.get("deployed"):
            jobs.log(f"Kaggle 账号 {clean_user} 已验证并保存（检测到云端已部署 OmniVoice，无需重复部署）。")
        else:
            jobs.log(f"Kaggle 账号 {clean_user} 已验证并保存（云端尚未部署 OmniVoice）。")
        state = await _build_state_payload()
        if state.get("quota"):
            jobs._broadcast("quota_update", state["quota"])
        return {
            "ok": True,
            "quota": quota,
            "already_deployed": bool(probe.get("deployed")),
            "state": state,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/accounts/{username}/probe-remote")
async def probe_account_remote(username: str) -> dict[str, Any]:
    try:
        probe = await probe_remote_omnivoice(username)
        state = await _build_state_payload()
        return {"ok": True, "probe": probe, "state": state}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/accounts/{username}/deploy")
async def deploy_account_omnivoice(username: str, body: AccountDeployPayload | None = None) -> dict[str, Any]:
    try:
        force = bool(body.force) if body else False
        res = await deploy_omnivoice_for_account(jobs=jobs, username=username, force=force)
        state = await _build_state_payload()
        return {"ok": True, **res, "state": state}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/accounts/switch")
async def switch_account(body: AccountSwitchPayload) -> dict[str, Any]:
    if jobs.is_busy():
        raise HTTPException(status_code=409, detail="当前有 GPU 任务正在运行，请等待结束后再切换 Kaggle 账号。")
    try:
        switched = switch_kaggle_account(body.username)
        jobs.log(f"已切换当前 Kaggle 账号为：{switched['username']}")
        state = await _build_state_payload()
        if state.get("quota"):
            jobs._broadcast("quota_update", state["quota"])
        return {"ok": True, "state": state}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/accounts/{username}")
async def remove_account(username: str) -> dict[str, Any]:
    active_job = jobs.get_active_job()
    if active_job and active_job.get("owner") == username:
        raise HTTPException(status_code=409, detail=f"账号 {username} 当前有任务正在运行，暂不可删除。")
    try:
        delete_kaggle_account(username)
        jobs.log(f"已移除 Kaggle 账号：{username}")
        state = await _build_state_payload()
        if state.get("quota"):
            jobs._broadcast("quota_update", state["quota"])
        return {"ok": True, "state": state}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/auth/token")
async def set_token(body: TokenPayload) -> dict[str, Any]:
    try:
        target_user = (body.username or "").strip() or load_kaggle_owner()
        if target_user:
            res = await add_or_update_account(
                AccountUpsertPayload(
                    username=target_user,
                    token=body.token,
                    label=body.label,
                    use_pinned_image=body.use_pinned_image,
                    activate=True,
                )
            )
            return {
                "ok": True,
                "token_masked": res["state"]["token_masked"],
                "quota": res["quota"],
                "state": res["state"],
            }
        save_kaggle_token(body.token)
        quota = await client.get_quota()
        jobs.log("Kaggle KGAT Token 已更新并验通。")
        return {"ok": True, "token_masked": mask_token(load_kaggle_token()), "quota": quota}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/quota/refresh")
async def refresh_quota() -> dict[str, Any]:
    acct = get_kaggle_account()
    if not acct or not acct.get("token") or not acct.get("username"):
        raise HTTPException(status_code=400, detail="尚未连接 Kaggle 账号，请先添加用户名与 API Key。")
    try:
        acct_client = KaggleClient(token=str(acct["token"]), owner=str(acct["username"]))
        quota = await acct_client.get_quota(owner=str(acct["username"]))
        jobs._broadcast("quota_update", quota)
        return quota
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/datasets/status")
async def datasets_status() -> dict[str, Any]:
    try:
        items = await asyncio.gather(*(client.check_dataset_status(slug) for slug in H3_DATASETS))
        return {"datasets": list(items)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/settings")
async def update_settings(body: SettingsPayload) -> dict[str, Any]:
    s = load_settings()
    s["colab_fallback_enabled"] = body.colab_fallback_enabled
    save_settings(s)
    return s


@app.post("/api/logs/clear")
async def clear_logs() -> dict[str, Any]:
    jobs.clear_logs()
    return {"ok": True}


@app.get("/api/voice/capabilities")
async def voice_capabilities() -> dict[str, Any]:
    return OMNIVOICE_CAPABILITIES


@app.post("/api/voice/probe-ref")
async def probe_ref_audio(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "ref.m4a").suffix.lower()
    if suffix not in (".m4a", ".wav", ".mp3", ".flac", ".opus"):
        raise HTTPException(status_code=400, detail="仅支持 .m4a / .wav / .mp3 / .flac / .opus 参考音频。")
    ref_id = uuid.uuid4().hex
    tmp_dir = Path(tempfile.gettempdir()) / "h3_console_refs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest = tmp_dir / f"{ref_id}_{Path(file.filename or 'ref').name}"
    dest.write_bytes(await file.read())
    _UPLOADED_REFS[ref_id] = dest
    meta = probe_audio_meta(dest)
    meta["ref_id"] = ref_id
    return meta


@app.post("/api/voice/generate")
async def voice_generate(body: VoiceGeneratePayload) -> dict[str, Any]:
    sentences = [s.strip() for s in body.sentences if s.strip()]
    if not sentences:
        raise HTTPException(status_code=400, detail="请至少输入一句有效文本。")
    if len(sentences) > 50:
        raise HTTPException(status_code=400, detail="单次任务最多 50 句，请分批提交。")
    if body.mode not in ("design", "clone", "auto"):
        raise HTTPException(status_code=400, detail="无效的语音模式（仅支持 design / clone / auto）。")
    if body.speed is not None and not (0.5 <= body.speed <= 2.0):
        raise HTTPException(status_code=400, detail="语速范围需在 0.50 – 2.00 之间。")
    if body.duration is not None and not (0.3 <= body.duration <= 60.0):
        raise HTTPException(status_code=400, detail="单句固定时长需在 0.3 – 60.0 秒之间。")
    if not (4 <= body.num_step <= 64):
        raise HTTPException(status_code=400, detail="扩散步数 num_step 需在 4 – 64 之间。")
    if body.pause_sec is not None and not (0.0 <= body.pause_sec <= 5.0):
        raise HTTPException(status_code=400, detail="句间停顿需在 0.0 – 5.0 秒之间。")

    ref_path: Path | None = None
    ref_name: str | None = None
    prompt_path: Path | None = None
    instruct = body.instruct
    warnings: list[str] = []

    if body.mode == "design":
        try:
            validate_instruct(instruct or "", allow_mixed=body.allow_mixed_instruct)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        warnings = check_instruct_warnings(instruct or "", body.language)
    elif body.mode == "clone":
        if body.voice_id:
            item = get_voice_library_item(body.voice_id)
            if not item or item.get("kind") != "clone_prompt":
                raise HTTPException(status_code=400, detail="未找到对应的克隆声音库条目。")
            candidate = VOICE_LIBRARY_DIR / body.voice_id / item.get("prompt_file", "voice_prompt.pt")
            if not candidate.is_file():
                raise HTTPException(status_code=400, detail="声音库中的 voice_prompt.pt 文件已缺失。")
            prompt_path = candidate
            ref_name = f"声音库:{item.get('name')}"
        else:
            if not body.ref_id or body.ref_id not in _UPLOADED_REFS:
                raise HTTPException(status_code=400, detail="请先上传参考音频或从声音库选择已存声音。")
            ref_path = _UPLOADED_REFS[body.ref_id]
            ref_name = ref_path.name.split("_", 1)[-1]

    try:
        job = await execute_voice_generation(
            jobs=jobs,
            client=client,
            mode=body.mode,
            sentences=sentences,
            instruct=instruct,
            language=(body.language or "").strip() or None,
            speed=body.speed,
            duration=body.duration,
            num_step=body.num_step,
            guidance_scale=body.guidance_scale,
            denoise=body.denoise,
            normalize_text=body.normalize_text,
            postprocess_output=body.postprocess_output,
            pause_sec=body.pause_sec,
            ref_path=ref_path,
            ref_display_name=ref_name,
            ref_text=(body.ref_text or "").strip() or None,
            voice_id=body.voice_id,
            prompt_path=prompt_path,
        )
        return {"ok": True, "job": job, "warnings": warnings}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/voice/library")
async def get_voice_library() -> dict[str, Any]:
    return {"items": list_voice_library()}


@app.post("/api/voice/library")
async def add_voice_library(body: VoiceLibrarySavePayload) -> dict[str, Any]:
    try:
        item = save_voice_to_library(
            name=body.name,
            run_id=body.run_id,
            mode=body.mode,
            instruct=body.instruct,
            language=body.language,
            speed=body.speed,
        )
        return {"ok": True, "item": item}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/voice/library/{voice_id}")
async def remove_voice_library(voice_id: str) -> dict[str, Any]:
    delete_voice_from_library(voice_id)
    return {"ok": True}


@app.get("/api/voice/srt/{run_id}")
async def download_voice_srt(run_id: str) -> Response:
    runs = load_voice_runs()
    run = next((r for r in runs if r.get("run_id") == run_id), None)
    if run is None:
        raise HTTPException(status_code=404, detail="未找到运行记录。")
    srt = build_srt_content(run)
    if not srt.strip():
        raise HTTPException(status_code=400, detail="该次记录没有可用分句音频，无法生成字幕。")
    return Response(
        content=srt.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="voice_{run_id}.srt"'},
    )


@app.get("/api/voice/history")
async def voice_history() -> dict[str, Any]:
    return {"runs": load_voice_runs()}


@app.delete("/api/voice/history/{run_id}")
async def delete_voice_run(run_id: str) -> dict[str, Any]:
    runs = load_voice_runs()
    kept = [r for r in runs if r.get("run_id") != run_id]
    save_voice_runs(kept)
    out_dir = VOICE_OUTPUTS_DIR / run_id
    if out_dir.is_dir():
        shutil.rmtree(out_dir, ignore_errors=True)
    return {"ok": True}


@app.get("/api/voice/audio/{run_id}/{filename}")
async def get_voice_audio(run_id: str, filename: str) -> FileResponse:
    safe_name = Path(filename).name
    target = VOICE_OUTPUTS_DIR / run_id / safe_name
    if not target.is_file():
        raise HTTPException(status_code=404, detail="音频文件不存在。")
    return FileResponse(target, media_type="audio/wav", filename=safe_name)


@app.get("/api/voice/download-all/{run_id}")
async def download_all_voice(run_id: str) -> FileResponse:
    target_dir = VOICE_OUTPUTS_DIR / run_id
    if not target_dir.is_dir():
        raise HTTPException(status_code=404, detail="产物目录不存在。")
    zip_path = target_dir / f"voice_{run_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(target_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in (".wav", ".json", ".log"):
                zf.write(p, arcname=p.name)
    return FileResponse(zip_path, media_type="application/zip", filename=f"voice_{run_id}.zip")
