"""FastAPI router for the Video Clone module (/api/vclone/*)."""

from __future__ import annotations

import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from server.config import VCLONE_CAPABILITIES, VCLONE_DIR, VCLONE_MAX_DURATION_SEC
from server.jobs import JobManager
from server.storage import load_settings, load_vclone_runs, save_settings
from server.vclone.compose import compose_cloned_video
from server.vclone.cookie_harvester import auto_harvest_cookies
from server.vclone.downloader import (
    _effective_proxy,
    list_cookies_status,
    resolve_video_url,
    save_platform_cookies,
)
from server.vclone.kaggle_jobs import (
    build_vclone_env,
    check_vclone_env_status,
    delete_vclone_run,
    execute_vclone_prep,
    execute_vclone_tts,
    get_vclone_run,
    upsert_vclone_run,
)
from server.vclone.media import (
    check_ffmpeg,
    extract_custom_ref_clip,
    probe_media,
)

_LOCAL_VIDEOS: dict[str, tuple[Path, str]] = {}


def _fmt_names(names: list[str] | None) -> str:
    if not names:
        return "-"
    return ", ".join(names[:6]) + ("..." if len(names) > 6 else "")


class ResolveUrlPayload(BaseModel):
    url: str


class CookiesAndNetworkPayload(BaseModel):
    platform: str | None = None
    cookies_text: str | None = None
    proxy_url: str | None = None
    proxy_mode: str | None = None
    cookies_browser: str | None = None


class AutoHarvestCookiePayload(BaseModel):
    platform: str = "douyin"
    headless: bool = True
    target_url: str | None = None


class BuildEnvPayload(BaseModel):
    include_roformer: bool = False


class PrepPayload(BaseModel):
    url: str | None = None
    local_video_id: str | None = None
    clip_start: float = 0.0
    clip_end: float | None = None
    separator_model: str = "UVR-MDX-NET-Inst_HQ_3.onnx"
    whisper_model: str = "large-v3-turbo"
    source_language: str | None = None
    one_click_tts: bool = False
    cosy_mode: str = "zero_shot"
    cosy_instruct: str = ""
    cosy_speed: float = 1.0
    auto_align: bool = True
    bgm_volume: float = 0.65
    burn_subtitles: bool = False
    authorized_confirm: bool = False


class ScriptUpdatePayload(BaseModel):
    segments: list[dict[str, Any]]
    ref_text: str | None = None


class SelectRefPayload(BaseModel):
    segment_index: int | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    ref_text: str | None = None


class TtsGeneratePayload(BaseModel):
    segments: list[dict[str, Any]] | None = None
    ref_text: str | None = None
    cosy_mode: str = "zero_shot"
    cosy_instruct: str = ""
    cosy_speed: float = 1.0
    auto_align: bool = True
    vocal_volume: float = 1.0
    bgm_volume: float = 0.65
    burn_subtitles: bool = False
    bilingual_subtitles: bool = False


class LocalComposePayload(BaseModel):
    segments: list[dict[str, Any]] | None = None
    auto_align: bool = True
    vocal_volume: float = 1.0
    bgm_volume: float = 0.65
    burn_subtitles: bool = False
    bilingual_subtitles: bool = False


def create_vclone_router(jobs: JobManager) -> APIRouter:
    router = APIRouter(prefix="/api/vclone", tags=["vclone"])

    @router.get("/capabilities")
    async def get_capabilities() -> dict[str, Any]:
        env_st = await check_vclone_env_status()
        return {
            **VCLONE_CAPABILITIES,
            "ffmpeg": check_ffmpeg(),
            "cookies": list_cookies_status(),
            "settings": load_settings(),
            "env_status": env_st,
        }

    @router.get("/env-status")
    async def get_env_status() -> dict[str, Any]:
        return await check_vclone_env_status()

    @router.post("/build-env")
    async def start_build_env(body: BuildEnvPayload | None = None) -> dict[str, Any]:
        try:
            inc = bool(body.include_roformer) if body else False
            job = await build_vclone_env(jobs=jobs, include_roformer=inc)
            return {"ok": True, "job": job}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/cookies")
    async def get_cookies_and_network() -> dict[str, Any]:
        return {
            "cookies": list_cookies_status(),
            "settings": load_settings(),
        }

    @router.post("/cookies")
    async def update_cookies_and_network(body: CookiesAndNetworkPayload) -> dict[str, Any]:
        try:
            s = load_settings()
            if body.proxy_url is not None:
                s["proxy_url"] = body.proxy_url.strip()
            if body.proxy_mode in ("overseas", "all", "none"):
                s["proxy_mode"] = body.proxy_mode
            if body.cookies_browser is not None:
                s["cookies_browser"] = body.cookies_browser.strip().lower()
            save_settings(s)

            saved_cookie = None
            if body.platform and body.cookies_text is not None:
                saved_cookie = save_platform_cookies(body.platform, body.cookies_text)
                jobs.log(f"[Cookies] 已更新平台 {body.platform} 的 Cookies（共 {saved_cookie['cookie_count']} 条）。")
            return {
                "ok": True,
                "saved_cookie": saved_cookie,
                "cookies": list_cookies_status(),
                "settings": s,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/cookies/auto-harvest")
    async def harvest_platform_cookies(body: AutoHarvestCookiePayload) -> dict[str, Any]:
        plat = (body.platform or "douyin").strip().lower()
        s = load_settings()
        proxy = _effective_proxy(plat, s)
        mode_desc = "后台静默提取" if body.headless else "弹窗扫码/登录提取"
        jobs.log(f"[Cookies-{mode_desc}] 正在启动本机浏览器提取 {plat} 的 Cookie...")
        try:
            res = await auto_harvest_cookies(
                plat,
                target_url=body.target_url,
                headless=body.headless,
                wait_login_sec=90,
                proxy_url=proxy,
            )
            auth_tag = "已包含登录态" if res.get("has_auth") else "访客/反爬凭据"
            jobs.log(
                f"[Cookies-{mode_desc}] 成功提取 {plat} Cookie 共 {res['cookie_count']} 条（{auth_tag}：{_fmt_names(res.get('cookie_names'))}）。"
            )
            return {
                "ok": True,
                "harvest": res,
                "cookies": list_cookies_status(),
                "settings": s,
            }
        except Exception as exc:
            jobs.log(f"[Cookies-{mode_desc}] 提取失败：{exc}", level="error")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/resolve")
    async def resolve_url(body: ResolveUrlPayload) -> dict[str, Any]:
        try:
            info = await resolve_video_url(body.url)
            return {"ok": True, "info": info}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/upload-local")
    async def upload_local_video(file: UploadFile = File(...)) -> dict[str, Any]:
        orig_name = Path(file.filename or "video.mp4").name
        suffix = Path(orig_name).suffix.lower()
        if suffix not in (".mp4", ".mov", ".mkv", ".webm", ".avi", ".flv", ".m4v"):
            raise HTTPException(status_code=400, detail="仅支持 .mp4 / .mov / .mkv / .webm / .avi / .flv 视频文件。")
        vid_id = uuid.uuid4().hex
        tmp_dir = Path(tempfile.gettempdir()) / "yoyo_vclone_uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        dest = tmp_dir / f"{vid_id}{suffix}"
        dest.write_bytes(await file.read())
        meta = probe_media(dest)
        if not meta.get("ok") or not meta.get("has_audio"):
            dest.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=meta.get("error") or "上传的视频文件缺少音轨或无法解析。")
        _LOCAL_VIDEOS[vid_id] = (dest, Path(orig_name).stem)
        dur = float(meta.get("duration_sec") or 0.0)
        return {
            "ok": True,
            "local_video_id": vid_id,
            "info": {
                "url": "",
                "platform": "local",
                "platform_label": "本地上传视频",
                "title": Path(orig_name).stem,
                "uploader": "本地文件",
                "duration": dur,
                "width": meta.get("width", 0),
                "height": meta.get("height", 0),
                "thumbnail": "",
                "exceeds_limit": dur > VCLONE_MAX_DURATION_SEC,
                "max_duration_sec": VCLONE_MAX_DURATION_SEC,
                "recommended_clip_start": 0.0,
                "recommended_clip_end": round(min(dur, VCLONE_MAX_DURATION_SEC), 1) if dur > 0 else VCLONE_MAX_DURATION_SEC,
            },
        }

    @router.post("/prep")
    async def start_prep(body: PrepPayload) -> dict[str, Any]:
        if not body.authorized_confirm:
            raise HTTPException(status_code=400, detail="请先勾选确认拥有该视频素材及声音的使用授权。")
        if body.clip_end is not None and body.clip_end <= body.clip_start:
            raise HTTPException(status_code=400, detail="裁剪结束时间必须大于起始时间。")
        if body.clip_end is not None and (body.clip_end - body.clip_start) > VCLONE_MAX_DURATION_SEC + 0.5:
            raise HTTPException(status_code=400, detail=f"单次处理素材时长上限为 {int(VCLONE_MAX_DURATION_SEC)} 秒（3 分钟），请调整起止时间。")

        local_path: Path | None = None
        local_title: str | None = None
        if body.local_video_id:
            if body.local_video_id not in _LOCAL_VIDEOS:
                raise HTTPException(status_code=400, detail="本地上传的视频缓存已过期，请重新选择文件。")
            local_path, local_title = _LOCAL_VIDEOS[body.local_video_id]

        try:
            record, job = await execute_vclone_prep(
                jobs=jobs,
                raw_url=body.url,
                local_video_path=local_path,
                local_video_title=local_title,
                clip_start=body.clip_start,
                clip_end=body.clip_end,
                separator_model=body.separator_model,
                whisper_model=body.whisper_model,
                source_language=body.source_language,
                one_click_tts=body.one_click_tts,
                cosy_mode=body.cosy_mode,
                cosy_instruct=body.cosy_instruct,
                cosy_speed=body.cosy_speed,
                auto_align=body.auto_align,
                bgm_volume=body.bgm_volume,
                burn_subtitles=body.burn_subtitles,
            )
            return {"ok": True, "run": record, "job": job}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/runs")
    async def list_runs() -> dict[str, Any]:
        return {"runs": load_vclone_runs()}

    @router.get("/runs/{run_id}")
    async def get_run_detail(run_id: str) -> dict[str, Any]:
        r = get_vclone_run(run_id)
        if not r:
            raise HTTPException(status_code=404, detail="未找到该视频克隆记录。")
        return {"run": r}

    @router.delete("/runs/{run_id}")
    async def remove_run(run_id: str) -> dict[str, Any]:
        delete_vclone_run(run_id)
        return {"ok": True}

    @router.put("/runs/{run_id}/script")
    async def save_run_script(run_id: str, body: ScriptUpdatePayload) -> dict[str, Any]:
        record = get_vclone_run(run_id)
        if not record:
            raise HTTPException(status_code=404, detail="未找到该视频克隆记录。")
        by_idx = {int(s["index"]): s for s in record.get("segments") or []}
        updated: list[dict[str, Any]] = []
        for s in body.segments:
            idx = int(s["index"])
            base = dict(by_idx.get(idx, s))
            base["text"] = (s.get("text") or "").strip()
            if s.get("start") is not None:
                base["start"] = round(float(s["start"]), 3)
            if s.get("end") is not None:
                base["end"] = round(float(s["end"]), 3)
            base["duration"] = round(max(0.1, float(base["end"]) - float(base["start"])), 3)
            updated.append(base)
        record["segments"] = updated
        if body.ref_text is not None:
            if not record.get("ref_clip"):
                record["ref_clip"] = {}
            record["ref_clip"]["ref_text"] = body.ref_text.strip()
        upsert_vclone_run(record)
        return {"ok": True, "run": record}

    @router.post("/runs/{run_id}/select-ref")
    async def select_reference_clip(run_id: str, body: SelectRefPayload) -> dict[str, Any]:
        record = get_vclone_run(run_id)
        if not record:
            raise HTTPException(status_code=404, detail="未找到该视频克隆记录。")
        out_dir = VCLONE_DIR / run_id
        vocals_wav = out_dir / "vocals.wav"
        if not vocals_wav.is_file():
            raise HTTPException(status_code=400, detail="纯净人声 vocals.wav 不存在。")

        start_sec = body.start_sec
        end_sec = body.end_sec
        ref_txt = (body.ref_text or "").strip()
        seg_indices: list[int] = []

        if body.segment_index is not None:
            seg = next((s for s in (record.get("segments") or []) if int(s["index"]) == int(body.segment_index)), None)
            if not seg:
                raise HTTPException(status_code=400, detail=f"未找到第 {body.segment_index} 句。")
            start_sec = float(seg["start"])
            end_sec = float(seg["end"])
            seg_indices = [int(seg["index"])]
            if not ref_txt:
                ref_txt = (seg.get("orig_text") or seg.get("text") or "").strip()
            # If a single segment is very short (< 2.5s), try appending the next segment so reference audio is >= 3s
            if (end_sec - start_sec) < 2.5:
                next_seg = next((s for s in (record.get("segments") or []) if int(s["index"]) == int(body.segment_index) + 1), None)
                if next_seg and (float(next_seg["end"]) - start_sec) <= 12.0:
                    end_sec = float(next_seg["end"])
                    seg_indices.append(int(next_seg["index"]))
                    n_txt = (next_seg.get("orig_text") or "").strip()
                    if n_txt:
                        ref_txt = f"{ref_txt}，{n_txt}"

        if start_sec is None or end_sec is None:
            raise HTTPException(status_code=400, detail="请指定参考音所在的句子序号或起止秒数。")

        try:
            clip_info = extract_custom_ref_clip(vocals_wav, start_sec, end_sec, out_dir / "ref_clip.wav")
            record["ref_clip"] = {
                **clip_info,
                "ref_text": ref_txt,
                "segment_indices": seg_indices,
            }
            upsert_vclone_run(record)
            return {"ok": True, "run": record}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/runs/{run_id}/tts")
    async def start_tts(run_id: str, body: TtsGeneratePayload) -> dict[str, Any]:
        try:
            record, job = await execute_vclone_tts(
                jobs=jobs,
                run_id=run_id,
                segments=body.segments,
                ref_text=body.ref_text,
                cosy_mode=body.cosy_mode,
                cosy_instruct=body.cosy_instruct,
                cosy_speed=body.cosy_speed,
                auto_align=body.auto_align,
                vocal_volume=body.vocal_volume,
                bgm_volume=body.bgm_volume,
                burn_subtitles=body.burn_subtitles,
                bilingual_subtitles=body.bilingual_subtitles,
            )
            return {"ok": True, "run": record, "job": job}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/runs/{run_id}/compose")
    async def recompose_local(run_id: str, body: LocalComposePayload) -> dict[str, Any]:
        record = get_vclone_run(run_id)
        if not record:
            raise HTTPException(status_code=404, detail="未找到该视频克隆记录。")
        out_dir = VCLONE_DIR / run_id
        if not any(out_dir.glob("tts_s*.wav")):
            raise HTTPException(status_code=400, detail="尚未生成 CosyVoice 2 配音音频，请先点击「启动 CosyVoice 2 克隆配音」。")

        if body.segments is not None:
            by_idx = {int(s["index"]): s for s in record.get("segments") or []}
            merged: list[dict[str, Any]] = []
            for s in body.segments:
                idx = int(s["index"])
                base = dict(by_idx.get(idx, s))
                base["text"] = (s.get("text") or base.get("orig_text") or "").strip()
                if s.get("start") is not None:
                    base["start"] = round(float(s["start"]), 3)
                if s.get("end") is not None:
                    base["end"] = round(float(s["end"]), 3)
                merged.append(base)
            record["segments"] = merged

        try:
            comp = await compose_cloned_video(
                out_dir,
                record["segments"],
                auto_align=body.auto_align,
                vocal_volume=body.vocal_volume,
                bgm_volume=body.bgm_volume,
                burn_subtitles=body.burn_subtitles,
                bilingual_subtitles=body.bilingual_subtitles,
            )
            record["compose"] = comp
            record["stage"] = "completed"
            upsert_vclone_run(record)
            jobs.log(f"[视频克隆-本地重混音] 已完成时间轴对齐与视频重封装（0 GPU 消耗）。")
            return {"ok": True, "run": record}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/file/{run_id}/{filename}")
    async def get_vclone_file(run_id: str, filename: str) -> FileResponse:
        safe_name = Path(filename).name
        target = VCLONE_DIR / run_id / safe_name
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"文件 {safe_name} 不存在。")
        ext = target.suffix.lower()
        media_map = {
            ".mp4": "video/mp4",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".srt": "text/plain; charset=utf-8",
            ".json": "application/json",
            ".log": "text/plain; charset=utf-8",
        }
        return FileResponse(target, media_type=media_map.get(ext, "application/octet-stream"), filename=safe_name)

    @router.get("/download-all/{run_id}")
    async def download_all_vclone(run_id: str) -> FileResponse:
        target_dir = VCLONE_DIR / run_id
        if not target_dir.is_dir():
            raise HTTPException(status_code=404, detail="产物目录不存在。")
        zip_path = target_dir / f"vclone_{run_id}.zip"
        keep_files = {
            "final_cloned.mp4",
            "video_std.mp4",
            "vocals.wav",
            "bgm.wav",
            "ref_clip.wav",
            "new_vocals_aligned.wav",
            "final_mixed_audio.wav",
            "tts_full.wav",
            "origin.srt",
            "cloned.srt",
            "bilingual.srt",
            "prep_result.json",
            "tts_result.json",
            "prep_kernel.log",
            "tts_kernel.log",
        }
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(target_dir.iterdir()):
                if p.is_file() and (p.name in keep_files or p.name.startswith("tts_s")):
                    zf.write(p, arcname=p.name)
        return FileResponse(zip_path, media_type="application/zip", filename=f"vclone_{run_id}.zip")

    return router
