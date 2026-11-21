"""Kaggle job orchestration for Video Clone (env cache builder, Stage 1 Prep, Stage 2 CosyVoice 2 TTS)."""

from __future__ import annotations

import base64
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from server.config import (
    KERNELS_DIR,
    OMNIVOICE_DOCKER_IMAGE,
    VCLONE_DIR,
    VCLONE_ENV_KERNEL,
    VCLONE_MAX_DURATION_SEC,
    VCLONE_PREP_KERNEL,
    VCLONE_TTS_KERNEL,
    get_kaggle_account,
)
from server.jobs import JobManager
from server.kaggle_client import KaggleClient, clean_kaggle_log
from server.storage import load_vclone_runs, save_vclone_runs
from server.vclone.compose import compose_cloned_video
from server.vclone.downloader import PLATFORM_LABELS, download_video_url
from server.vclone.media import (
    build_segments_srt,
    probe_media,
    slice_segment_clips,
    standardize_and_extract,
)


def _b64_chunks(data: bytes | None) -> list[str]:
    if not data:
        return []
    b64 = base64.b64encode(data).decode("ascii")
    return [b64[i : i + 120] for i in range(0, len(b64), 120)]


def get_vclone_run(run_id: str) -> dict[str, Any] | None:
    for r in load_vclone_runs():
        if r.get("run_id") == run_id:
            return r
    return None


def upsert_vclone_run(record: dict[str, Any]) -> dict[str, Any]:
    runs = load_vclone_runs()
    for i, r in enumerate(runs):
        if r.get("run_id") == record["run_id"]:
            runs[i] = record
            save_vclone_runs(runs)
            return record
    runs.insert(0, record)
    save_vclone_runs(runs)
    return record


def delete_vclone_run(run_id: str) -> None:
    runs = [r for r in load_vclone_runs() if r.get("run_id") != run_id]
    save_vclone_runs(runs)
    out_dir = VCLONE_DIR / run_id
    if out_dir.is_dir():
        shutil.rmtree(out_dir, ignore_errors=True)


async def check_vclone_env_status(username: str | None = None) -> dict[str, Any]:
    """Read-only check (0 GPU seconds) whether <owner>/yoyo-vclone-env is ready on Kaggle."""
    acct = get_kaggle_account(username)
    if not acct or not acct.get("token") or not acct.get("username"):
        return {"configured": False, "ready": False, "status": "no_account"}
    owner = str(acct["username"])
    slug = VCLONE_ENV_KERNEL["slug"]
    kc = KaggleClient(token=str(acct["token"]), owner=owner)
    try:
        st = await kc.get_kernel_status(slug, owner=owner)
        kstatus = str(st.get("status") or "").lower()
        return {
            "configured": True,
            "owner": owner,
            "slug": slug,
            "ready": kstatus == "complete",
            "status": kstatus or "missing",
            "failure_message": st.get("failureMessage") or "",
        }
    except Exception:
        return {
            "configured": True,
            "owner": owner,
            "slug": slug,
            "ready": False,
            "status": "missing",
        }


async def build_vclone_env(
    *,
    jobs: JobManager,
    include_roformer: bool = False,
) -> dict[str, Any]:
    """Push CPU kernel yoyo-vclone-env (enable_gpu=False, 0 GPU quota consumed) to pre-cache wheels & weights."""
    acct = get_kaggle_account()
    if not acct or not acct.get("username") or not acct.get("token"):
        raise ValueError("尚未连接 Kaggle 账号，请先填写 Kaggle 用户名与 KGAT_ API Key。")
    owner = str(acct["username"])
    kc = KaggleClient(token=str(acct["token"]), owner=owner)
    slug = VCLONE_ENV_KERNEL["slug"]

    tpl = (KERNELS_DIR / "vclone_env_mover.py.tpl").read_text(encoding="utf-8")
    params = {"include_roformer": bool(include_roformer)}
    script_text = tpl.replace("__PARAMS_JSON_STR__", repr(json.dumps(params, ensure_ascii=False)))

    job_id = f"vclone-env-{time.strftime('%H%M%S')}"

    async def _runner(job: dict[str, Any]) -> None:
        jobs.log(f"[加速环境] 正在向 {owner}/{slug} 推送 CPU 缓存构建内核（不消耗 GPU 额度）...")
        jobs.update_job(job["job_id"], state="pushing")
        push_resp = await kc.push_kernel(
            slug=slug,
            title=VCLONE_ENV_KERNEL["title"],
            code_text=script_text,
            enable_gpu=False,
            enable_internet=True,
            owner=owner,
        )
        ver = push_resp.get("versionNumber")
        jobs.update_job(job["job_id"], state="queued", kaggle_status="queued", version_number=ver)
        jobs.log(f"[加速环境] 已推送 {owner}/{slug} v{ver}（CPU 节点），正在后台预下载 wheels 与 CosyVoice 2 / Whisper / UVR 模型...")

        final_st = await jobs.poll_until_terminal(
            job["job_id"],
            slug,
            poll_interval=15,
            max_wait_sec=2400,
            client_override=kc,
        )
        jobs.update_job(job["job_id"], state="fetching")
        out_data = await kc.get_kernel_output(slug, owner=owner)
        log_lines = clean_kaggle_log(out_data.get("logNullable") or out_data.get("log") or "")
        for line in log_lines[-15:]:
            jobs.kaggle_log(f"[{slug}] {line}", level="error" if "FAIL" in line else "info")

        ok = final_st == "complete" and any("DONE ENV_OK" in ln for ln in log_lines)
        jobs.update_job(
            job["job_id"],
            state="done" if ok else "failed",
            gpu_seconds_used=0.0,
            error=None if ok else f"加速环境构建未通过（status={final_st}）",
        )
        env_st = await check_vclone_env_status(owner)
        jobs._broadcast("vclone_env_updated", env_st)
        jobs.log(
            f"[加速环境] 构建{'成功（后续克隆任务将自动挂载高速缓存）' if ok else '失败'}！消耗 GPU：0 秒。",
            level="info" if ok else "error",
        )

    return await jobs.start_gpu_job(
        job_id=job_id,
        kind="构建克隆加速环境(CPU·0额度)",
        slug=slug,
        params_summary={"owner": owner, "slug": slug, "include_roformer": include_roformer},
        runner=_runner,
        owner=owner,
        client_override=kc,
    )


async def execute_vclone_prep(
    *,
    jobs: JobManager,
    raw_url: str | None = None,
    local_video_path: Path | None = None,
    local_video_title: str | None = None,
    clip_start: float = 0.0,
    clip_end: float | None = None,
    separator_model: str = "UVR-MDX-NET-Inst_HQ_3.onnx",
    whisper_model: str = "large-v3-turbo",
    source_language: str | None = None,
    one_click_tts: bool = False,
    cosy_mode: str = "zero_shot",
    cosy_instruct: str = "",
    cosy_speed: float = 1.0,
    auto_align: bool = True,
    bgm_volume: float = 0.65,
    burn_subtitles: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    acct = get_kaggle_account()
    if not acct or not acct.get("username") or not acct.get("token"):
        raise ValueError("尚未连接 Kaggle 账号，请先填写 Kaggle 用户名与 KGAT_ API Key。")
    if jobs.is_busy():
        active = jobs.get_active_job()
        raise ValueError(f"当前已有任务在运行（{active.get('kind') if active else 'GPU占用中'}），请等待完成后再提交。")

    owner = str(acct["username"])
    kc = KaggleClient(token=str(acct["token"]), owner=owner)
    docker_image = OMNIVOICE_DOCKER_IMAGE if acct.get("use_pinned_image") else None
    slug = VCLONE_PREP_KERNEL["slug"]

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    out_dir = VCLONE_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download video or copy local uploaded video
    if local_video_path and local_video_path.is_file():
        jobs.log(f"[视频克隆] 正在读取本地上传视频：{local_video_title or local_video_path.name}")
        raw_video = out_dir / f"raw_video{local_video_path.suffix.lower() or '.mp4'}"
        shutil.copy2(local_video_path, raw_video)
        p_meta = probe_media(raw_video)
        dl_meta = {
            "url": "",
            "platform": "local",
            "platform_label": PLATFORM_LABELS["local"],
            "title": local_video_title or local_video_path.stem,
            "uploader": "本地文件",
            "duration": float(p_meta.get("duration_sec") or 0.0),
            "thumbnail": "",
            "raw_filename": raw_video.name,
        }
    elif raw_url and raw_url.strip():
        jobs.log(f"[视频克隆] 正在通过 yt-dlp 下载视频：{raw_url.strip()[:80]}...")
        raw_video, dl_meta = await download_video_url(raw_url.strip(), out_dir)
        jobs.log(f"[视频克隆] 下载完成：《{dl_meta['title']}》（{dl_meta['platform_label']}，时长 {dl_meta['duration']}s）")
    else:
        raise ValueError("请提供视频链接或上传本地视频文件。")

    # Step 2: Local FFmpeg standardize & audio extract
    jobs.log(f"[视频克隆] 正在用本地 FFmpeg 标准化画面 (25fps H.264) 并抽取音轨（裁剪区间 {clip_start}s ~ {clip_end or '末尾'}）...")
    media_info = await standardize_and_extract(
        raw_video,
        out_dir,
        clip_start=clip_start,
        clip_end=clip_end,
        max_duration=VCLONE_MAX_DURATION_SEC,
    )
    upload_audio_path = out_dir / media_info["upload_audio"]
    audio_bytes = upload_audio_path.read_bytes()

    # Upload lossless FLAC to private Kaggle dataset yoyo-vclone-input when > 600 KB (bypasses 1MB script limit)
    dataset_sources: list[str] = []
    input_filename = ""
    inline_audio_bytes: bytes | None = audio_bytes
    inline_ext = media_info["upload_ext"]

    if len(audio_bytes) > 600 * 1024:
        try:
            jobs.log(
                f"[视频克隆] 正在将无损音频 {upload_audio_path.name} ({round(len(audio_bytes)/1024/1024, 2)} MB) "
                f"上传至 Kaggle 私有数据集 {owner}/yoyo-vclone-input（绕过 1MB 脚本限制）..."
            )
            ds_res = await kc.upload_file_to_dataset(
                slug="yoyo-vclone-input",
                file_path=upload_audio_path,
                title="YoYo VClone Input",
                owner=owner,
            )
            if ds_res.get("status") == "ready":
                dataset_sources = [f"{owner}/yoyo-vclone-input"]
                input_filename = upload_audio_path.name
                inline_audio_bytes = None
                jobs.log(f"[视频克隆] 无损音频数据集 {owner}/yoyo-vclone-input 已就绪（0 音质损失）。")
            else:
                raise RuntimeError(f"数据集状态未就绪: {ds_res.get('status')}")
        except Exception as ds_exc:
            fb_name = media_info.get("fallback_ogg")
            fb_path = (out_dir / fb_name) if fb_name else None
            if fb_path and fb_path.is_file() and fb_path.stat().st_size <= 680 * 1024:
                jobs.log(f"[视频克隆] 数据集上传失败（{ds_exc}），自动回退为压缩 Opus 内嵌上传。", level="warn")
                inline_audio_bytes = fb_path.read_bytes()
                inline_ext = ".ogg"
            else:
                raise

    # Check if yoyo-vclone-env acceleration kernel output is ready
    env_st = await check_vclone_env_status(owner)
    kernel_sources = [f"{owner}/{VCLONE_ENV_KERNEL['slug']}"] if env_st.get("ready") else []

    tpl = (KERNELS_DIR / "vclone_prep.py.tpl").read_text(encoding="utf-8")
    params = {
        "audio_ext": inline_ext,
        "audio_b64_parts": _b64_chunks(inline_audio_bytes),
        "input_filename": input_filename,
        "separator_model": separator_model,
        "whisper_model": whisper_model,
        "source_language": (source_language or "").strip() or None,
        "one_click_tts": bool(one_click_tts),
        "cosy_mode": cosy_mode,
        "cosy_instruct": cosy_instruct,
        "cosy_speed": float(cosy_speed),
    }
    script_text = tpl.replace("__PARAMS_JSON_STR__", repr(json.dumps(params, ensure_ascii=False)))

    record: dict[str, Any] = {
        "run_id": run_id,
        "owner": owner,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": "prep_running",
        "one_click": bool(one_click_tts),
        "source": dl_meta,
        "media": media_info,
        "separator_model": separator_model,
        "whisper_model": whisper_model,
        "source_language": source_language or "",
        "detected_language": "",
        "used_cached_env": bool(kernel_sources),
        "segments": [],
        "ref_clip": None,
        "cosy_mode": cosy_mode,
        "cosy_instruct": cosy_instruct,
        "cosy_speed": cosy_speed,
        "compose": None,
        "prep_gpu_sec": 0.0,
        "tts_gpu_sec": 0.0,
        "total_gpu_sec": 0.0,
        "verdict": "",
        "error": None,
    }
    upsert_vclone_run(record)

    async def _runner(job: dict[str, Any]) -> None:
        try:
            await _runner_impl(job)
        except Exception as exc:
            record["stage"] = "prep_failed"
            record["error"] = str(exc)
            upsert_vclone_run(record)
            jobs._broadcast("vclone_updated", record)
            raise

    async def _runner_impl(job: dict[str, Any]) -> None:
        mode_tag = "一键全流程克隆" if one_click_tts else "提取人声与台词"
        accel_tag = "已挂载加速缓存" if kernel_sources else "联网安装模式"
        jobs.log(f"[视频克隆-{mode_tag}] 推送 {owner}/{slug}（{accel_tag}，音频时长 {media_info['clip_duration']}s，载荷 {round(len(script_text)/1024, 1)} KB）...")
        jobs.update_job(job["job_id"], state="pushing")

        push_resp = await kc.push_kernel(
            slug=slug,
            title=VCLONE_PREP_KERNEL["title"],
            code_text=script_text,
            enable_gpu=True,
            enable_internet=True,
            machine_shape=VCLONE_PREP_KERNEL["machine_shape"],
            docker_image=docker_image,
            dataset_sources=dataset_sources,
            kernel_sources=kernel_sources,
            owner=owner,
        )
        ver = push_resp.get("versionNumber")
        jobs.update_job(job["job_id"], state="queued", kaggle_status="queued", version_number=ver)
        jobs.set_kaggle_status(slug=slug, status="queued", version=ver, elapsed_sec=0)
        jobs.kaggle_log(f"[{slug}] 已推送 {owner}/{slug} v{ver}，开始排队...")
        jobs.log(f"[视频克隆-{mode_tag}] 推送成功（{owner}/{slug} v{ver}），等待 Kaggle T4 执行...")

        final_kstatus = await jobs.poll_until_terminal(
            job["job_id"],
            slug,
            poll_interval=15,
            max_wait_sec=2700,
            client_override=kc,
        )
        jobs.update_job(job["job_id"], state="fetching")
        out_data = await kc.get_kernel_output(slug, owner=owner)
        log_lines = clean_kaggle_log(out_data.get("logNullable") or out_data.get("log") or "")
        (out_dir / "prep_kernel.log").write_text("\n".join(log_lines), encoding="utf-8")
        for line in log_lines[-80:]:
            jobs.kaggle_log(f"[{slug}] {line}", level="error" if "FAIL" in line or "Error" in line else "info")

        for f in out_data.get("files", []):
            fn = f.get("fileName")
            if fn and f.get("url"):
                dest_name = "prep_result.json" if fn == "result.json" else fn
                await kc.download_file(f["url"], out_dir / dest_name)

        quota_after = await jobs.reconcile_quota(
            job["job_id"],
            job["quota_before_sec"],
            client_override=kc,
        )
        gpu_sec = round(max(0.0, quota_after["used_sec"] - job["quota_before_sec"]), 2)

        rj_path = out_dir / "prep_result.json"
        res_json = json.loads(rj_path.read_text(encoding="utf-8")) if rj_path.is_file() else {}
        verdict = str(res_json.get("verdict") or "")
        segments = res_json.get("segments") or []

        if final_kstatus != "complete" or not verdict.endswith("OK") or not segments:
            err_msg = f"云端预处理未通过：status={final_kstatus}, verdict={verdict or '无结果'}, detail={res_json.get('error') or ''}"
            record["stage"] = "prep_failed"
            record["verdict"] = verdict or final_kstatus
            record["error"] = err_msg
            record["prep_gpu_sec"] = gpu_sec
            record["total_gpu_sec"] = round(record["prep_gpu_sec"] + record["tts_gpu_sec"], 2)
            upsert_vclone_run(record)
            jobs.update_job(job["job_id"], state="failed", error=err_msg)
            jobs._broadcast("vclone_updated", record)
            return

        # Slice per-sentence original vocal clips locally for browser audition
        vocals_wav = out_dir / "vocals.wav"
        await slice_segment_clips(vocals_wav, segments, out_dir)
        (out_dir / "origin.srt").write_text(build_segments_srt(segments, use_orig=True), encoding="utf-8")

        best_ref = res_json.get("best_ref") or {}
        best_ref["filename"] = "ref_clip.wav" if (out_dir / "ref_clip.wav").is_file() else None

        record["detected_language"] = res_json.get("detected_language") or ""
        record["used_cached_env"] = bool(res_json.get("used_cached_env"))
        record["segments"] = segments
        record["ref_clip"] = best_ref
        record["prep_timings"] = res_json.get("timings") or {}
        record["prep_gpu_sec"] = gpu_sec
        record["total_gpu_sec"] = round(record["prep_gpu_sec"] + record["tts_gpu_sec"], 2)
        record["verdict"] = verdict
        record["error"] = None

        if one_click_tts and verdict == "PREP_AND_TTS_OK":
            jobs.log("[视频克隆-一键模式] 云端 CosyVoice 2 配音完成，正在本地对齐时间轴并与伴奏合成 MP4...")
            comp = await compose_cloned_video(
                out_dir,
                segments,
                auto_align=auto_align,
                bgm_volume=bgm_volume,
                burn_subtitles=burn_subtitles,
            )
            record["compose"] = comp
            record["stage"] = "completed"
            upsert_vclone_run(record)
            jobs.update_job(job["job_id"], state="done")
            jobs._broadcast("vclone_completed", record)
            jobs.log(f"[视频克隆-一键模式] 全部完成！成片已生成（耗用 GPU {gpu_sec:.1f}s）。")
        else:
            record["stage"] = "prep_ready"
            upsert_vclone_run(record)
            jobs.update_job(job["job_id"], state="done")
            jobs._broadcast("vclone_updated", record)
            jobs.log(
                f"[视频克隆-提取完成] 已分离纯净人声与伴奏，并识别出 {len(segments)} 句台词（耗用 GPU {gpu_sec:.1f}s）。"
                f"现在可在中列校对/改写台词并启动 CosyVoice 2 配音！"
            )

    job = await jobs.start_gpu_job(
        job_id=f"vclone-prep-{run_id}",
        kind="视频克隆(全流程)" if one_click_tts else "视频克隆(①人声分离+识别)",
        slug=slug,
        params_summary={
            "owner": owner,
            "run_id": run_id,
            "title": dl_meta["title"],
            "one_click": one_click_tts,
        },
        runner=_runner,
        owner=owner,
        client_override=kc,
    )
    return record, job


async def execute_vclone_tts(
    *,
    jobs: JobManager,
    run_id: str,
    segments: list[dict[str, Any]] | None = None,
    ref_text: str | None = None,
    cosy_mode: str = "zero_shot",
    cosy_instruct: str = "",
    cosy_speed: float = 1.0,
    auto_align: bool = True,
    vocal_volume: float = 1.0,
    bgm_volume: float = 0.65,
    burn_subtitles: bool = False,
    bilingual_subtitles: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = get_vclone_run(run_id)
    if not record:
        raise ValueError(f"未找到视频克隆任务记录 {run_id}")

    out_dir = VCLONE_DIR / run_id
    if not out_dir.is_dir():
        raise ValueError("该任务的本地素材目录已缺失。")

    acct = get_kaggle_account()
    if not acct or not acct.get("username") or not acct.get("token"):
        raise ValueError("尚未连接 Kaggle 账号，请先填写 Kaggle 用户名与 KGAT_ API Key。")
    if jobs.is_busy():
        active = jobs.get_active_job()
        raise ValueError(f"当前已有任务在运行（{active.get('kind') if active else 'GPU占用中'}），请等待完成后再提交。")

    owner = str(acct["username"])
    kc = KaggleClient(token=str(acct["token"]), owner=owner)
    docker_image = OMNIVOICE_DOCKER_IMAGE if acct.get("use_pinned_image") else None
    slug = VCLONE_TTS_KERNEL["slug"]

    if segments is not None:
        # Merge updated text while preserving timestamps
        by_idx = {int(s["index"]): s for s in record.get("segments") or []}
        merged_segs: list[dict[str, Any]] = []
        for s in segments:
            idx = int(s["index"])
            base = dict(by_idx.get(idx, s))
            base["text"] = (s.get("text") or base.get("orig_text") or "").strip()
            if s.get("start") is not None:
                base["start"] = round(float(s["start"]), 3)
            if s.get("end") is not None:
                base["end"] = round(float(s["end"]), 3)
            base["duration"] = round(max(0.1, float(base["end"]) - float(base["start"])), 3)
            merged_segs.append(base)
        record["segments"] = merged_segs

    active_segments = [s for s in (record.get("segments") or []) if (s.get("text") or "").strip()]
    if not active_segments:
        raise ValueError("台词列表为空，请至少保留一句有效台词。")

    ref_wav_path = out_dir / "ref_clip.wav"
    if not ref_wav_path.is_file():
        vocals_wav = out_dir / "vocals.wav"
        if vocals_wav.is_file():
            ref_wav_path = vocals_wav
        else:
            raise ValueError("未找到参考人声文件 ref_clip.wav。")

    if ref_text is not None:
        if not record.get("ref_clip"):
            record["ref_clip"] = {}
        record["ref_clip"]["ref_text"] = ref_text.strip()

    eff_ref_text = ((record.get("ref_clip") or {}).get("ref_text") or active_segments[0].get("orig_text") or "").strip()

    env_st = await check_vclone_env_status(owner)
    kernel_sources = [f"{owner}/{VCLONE_ENV_KERNEL['slug']}"] if env_st.get("ready") else []

    tpl = (KERNELS_DIR / "vclone_tts.py.tpl").read_text(encoding="utf-8")
    params = {
        "ref_ext": ".wav",
        "ref_b64_parts": _b64_chunks(ref_wav_path.read_bytes()),
        "ref_text": eff_ref_text,
        "segments": [
            {
                "index": int(s["index"]),
                "start": float(s["start"]),
                "end": float(s["end"]),
                "text": (s.get("text") or "").strip(),
            }
            for s in active_segments
        ],
        "cosy_mode": cosy_mode,
        "cosy_instruct": cosy_instruct,
        "cosy_speed": float(cosy_speed),
    }
    script_text = tpl.replace("__PARAMS_JSON_STR__", repr(json.dumps(params, ensure_ascii=False)))

    record["stage"] = "tts_running"
    record["cosy_mode"] = cosy_mode
    record["cosy_instruct"] = cosy_instruct
    record["cosy_speed"] = cosy_speed
    record["error"] = None
    upsert_vclone_run(record)

    async def _runner(job: dict[str, Any]) -> None:
        try:
            await _runner_impl(job)
        except Exception as exc:
            record["stage"] = "tts_failed"
            record["error"] = str(exc)
            upsert_vclone_run(record)
            jobs._broadcast("vclone_updated", record)
            raise

    async def _runner_impl(job: dict[str, Any]) -> None:
        accel_tag = "已挂载加速缓存" if kernel_sources else "联网安装模式"
        jobs.log(f"[视频克隆-②配音] 推送 {owner}/{slug}（CosyVoice 2 · {cosy_mode} · 共 {len(active_segments)} 句 · {accel_tag}）...")
        jobs.update_job(job["job_id"], state="pushing")

        push_resp = await kc.push_kernel(
            slug=slug,
            title=VCLONE_TTS_KERNEL["title"],
            code_text=script_text,
            enable_gpu=True,
            enable_internet=True,
            machine_shape=VCLONE_TTS_KERNEL["machine_shape"],
            docker_image=docker_image,
            kernel_sources=kernel_sources,
            owner=owner,
        )
        ver = push_resp.get("versionNumber")
        jobs.update_job(job["job_id"], state="queued", kaggle_status="queued", version_number=ver)
        jobs.set_kaggle_status(slug=slug, status="queued", version=ver, elapsed_sec=0)
        jobs.kaggle_log(f"[{slug}] 已推送 {owner}/{slug} v{ver}，开始排队...")
        jobs.log(f"[视频克隆-②配音] 推送成功（{owner}/{slug} v{ver}），等待 Kaggle T4 运行 CosyVoice 2...")

        final_kstatus = await jobs.poll_until_terminal(
            job["job_id"],
            slug,
            poll_interval=15,
            max_wait_sec=2400,
            client_override=kc,
        )
        jobs.update_job(job["job_id"], state="fetching")
        out_data = await kc.get_kernel_output(slug, owner=owner)
        log_lines = clean_kaggle_log(out_data.get("logNullable") or out_data.get("log") or "")
        (out_dir / "tts_kernel.log").write_text("\n".join(log_lines), encoding="utf-8")
        for line in log_lines[-80:]:
            jobs.kaggle_log(f"[{slug}] {line}", level="error" if "FAIL" in line or "Error" in line else "info")

        for f in out_data.get("files", []):
            fn = f.get("fileName")
            if fn and f.get("url"):
                dest_name = "tts_result.json" if fn == "result.json" else fn
                await kc.download_file(f["url"], out_dir / dest_name)

        quota_after = await jobs.reconcile_quota(
            job["job_id"],
            job["quota_before_sec"],
            client_override=kc,
        )
        gpu_sec = round(max(0.0, quota_after["used_sec"] - job["quota_before_sec"]), 2)

        rj_path = out_dir / "tts_result.json"
        res_json = json.loads(rj_path.read_text(encoding="utf-8")) if rj_path.is_file() else {}
        verdict = str(res_json.get("verdict") or "")

        record["tts_gpu_sec"] = round(float(record.get("tts_gpu_sec") or 0.0) + gpu_sec, 2)
        record["total_gpu_sec"] = round(float(record.get("prep_gpu_sec") or 0.0) + record["tts_gpu_sec"], 2)
        record["tts_timings"] = res_json.get("timings") or {}
        record["verdict"] = verdict or final_kstatus

        if final_kstatus != "complete" or not verdict.endswith("OK"):
            err_msg = f"CosyVoice 2 配音未通过：status={final_kstatus}, verdict={verdict}, detail={res_json.get('error') or ''}"
            record["stage"] = "tts_failed"
            record["error"] = err_msg
            upsert_vclone_run(record)
            jobs.update_job(job["job_id"], state="failed", error=err_msg)
            jobs._broadcast("vclone_updated", record)
            return

        jobs.log("[视频克隆-合成] 云端音频已拉回，正在本地对齐时间轴、混入伴奏并封装 MP4...")
        comp = await compose_cloned_video(
            out_dir,
            record["segments"],
            auto_align=auto_align,
            vocal_volume=vocal_volume,
            bgm_volume=bgm_volume,
            burn_subtitles=burn_subtitles,
            bilingual_subtitles=bilingual_subtitles,
        )
        record["compose"] = comp
        record["stage"] = "completed"
        record["error"] = None
        upsert_vclone_run(record)

        jobs.update_job(job["job_id"], state="done")
        jobs._broadcast("vclone_completed", record)
        jobs.log(
            f"[视频克隆-完成] 成片已生成！本次配音耗用 GPU {gpu_sec:.1f}s（对齐 {comp['aligned_count']} 句，"
            f"加速调整 {comp['stretched_count']} 句，超长句 {comp['overflow_count']} 句）。"
        )

    job = await jobs.start_gpu_job(
        job_id=f"vclone-tts-{run_id}-{uuid.uuid4().hex[:3]}",
        kind="视频克隆(②CosyVoice 2配音)",
        slug=slug,
        params_summary={
            "owner": owner,
            "run_id": run_id,
            "cosy_mode": cosy_mode,
            "sentence_count": len(active_segments),
        },
        runner=_runner,
        owner=owner,
        client_override=kc,
    )
    return record, job
