"""Kaggle OmniVoice (design / clone / auto) service, guardrails, voice library, and 5-point WAV verification."""

from __future__ import annotations

import base64
import json
import math
import re
import shutil
import struct
import subprocess
import time
import uuid
import wave
from pathlib import Path
from typing import Any

from server.config import (
    KERNELS_DIR,
    OMNIVOICE_CAPABILITIES,
    OMNIVOICE_CLONE_KERNEL,
    OMNIVOICE_DESIGN_KERNEL,
    OMNIVOICE_USE_UNIFIED_KERNEL,
    VOICE_LIBRARY_DIR,
    VOICE_OUTPUTS_DIR,
    get_kaggle_account,
    list_kaggle_accounts_public,
    update_kaggle_account_deploy,
)
from server.jobs import JobManager
from server.kaggle_client import KaggleClient, clean_kaggle_log
from server.storage import (
    load_voice_library,
    load_voice_runs,
    save_voice_library,
    save_voice_runs,
)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def validate_instruct(instruct: str, *, allow_mixed: bool = False) -> None:
    text = (instruct or "").strip()
    if not text:
        raise ValueError("声音设计模式下，声线 instruct 不能为空。")
    if not allow_mixed and _CJK_RE.search(text) and _LATIN_RE.search(text):
        raise ValueError("校验拦截：声线 instruct 不允许中英混写，请使用纯中文或纯英文（或开启允许混写）。")


def check_instruct_warnings(instruct: str, language: str | None) -> list[str]:
    """Return non-blocking warnings for instruct attribute conflicts or language mismatch."""
    warnings: list[str] = []
    tokens = [t.strip().lower() for t in re.split(r"[,，]+", instruct or "") if t.strip()]
    if not tokens:
        return warnings

    attrs = OMNIVOICE_CAPABILITIES["attrs"]
    cat_maps: dict[str, set[str]] = {
        "gender": {x["zh"].lower() for x in attrs["gender"]} | {x["en"].lower() for x in attrs["gender"]},
        "age": {x["zh"].lower() for x in attrs["age"]} | {x["en"].lower() for x in attrs["age"]},
        "pitch": {x["zh"].lower() for x in attrs["pitch"]} | {x["en"].lower() for x in attrs["pitch"]},
        "style": {x["zh"].lower() for x in attrs["style"]} | {x["en"].lower() for x in attrs["style"]},
        "dialect": {x.lower() for x in attrs["dialect"]},
        "accent": {x.lower() for x in attrs["accent"]},
    }
    hits: dict[str, list[str]] = {k: [] for k in cat_maps}
    for tok in tokens:
        for cat, vocab in cat_maps.items():
            if tok in vocab:
                hits[cat].append(tok)
    for cat, matched in hits.items():
        if len(matched) > 1:
            warnings.append(f"属性类目 {cat} 选了多个值（{', '.join(matched)}），同一类目建议只保留 1 个。")

    lang_l = (language or "").strip().lower()
    if hits["dialect"] and lang_l and lang_l not in ("chinese", "zh", "cmn", "mandarin"):
        warnings.append("中文方言仅在合成中文文本时有效。")
    if hits["accent"] and lang_l and lang_l not in ("english", "en"):
        warnings.append("英文口音仅在合成英文文本时有效。")
    return warnings


def probe_audio_meta(path: Path) -> dict[str, Any]:
    size_bytes = path.stat().st_size
    duration_sec: float | None = None
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            data = json.loads(proc.stdout)
            dur = data.get("format", {}).get("duration")
            if dur is not None:
                duration_sec = round(float(dur), 2)
    except Exception:
        pass
    warning = None
    if duration_sec is not None and (duration_sec < 3.0 or duration_sec > 10.0):
        warning = f"参考音频实测 {duration_sec:.2f} 秒（建议 3–10 秒最佳，仍可继续提交）"
    elif size_bytes > 2 * 1024 * 1024:
        warning = f"参考音频体积 {size_bytes / (1024 * 1024):.2f} MB 较大，建议转为 m4a/opus 以加快推送"
    return {
        "filename": path.name,
        "size_bytes": size_bytes,
        "duration_sec": duration_sec,
        "warning": warning,
    }


def analyze_wav(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    duration = round(nframes / sr, 3) if sr > 0 else 0.0
    rms = 0.0
    if nframes > 0 and sw == 2:
        count = len(raw) // 2
        samples = struct.unpack(f"<{count}h", raw)
        sq_sum = sum((s / 32768.0) ** 2 for s in samples)
        rms = round(math.sqrt(sq_sum / max(1, count)), 5)
    elif nframes > 0 and sw == 4:
        count = len(raw) // 4
        samples = struct.unpack(f"<{count}i", raw)
        sq_sum = sum((s / 2147483648.0) ** 2 for s in samples)
        rms = round(math.sqrt(sq_sum / max(1, count)), 5)
    return {
        "sample_rate": sr,
        "channels": ch,
        "sample_width": sw,
        "frames": nframes,
        "duration_sec": duration,
        "rms": rms,
    }


def _b64_chunks(data: bytes | None) -> list[str]:
    if not data:
        return []
    b64 = base64.b64encode(data).decode("ascii")
    return [b64[i : i + 100] for i in range(0, len(b64), 100)]


def render_design_script(sentences: list[str], instruct: str, speed: float) -> str:
    tpl = (KERNELS_DIR / "omnivoice_design.py.tpl").read_text(encoding="utf-8")
    return (
        tpl.replace("__SENTENCES_JSON__", json.dumps(sentences, ensure_ascii=False, indent=4))
        .replace("__INSTRUCT_JSON__", json.dumps(instruct, ensure_ascii=False))
        .replace("__SPEED_STR__", repr(str(round(speed, 2))))
        .replace("__PAUSE_SEC__", repr(OMNIVOICE_DESIGN_KERNEL["pause_sec"]))
    )


def render_clone_script(sentences: list[str], ref_bytes: bytes, speed: float) -> str:
    parts = _b64_chunks(ref_bytes)
    tpl = (KERNELS_DIR / "omnivoice_clone.py.tpl").read_text(encoding="utf-8")
    return (
        tpl.replace("__SPEED_STR__", repr(str(round(speed, 2))))
        .replace("__PAUSE_SEC__", repr(OMNIVOICE_CLONE_KERNEL["pause_sec"]))
        .replace("__SENTENCES_JSON__", json.dumps(sentences, ensure_ascii=False, indent=4))
        .replace("__REF_B64_PARTS_JSON__", json.dumps(parts, indent=4))
    )


def render_unified_script(
    *,
    mode: str,
    sentences: list[str],
    instruct: str | None,
    language: str | None,
    speed: float | None,
    duration: float | None,
    num_step: int,
    guidance_scale: float,
    denoise: bool,
    normalize_text: bool,
    postprocess_output: bool,
    pause_sec: float,
    full_wav_name: str,
    ref_ext: str = ".m4a",
    ref_text: str | None = None,
    ref_bytes: bytes | None = None,
    prompt_bytes: bytes | None = None,
) -> str:
    tpl = (KERNELS_DIR / "omnivoice_unified.py.tpl").read_text(encoding="utf-8")
    params = {
        "mode": mode,
        "sentences": sentences,
        "instruct": instruct,
        "language": language or None,
        "speed": round(speed, 2) if speed is not None else None,
        "duration": round(duration, 2) if duration is not None else None,
        "num_step": int(num_step),
        "guidance_scale": float(guidance_scale),
        "denoise": bool(denoise),
        "normalize_text": bool(normalize_text),
        "postprocess_output": bool(postprocess_output),
        "pause_sec": round(float(pause_sec), 2),
        "full_wav_name": full_wav_name,
        "ref_ext": ref_ext,
        "ref_text": ref_text or None,
        "ref_b64_parts": _b64_chunks(ref_bytes),
        "prompt_b64_parts": _b64_chunks(prompt_bytes),
    }
    return tpl.replace("__PARAMS_JSON_STR__", repr(json.dumps(params, ensure_ascii=False)))


def verify_voice_outputs(
    *,
    out_dir: Path,
    mode: str,
    sentences: list[str],
    pause_sec: float,
    quota_ok: bool,
    gpu_seconds_used: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    full_name = (
        OMNIVOICE_CLONE_KERNEL["full_wav_name"]
        if mode == "clone"
        else OMNIVOICE_DESIGN_KERNEL["full_wav_name"]
    )
    rj_path = out_dir / "result.json"
    result_data = json.loads(rj_path.read_text(encoding="utf-8")) if rj_path.is_file() else {}
    verdict = str(result_data.get("verdict", ""))
    failed_sentences = result_data.get("failed_sentences") or []
    failed_by_idx = {int(f["index"]): f.get("error", "生成失败") for f in failed_sentences if "index" in f}
    verdict_ok = verdict.endswith("OK") and not failed_sentences

    full_path = out_dir / full_name
    full_meta = analyze_wav(full_path) if full_path.is_file() else None
    rms_ok = bool(full_meta and full_meta["rms"] > 0.01)
    sr_ok = bool(full_meta and full_meta["sample_rate"] == 24000)

    infer_map: dict[int, float] = {}
    for s_meta in result_data.get("sentences") or []:
        if isinstance(s_meta, dict) and "index" in s_meta and "infer_sec" in s_meta:
            infer_map[int(s_meta["index"])] = float(s_meta["infer_sec"])

    sentence_items: list[dict[str, Any]] = []
    sum_sentence_dur = 0.0
    succeeded_count = 0
    for idx, text in enumerate(sentences, 1):
        sf = out_dir / f"s{idx}.wav"
        meta = analyze_wav(sf) if sf.is_file() else None
        dur = meta["duration_sec"] if meta else 0.0
        if sf.is_file():
            sum_sentence_dur += dur
            succeeded_count += 1
        sentence_items.append(
            {
                "index": idx,
                "text": text,
                "filename": f"s{idx}.wav" if sf.is_file() else None,
                "duration_sec": dur,
                "sample_rate": meta["sample_rate"] if meta else None,
                "infer_sec": infer_map.get(idx),
                "error": failed_by_idx.get(idx),
            }
        )

    expected_dur = sum_sentence_dur + pause_sec * max(0, succeeded_count - 1)
    actual_dur = full_meta["duration_sec"] if full_meta else 0.0
    duration_diff = round(abs(actual_dur - expected_dur), 3)
    duration_ok = bool(full_meta and duration_diff <= 0.3)

    verdict_detail = verdict or "缺失 result.json"
    if failed_sentences:
        verdict_detail = f"{verdict_detail}（失败 {len(failed_sentences)} 句）"

    checks = {
        "verdict_ok": {"pass": verdict_ok, "detail": verdict_detail},
        "rms_ok": {
            "pass": rms_ok,
            "detail": f"RMS={full_meta['rms']}" if full_meta else "缺失整段音频",
        },
        "sample_rate_ok": {
            "pass": sr_ok,
            "detail": f"{full_meta['sample_rate']} Hz" if full_meta else "未知",
        },
        "duration_ok": {
            "pass": duration_ok,
            "detail": f"实测 {actual_dur:.2f}s vs 理论 {expected_dur:.2f}s (误差 {duration_diff:.2f}s)",
        },
        "quota_ok": {
            "pass": quota_ok,
            "detail": f"耗用 {gpu_seconds_used:.1f}s，reserved 已归 0" if quota_ok else "reserved 未归 0",
        },
    }
    all_passed = all(c["pass"] for c in checks.values())
    summary = {
        "all_passed": all_passed,
        "verdict": verdict,
        "full_wav": full_name if full_path.is_file() else None,
        "full_duration_sec": actual_dur,
        "fallback_used": "FALLBACK" in verdict,
        "fallback_ref_text": result_data.get("fallback_ref_text"),
        "resolved_ref_text": result_data.get("resolved_ref_text"),
        "timings": result_data.get("timings") or {},
        "unsupported_params": result_data.get("unsupported_params") or [],
        "tn_fallback": bool(result_data.get("tn_fallback")),
        "used_saved_prompt": bool(result_data.get("used_saved_prompt")),
        "has_voice_prompt": (out_dir / "voice_prompt.pt").is_file(),
        "failed_sentences": failed_sentences,
        "checks": checks,
        "raw_result": result_data,
    }
    return summary, sentence_items


async def execute_voice_generation(
    *,
    jobs: JobManager,
    client: KaggleClient,
    mode: str,
    sentences: list[str],
    instruct: str | None = None,
    language: str | None = None,
    speed: float | None = 0.85,
    duration: float | None = None,
    num_step: int = 32,
    guidance_scale: float = 2.0,
    denoise: bool = True,
    normalize_text: bool = False,
    postprocess_output: bool = True,
    pause_sec: float | None = None,
    ref_path: Path | None = None,
    ref_display_name: str | None = None,
    ref_text: str | None = None,
    voice_id: str | None = None,
    prompt_path: Path | None = None,
) -> dict[str, Any]:
    cfg = OMNIVOICE_CLONE_KERNEL if mode == "clone" else OMNIVOICE_DESIGN_KERNEL
    slug = cfg["slug"]
    eff_pause = float(cfg["pause_sec"] if pause_sec is None else pause_sec)
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    out_dir = VOICE_OUTPUTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_bytes: bytes | None = None
    ref_ext = ".m4a"
    prompt_bytes: bytes | None = None

    if mode == "clone":
        if prompt_path is not None and prompt_path.is_file():
            prompt_bytes = prompt_path.read_bytes()
        elif ref_path is not None and ref_path.is_file():
            ref_bytes = ref_path.read_bytes()
            ref_ext = ref_path.suffix.lower() or ".m4a"
        else:
            raise ValueError("声音克隆模式下，必须先选择参考音频或声音库中的已存声音。")

    if OMNIVOICE_USE_UNIFIED_KERNEL:
        script_text = render_unified_script(
            mode=mode,
            sentences=sentences,
            instruct=instruct if mode == "design" else None,
            language=language or None,
            speed=speed,
            duration=duration,
            num_step=num_step,
            guidance_scale=guidance_scale,
            denoise=denoise,
            normalize_text=normalize_text,
            postprocess_output=postprocess_output,
            pause_sec=eff_pause,
            full_wav_name=cfg["full_wav_name"],
            ref_ext=ref_ext,
            ref_text=ref_text,
            ref_bytes=ref_bytes,
            prompt_bytes=prompt_bytes,
        )
    elif mode == "design":
        script_text = render_design_script(sentences, instruct or "", speed or 0.85)
    elif mode == "clone" and ref_bytes is not None:
        script_text = render_clone_script(sentences, ref_bytes, speed or 1.0)
    else:
        raise ValueError("旧版模板不支持当前模式或声音库复用，请开启统一内核。")

    acct = get_kaggle_account()
    if not acct or not acct.get("username") or not acct.get("token"):
        raise ValueError("尚未连接 Kaggle 账号，请先填写 Kaggle 用户名与 KGAT_ API Key。")

    job_owner = str(acct["username"])
    job_client = KaggleClient(token=str(acct["token"]), owner=job_owner)
    docker_image = cfg["docker_image"] if acct.get("use_pinned_image") else None

    kind_map = {"design": "语音(设计)", "clone": "语音(克隆)", "auto": "语音(自动)"}
    params_summary = {
        "owner": job_owner,
        "mode": mode,
        "slug": slug,
        "sentence_count": len(sentences),
        "language": language or None,
        "speed": speed,
        "duration": duration,
        "num_step": num_step,
        "guidance_scale": guidance_scale,
        "denoise": denoise,
        "normalize_text": normalize_text,
        "pause_sec": eff_pause,
        "instruct": instruct if mode == "design" else None,
        "ref_name": ref_display_name if mode == "clone" else None,
        "ref_text": ref_text if mode == "clone" else None,
        "voice_id": voice_id,
        "run_id": run_id,
    }

    async def _runner(job: dict[str, Any]) -> None:
        jobs.log(f"[语音-{mode}] 账号 {job_owner} 开始推送内核 {slug}（共 {len(sentences)} 句，lang={language or 'auto'}）")
        jobs.update_job(job["job_id"], state="pushing")
        push_resp = await job_client.push_kernel(
            slug=slug,
            title=cfg["title"],
            code_text=script_text,
            enable_gpu=True,
            enable_internet=True,
            machine_shape=cfg["machine_shape"],
            docker_image=docker_image,
            owner=job_owner,
        )
        ver = push_resp.get("versionNumber")
        jobs.update_job(job["job_id"], state="queued", kaggle_status="queued", version_number=ver)
        jobs.log(f"[语音-{mode}] 推送成功（{job_owner}/{slug} v{ver}），等待 Kaggle 执行...")

        final_kstatus = await jobs.poll_until_terminal(
            job["job_id"],
            slug,
            poll_interval=15,
            max_wait_sec=2400,
            client_override=job_client,
        )
        jobs.update_job(job["job_id"], state="fetching")
        out_data = await job_client.get_kernel_output(slug, owner=job_owner)
        log_lines = clean_kaggle_log(out_data.get("logNullable") or out_data.get("log") or "")
        (out_dir / "kernel.log").write_text("\n".join(log_lines), encoding="utf-8")
        for line in log_lines[-20:]:
            jobs.log(f"[{slug}] {line}", level="error" if "FAIL" in line or "Error" in line else "info")

        for f in out_data.get("files", []):
            fn = f.get("fileName")
            if fn and f.get("url"):
                await job_client.download_file(f["url"], out_dir / fn)

        quota_after = await jobs.reconcile_quota(
            job["job_id"],
            job["quota_before_sec"],
            client_override=job_client,
        )
        gpu_sec = round(max(0.0, quota_after["used_sec"] - job["quota_before_sec"]), 2)
        quota_ok = quota_after["reserved_sec"] == 0

        verification, sentence_items = verify_voice_outputs(
            out_dir=out_dir,
            mode=mode,
            sentences=sentences,
            pause_sec=eff_pause,
            quota_ok=quota_ok,
            gpu_seconds_used=gpu_sec,
        )

        record = {
            "run_id": run_id,
            "owner": job_owner,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "slug": slug,
            "version_number": ver,
            "kaggle_status": final_kstatus,
            "sentences": sentences,
            "instruct": instruct if mode == "design" else None,
            "language": language or None,
            "ref_name": ref_display_name if mode == "clone" else None,
            "ref_text": ref_text if mode == "clone" else None,
            "resolved_ref_text": verification.get("resolved_ref_text"),
            "voice_id": voice_id,
            "speed": speed,
            "duration": duration,
            "num_step": num_step,
            "guidance_scale": guidance_scale,
            "denoise": denoise,
            "normalize_text": normalize_text,
            "pause_sec": eff_pause,
            "gpu_seconds_used": gpu_sec,
            "output_dir": str(out_dir),
            "verification": verification,
            "sentence_items": sentence_items,
            "passed": bool(final_kstatus == "complete" and verification["all_passed"]),
        }
        runs = load_voice_runs()
        runs.insert(0, record)
        save_voice_runs(runs)

        if record["passed"]:
            update_kaggle_account_deploy(
                job_owner,
                deployed=True,
                deploy_info={
                    "slug": slug,
                    "version": ver,
                    "gpu": verification.get("raw_result", {}).get("gpu", ""),
                    "model_load_sec": (verification.get("timings") or {}).get("model_load_sec"),
                    "pip_install_sec": (verification.get("timings") or {}).get("pip_install_sec"),
                    "verdict": verification.get("verdict"),
                    "run_id": run_id,
                },
            )
            jobs._broadcast("accounts_update", {"accounts": list_kaggle_accounts_public()})

        jobs.update_job(
            job["job_id"],
            state="done" if record["passed"] else "failed",
            error=None if record["passed"] else f"校验未全过或内核状态={final_kstatus}",
        )
        jobs._broadcast("voice_completed", record)
        jobs.log(
            f"[语音-{mode}] 完成（{job_owner}）：run_id={run_id}, verdict={verification['verdict']}, "
            f"passed={record['passed']}, gpu={gpu_sec:.1f}s",
            level="info" if record["passed"] else "error",
        )

    return await jobs.start_gpu_job(
        job_id=f"voice-{run_id}",
        kind=kind_map.get(mode, f"语音({mode})"),
        slug=slug,
        params_summary=params_summary,
        runner=_runner,
        owner=job_owner,
        client_override=job_client,
    )


async def probe_remote_omnivoice(username: str) -> dict[str, Any]:
    """Read-only check (0 GPU seconds) whether yoyo-omnivoice already exists on Kaggle."""
    acct = get_kaggle_account(username)
    if not acct or not acct.get("token"):
        raise ValueError(f"未找到账号 {username} 或缺少 API Key。")
    owner = str(acct["username"])
    slug = OMNIVOICE_DESIGN_KERNEL["slug"]
    job_client = KaggleClient(token=str(acct["token"]), owner=owner)
    try:
        pulled = await job_client.pull_kernel(slug, owner=owner)
        meta = pulled.get("metadata") or {}
        ver = meta.get("currentVersionNumber") or meta.get("currentVersionNumberNullable")
        enable_gpu = meta.get("enableGpu") if meta.get("enableGpu") is not None else meta.get("enableGpuNullable")
        if meta.get("id") or ver is not None:
            info = {
                "slug": slug,
                "version": ver,
                "enable_gpu": bool(enable_gpu),
                "source": "remote_detected",
                "last_run_time": meta.get("lastRunTime") or "",
            }
            update_kaggle_account_deploy(owner, deployed=True, deploy_info=info)
            return {"deployed": True, "already_existed": True, "deploy_info": info}
    except Exception:
        pass
    return {"deployed": False, "already_existed": False, "deploy_info": None}


async def deploy_omnivoice_for_account(
    *,
    jobs: JobManager,
    username: str,
    force: bool = False,
) -> dict[str, Any]:
    """Check if OmniVoice is already deployed on this Kaggle account; if not, deploy & smoke-test it."""
    acct = get_kaggle_account(username)
    if not acct or not acct.get("token"):
        raise ValueError(f"未找到账号 {username} 或缺少 API Key。")

    owner = str(acct["username"])
    slug = OMNIVOICE_DESIGN_KERNEL["slug"]

    if not force:
        if acct.get("deployed"):
            jobs.log(f"[部署检查] 账号 {owner} 本地记录已部署 OmniVoice，无需重复部署。")
            return {
                "already_deployed": True,
                "job": None,
                "deploy_info": acct.get("deploy_info"),
                "accounts": list_kaggle_accounts_public(),
            }
        probe = await probe_remote_omnivoice(owner)
        if probe["deployed"]:
            jobs.log(f"[部署检查] 检测到账号 {owner} 的 Kaggle 云端已存在 {slug}，跳过重复部署（0 秒 GPU 消耗）。")
            accounts = list_kaggle_accounts_public()
            jobs._broadcast("accounts_update", {"accounts": accounts})
            return {
                "already_deployed": True,
                "job": None,
                "deploy_info": probe["deploy_info"],
                "accounts": accounts,
            }

    job_client = KaggleClient(token=str(acct["token"]), owner=owner)
    docker_image = OMNIVOICE_DESIGN_KERNEL["docker_image"] if acct.get("use_pinned_image") else None
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    out_dir = VOICE_OUTPUTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    test_sentences = ["你好，OmniVoice 部署成功。"]

    script_text = render_unified_script(
        mode="design",
        sentences=test_sentences,
        instruct="女，青年，中音调",
        language="Chinese",
        speed=0.85,
        duration=None,
        num_step=16,
        guidance_scale=2.0,
        denoise=True,
        normalize_text=False,
        postprocess_output=True,
        pause_sec=0.0,
        full_wav_name=OMNIVOICE_DESIGN_KERNEL["full_wav_name"],
    )

    params_summary = {
        "owner": owner,
        "mode": "design",
        "deploy": True,
        "slug": slug,
        "sentence_count": 1,
        "run_id": run_id,
    }

    async def _runner(job: dict[str, Any]) -> None:
        try:
            jobs.log(f"[部署 OmniVoice] 账号 {owner} 开始部署内核 {slug}（含环境装包、加载 k2-fsa/OmniVoice 与首次出音自检）...")
            jobs.update_job(job["job_id"], state="pushing")
            try:
                push_resp = await job_client.push_kernel(
                    slug=slug,
                    title=OMNIVOICE_DESIGN_KERNEL["title"],
                    code_text=script_text,
                    enable_gpu=True,
                    enable_internet=True,
                    machine_shape=OMNIVOICE_DESIGN_KERNEL["machine_shape"],
                    docker_image=docker_image,
                    owner=owner,
                )
            except Exception as push_exc:
                raise RuntimeError(
                    f"推送到 Kaggle 失败（{push_exc}）。请核对：1) 该账号已完成手机号验证（Phone verification）；"
                    f"2) 用户名「{owner}」与填写的 KGAT_ API Key 属于同一账号。"
                ) from push_exc

            ver = push_resp.get("versionNumber")
            jobs.update_job(job["job_id"], state="queued", kaggle_status="queued", version_number=ver)
            jobs.log(f"[部署 OmniVoice] 推送成功（{owner}/{slug} v{ver}），等待 Kaggle 分配 T4 GPU...")

            final_kstatus = await jobs.poll_until_terminal(
                job["job_id"],
                slug,
                poll_interval=15,
                max_wait_sec=2400,
                client_override=job_client,
            )
            jobs.update_job(job["job_id"], state="fetching")
            out_data = await job_client.get_kernel_output(slug, owner=owner)
            log_lines = clean_kaggle_log(out_data.get("logNullable") or out_data.get("log") or "")
            (out_dir / "kernel.log").write_text("\n".join(log_lines), encoding="utf-8")
            for line in log_lines[-20:]:
                jobs.log(f"[部署-{slug}] {line}", level="error" if "FAIL" in line or "Error" in line else "info")

            for f in out_data.get("files", []):
                fn = f.get("fileName")
                if fn and f.get("url"):
                    await job_client.download_file(f["url"], out_dir / fn)

            quota_after = await jobs.reconcile_quota(
                job["job_id"],
                job["quota_before_sec"],
                client_override=job_client,
            )
            gpu_sec = round(max(0.0, quota_after["used_sec"] - job["quota_before_sec"]), 2)
            quota_ok = quota_after["reserved_sec"] == 0

            verification, sentence_items = verify_voice_outputs(
                out_dir=out_dir,
                mode="design",
                sentences=test_sentences,
                pause_sec=0.0,
                quota_ok=quota_ok,
                gpu_seconds_used=gpu_sec,
            )

            passed = bool(final_kstatus == "complete" and verification["all_passed"])
            record = {
                "run_id": run_id,
                "owner": owner,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "design",
                "slug": slug,
                "version_number": ver,
                "kaggle_status": final_kstatus,
                "sentences": test_sentences,
                "instruct": "女，青年，中音调",
                "language": "Chinese",
                "ref_name": None,
                "ref_text": None,
                "resolved_ref_text": None,
                "voice_id": None,
                "speed": 0.85,
                "duration": None,
                "num_step": 16,
                "guidance_scale": 2.0,
                "denoise": True,
                "normalize_text": False,
                "pause_sec": 0.0,
                "gpu_seconds_used": gpu_sec,
                "output_dir": str(out_dir),
                "verification": verification,
                "sentence_items": sentence_items,
                "passed": passed,
                "is_deploy_probe": True,
            }
            runs = load_voice_runs()
            runs.insert(0, record)
            save_voice_runs(runs)

            raw_res = verification.get("raw_result") or {}
            timings = verification.get("timings") or {}
            if passed:
                deploy_info = {
                    "slug": slug,
                    "version": ver,
                    "gpu": raw_res.get("gpu", ""),
                    "pip_install_sec": timings.get("pip_install_sec"),
                    "model_load_sec": timings.get("model_load_sec"),
                    "total_infer_sec": timings.get("total_infer_sec"),
                    "sample_rate": raw_res.get("sample_rate", 24000),
                    "verdict": verification.get("verdict"),
                    "run_id": run_id,
                    "full_wav": verification.get("full_wav"),
                }
                update_kaggle_account_deploy(owner, deployed=True, deploy_info=deploy_info)
                jobs.update_job(job["job_id"], state="done", error=None)
                accounts = list_kaggle_accounts_public()
                jobs._broadcast("accounts_update", {"accounts": accounts})
                jobs._broadcast("voice_completed", record)
                jobs._broadcast(
                    "deploy_completed",
                    {"ok": True, "username": owner, "deploy_info": deploy_info, "run": record, "accounts": accounts},
                )
                jobs.log(f"[部署 OmniVoice] 账号 {owner} 部署成功！耗时 GPU {gpu_sec:.1f}s，模型加载 {timings.get('model_load_sec')}s。")
            else:
                err_msg = f"部署自检未通过（状态={final_kstatus}, verdict={verification.get('verdict')}）"
                update_kaggle_account_deploy(owner, deployed=False, deploy_info={"error": err_msg, "run_id": run_id})
                jobs.update_job(job["job_id"], state="failed", error=err_msg)
                accounts = list_kaggle_accounts_public()
                jobs._broadcast("accounts_update", {"accounts": accounts})
                jobs._broadcast(
                    "deploy_completed",
                    {"ok": False, "username": owner, "error": err_msg, "accounts": accounts},
                )
                jobs.log(f"[部署 OmniVoice] 账号 {owner} 部署失败：{err_msg}", level="error")
        except Exception as exc:
            err_msg = str(exc)
            update_kaggle_account_deploy(owner, deployed=False, deploy_info={"error": err_msg})
            accounts = list_kaggle_accounts_public()
            jobs._broadcast("accounts_update", {"accounts": accounts})
            jobs._broadcast(
                "deploy_completed",
                {"ok": False, "username": owner, "error": err_msg, "accounts": accounts},
            )
            raise

    job = await jobs.start_gpu_job(
        job_id=f"deploy-{run_id}",
        kind="部署 OmniVoice",
        slug=slug,
        params_summary=params_summary,
        runner=_runner,
        owner=owner,
        client_override=job_client,
    )
    return {
        "already_deployed": False,
        "job": job,
        "deploy_info": None,
        "accounts": list_kaggle_accounts_public(),
    }


# --- Voice Library (P4) -----------------------------------------------------

def list_voice_library() -> list[dict[str, Any]]:
    return load_voice_library()


def get_voice_library_item(voice_id: str) -> dict[str, Any] | None:
    for item in load_voice_library():
        if item.get("voice_id") == voice_id:
            return item
    return None


def save_voice_to_library(
    *,
    name: str,
    run_id: str | None = None,
    mode: str | None = None,
    instruct: str | None = None,
    language: str | None = None,
    speed: float | None = None,
) -> dict[str, Any]:
    clean_name = (name or "").strip()
    if not clean_name:
        raise ValueError("声音名称不能为空。")

    voice_id = f"v-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    lib_dir = VOICE_LIBRARY_DIR / voice_id
    entry: dict[str, Any] = {
        "voice_id": voice_id,
        "name": clean_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if run_id:
        runs = load_voice_runs()
        run = next((r for r in runs if r.get("run_id") == run_id), None)
        if run is None:
            raise ValueError(f"未找到运行记录 {run_id}。")
        rmode = run.get("mode")
        if rmode == "clone":
            src_dir = VOICE_OUTPUTS_DIR / run_id
            src_pt = src_dir / "voice_prompt.pt"
            if not src_pt.is_file():
                raise ValueError("该次克隆任务未产出 voice_prompt.pt，无法存为可复用声线。")
            lib_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_pt, lib_dir / "voice_prompt.pt")
            entry.update(
                {
                    "kind": "clone_prompt",
                    "prompt_file": "voice_prompt.pt",
                    "prompt_size_bytes": src_pt.stat().st_size,
                    "ref_name": run.get("ref_name"),
                    "ref_text": run.get("resolved_ref_text") or run.get("ref_text"),
                    "language": run.get("language"),
                    "speed": run.get("speed"),
                    "source_run_id": run_id,
                }
            )
        elif rmode == "design":
            entry.update(
                {
                    "kind": "design_preset",
                    "instruct": run.get("instruct"),
                    "language": run.get("language"),
                    "speed": run.get("speed"),
                    "source_run_id": run_id,
                }
            )
        else:
            raise ValueError("自动声线模式不支持存为复用声音。")
    else:
        if mode != "design" or not (instruct or "").strip():
            raise ValueError("直接保存声音仅支持声线设计（instruct 不能为空）。")
        entry.update(
            {
                "kind": "design_preset",
                "instruct": instruct.strip(),
                "language": language or None,
                "speed": speed,
            }
        )

    items = load_voice_library()
    items.insert(0, entry)
    save_voice_library(items)
    return entry


def delete_voice_from_library(voice_id: str) -> None:
    items = load_voice_library()
    kept = [x for x in items if x.get("voice_id") != voice_id]
    save_voice_library(kept)
    lib_dir = VOICE_LIBRARY_DIR / voice_id
    if lib_dir.is_dir():
        shutil.rmtree(lib_dir, ignore_errors=True)


# --- SRT Subtitle Export (P5) -----------------------------------------------

def _fmt_srt_time(sec: float) -> str:
    ms_total = max(0, int(round(sec * 1000)))
    hrs = ms_total // 3_600_000
    rem = ms_total % 3_600_000
    mins = rem // 60_000
    rem %= 60_000
    secs = rem // 1000
    ms = rem % 1000
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def build_srt_content(run: dict[str, Any]) -> str:
    raw_pause = run.get("pause_sec")
    if raw_pause is None:
        raw_pause = OMNIVOICE_CLONE_KERNEL["pause_sec"] if run.get("mode") == "clone" else OMNIVOICE_DESIGN_KERNEL["pause_sec"]
    pause_sec = float(raw_pause)
    items = [s for s in (run.get("sentence_items") or []) if s.get("filename")]
    lines: list[str] = []
    cursor = 0.0
    for idx, item in enumerate(items, 1):
        dur = float(item.get("duration_sec") or 0.0)
        start = cursor
        end = cursor + dur
        lines.append(str(idx))
        lines.append(f"{_fmt_srt_time(start)} --> {_fmt_srt_time(end)}")
        lines.append(str(item.get("text") or "").strip())
        lines.append("")
        cursor = end + pause_sec
    return "\n".join(lines)
