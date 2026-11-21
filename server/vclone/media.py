"""Local FFmpeg media standardization, audio extraction, segment slicing, and probing."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

from server.config import VCLONE_MAX_DURATION_SEC


def check_ffmpeg() -> dict[str, Any]:
    try:
        r = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        first_line = ((r.stdout or r.stderr or "").splitlines() or [""])[0]
        return {"ok": r.returncode == 0, "version": first_line}
    except Exception as exc:
        return {"ok": False, "version": "", "error": str(exc)}


def probe_media(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"ok": False, "error": "文件不存在"}
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration,size:stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        data = json.loads(r.stdout or "{}")
        fmt = data.get("format") or {}
        streams = data.get("streams") or []
        v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
        dur = round(float(fmt.get("duration") or 0.0), 3)
        return {
            "ok": True,
            "filename": path.name,
            "size_bytes": int(fmt.get("size") or path.stat().st_size),
            "duration_sec": dur,
            "has_video": v_stream is not None,
            "has_audio": a_stream is not None,
            "width": int(v_stream.get("width") or 0) if v_stream else 0,
            "height": int(v_stream.get("height") or 0) if v_stream else 0,
            "video_codec": (v_stream or {}).get("codec_name") or "",
            "audio_codec": (a_stream or {}).get("codec_name") or "",
            "sample_rate": int((a_stream or {}).get("sample_rate") or 0),
            "channels": int((a_stream or {}).get("channels") or 0),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _sync_standardize_and_extract(
    raw_video: Path,
    out_dir: Path,
    *,
    clip_start: float = 0.0,
    clip_end: float | None = None,
    max_duration: float = VCLONE_MAX_DURATION_SEC,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = probe_media(raw_video)
    if not meta.get("ok"):
        raise ValueError(f"无法读取原视频信息：{meta.get('error')}")
    if not meta.get("has_audio"):
        raise ValueError("该视频不包含音轨，无法提取人声进行声音克隆。")

    src_dur = float(meta.get("duration_sec") or 0.0)
    start_sec = max(0.0, float(clip_start or 0.0))
    if src_dur > 0 and start_sec >= src_dur:
        raise ValueError(f"裁剪起始时间 ({start_sec}s) 超出视频总时长 ({src_dur}s)。")

    if clip_end is not None and float(clip_end) > start_sec:
        target_dur = float(clip_end) - start_sec
    elif src_dur > 0:
        target_dur = src_dur - start_sec
    else:
        target_dur = max_duration

    if target_dur > max_duration + 0.5:
        target_dur = max_duration

    video_std = out_dir / "video_std.mp4"
    raw_audio = out_dir / "raw_audio.wav"
    upload_audio = out_dir / "upload_audio.flac"

    # 1) Standardize video to 25fps H.264 + AAC (handles VFR, AV1, HEVC, odd dimensions)
    v_cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{target_dur:.3f}",
        "-i", str(raw_video),
        "-vf", "fps=25,scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        "-movflags", "+faststart",
        str(video_std),
    ]
    r_v = subprocess.run(v_cmd, capture_output=True, text=True, timeout=300)
    if r_v.returncode != 0 or not video_std.is_file():
        raise ValueError(f"FFmpeg 视频标准化失败：{(r_v.stderr or '')[-800:]}")

    # 2) Extract 44.1kHz stereo WAV from video_std.mp4
    a_cmd = [
        "ffmpeg", "-y",
        "-i", str(video_std),
        "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le",
        str(raw_audio),
    ]
    r_a = subprocess.run(a_cmd, capture_output=True, text=True, timeout=120)
    if r_a.returncode != 0 or not raw_audio.is_file():
        raise ValueError(f"FFmpeg 抽取音轨失败：{(r_a.stderr or '')[-800:]}")

    # 3) Produce lossless FLAC for Kaggle dataset mount (zero quality loss, bypasses 1MB script limit),
    # plus a compact Opus fallback (< 620 KB) in case inline push is ever needed.
    std_meta = probe_media(video_std)
    eff_dur = max(5.0, float(std_meta.get("duration_sec") or target_dur))
    upload_flac = out_dir / "upload_audio.flac"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(raw_audio),
            "-ac", "2", "-ar", "44100", "-c:a", "flac",
            str(upload_flac),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    target_kbps = min(96, max(24, int(4_600_000 / eff_dur / 1000)))
    upload_ogg = out_dir / "upload_audio.ogg"
    r_op = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(raw_audio),
            "-ac", "2", "-ar", "48000", "-c:a", "libopus", "-b:a", f"{target_kbps}k",
            str(upload_ogg),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r_op.returncode != 0 or not upload_ogg.is_file() or upload_ogg.stat().st_size > 680 * 1024:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(raw_audio),
                "-ac", "1", "-ar", "24000", "-c:a", "libopus", "-b:a", "24k",
                str(upload_ogg),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

    primary_audio = upload_flac if upload_flac.is_file() else upload_ogg
    return {
        "video_std": video_std.name,
        "raw_audio": raw_audio.name,
        "upload_audio": primary_audio.name,
        "upload_ext": primary_audio.suffix.lower(),
        "upload_bytes": primary_audio.stat().st_size,
        "fallback_ogg": upload_ogg.name if upload_ogg.is_file() else None,
        "clip_start": round(start_sec, 2),
        "clip_duration": round(float(std_meta.get("duration_sec") or target_dur), 3),
        "width": std_meta.get("width", 0),
        "height": std_meta.get("height", 0),
    }


async def standardize_and_extract(
    raw_video: Path,
    out_dir: Path,
    *,
    clip_start: float = 0.0,
    clip_end: float | None = None,
    max_duration: float = VCLONE_MAX_DURATION_SEC,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _sync_standardize_and_extract,
        raw_video,
        out_dir,
        clip_start=clip_start,
        clip_end=clip_end,
        max_duration=max_duration,
    )


def _sync_slice_segment_clips(vocals_wav: Path, segments: list[dict[str, Any]], out_dir: Path) -> None:
    if not vocals_wav.is_file():
        return
    for seg in segments:
        idx = int(seg["index"])
        start = max(0.0, float(seg.get("start") or 0.0))
        end = max(start + 0.1, float(seg.get("end") or (start + 1.0)))
        dur = end - start
        dest = out_dir / f"orig_s{idx}.wav"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-t", f"{dur:.3f}",
                "-i", str(vocals_wav),
                "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le",
                str(dest),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if dest.is_file():
            seg["orig_clip"] = dest.name


async def slice_segment_clips(vocals_wav: Path, segments: list[dict[str, Any]], out_dir: Path) -> None:
    await asyncio.to_thread(_sync_slice_segment_clips, vocals_wav, segments, out_dir)


def extract_custom_ref_clip(vocals_wav: Path, start_sec: float, end_sec: float, dest_wav: Path) -> dict[str, Any]:
    if not vocals_wav.is_file():
        raise ValueError("纯净人声文件 vocals.wav 不存在。")
    s0 = max(0.0, float(start_sec))
    s1 = float(end_sec)
    if s1 <= s0 + 0.5:
        raise ValueError("参考音片段时长至少需要 0.5 秒（建议 3~12 秒）。")
    dur = min(15.0, s1 - s0)
    r = subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{s0:.3f}",
            "-t", f"{dur:.3f}",
            "-i", str(vocals_wav),
            "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le",
            str(dest_wav),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if r.returncode != 0 or not dest_wav.is_file():
        raise ValueError(f"截取参考音失败：{(r.stderr or '')[-500:]}")
    return {"start": round(s0, 3), "end": round(s0 + dur, 3), "duration": round(dur, 3), "filename": dest_wav.name}


def fmt_srt_timestamp(sec: float) -> str:
    ms_total = max(0, int(round(sec * 1000)))
    hrs = ms_total // 3_600_000
    rem = ms_total % 3_600_000
    mins = rem // 60_000
    rem %= 60_000
    secs = rem // 1000
    ms = rem % 1000
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def build_segments_srt(segments: list[dict[str, Any]], *, use_orig: bool = False, bilingual: bool = False) -> str:
    blocks: list[str] = []
    for i, seg in enumerate(segments, 1):
        start = float(seg.get("aligned_start") if not use_orig and seg.get("aligned_start") is not None else seg.get("start", 0.0))
        end = float(seg.get("aligned_end") if not use_orig and seg.get("aligned_end") is not None else seg.get("end", start + 1.0))
        orig_t = (seg.get("orig_text") or "").strip()
        new_t = (seg.get("text") or orig_t).strip()
        if bilingual and orig_t and new_t and orig_t != new_t:
            line = f"{new_t}\n{orig_t}"
        else:
            line = orig_t if use_orig else new_t
        if not line:
            continue
        blocks.append(f"{i}\n{fmt_srt_timestamp(start)} --> {fmt_srt_timestamp(end)}\n{line}\n")
    return "\n".join(blocks)
