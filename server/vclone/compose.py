"""Timeline alignment, vocal+BGM mixing, SRT export, and final MP4 muxing using local FFmpeg."""

from __future__ import annotations

import array
import asyncio
import subprocess
import wave
from pathlib import Path
from typing import Any

from server.vclone.media import build_segments_srt, probe_media

MASTER_SR = 24000


def _read_wav_mono_s16(path: Path) -> tuple[array.array[int], float]:
    with wave.open(str(path), "rb") as wf:
        nframes = wf.getnframes()
        sr = wf.getframerate()
        raw = wf.readframes(nframes)
    samples = array.array("h")
    samples.frombytes(raw)
    dur = round(len(samples) / float(sr or MASTER_SR), 3)
    return samples, dur


def _write_wav_mono_s16(path: Path, samples: array.array[int], sr: int = MASTER_SR) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def _sync_compose_cloned_video(
    out_dir: Path,
    segments: list[dict[str, Any]],
    *,
    auto_align: bool = True,
    max_atempo: float = 1.35,
    min_atempo: float = 0.88,
    vocal_volume: float = 1.0,
    bgm_volume: float = 0.65,
    burn_subtitles: bool = False,
    bilingual_subtitles: bool = False,
) -> dict[str, Any]:
    video_std = out_dir / "video_std.mp4"
    if not video_std.is_file():
        raise ValueError("未找到标准化原视频 video_std.mp4。")

    v_meta = probe_media(video_std)
    video_dur = float(v_meta.get("duration_sec") or 0.0)
    if video_dur <= 0:
        video_dur = max((float(s.get("end") or 0.0) for s in segments), default=10.0) + 1.0

    total_samples = int((video_dur + 15.0) * MASTER_SR)
    master_pcm = array.array("h", [0] * total_samples)

    cursor_sec = 0.0
    aligned_count = 0
    stretched_count = 0
    overflow_count = 0

    for i, seg in enumerate(segments):
        idx = int(seg["index"])
        tts_wav = out_dir / f"tts_s{idx}.wav"
        if not tts_wav.is_file():
            seg["align_status"] = "missing"
            continue

        orig_start = max(0.0, float(seg.get("start") or 0.0))
        orig_end = max(orig_start + 0.15, float(seg.get("end") or (orig_start + 1.0)))
        orig_dur = round(orig_end - orig_start, 3)

        if i + 1 < len(segments):
            next_start = max(orig_end, float(segments[i + 1].get("start") or orig_end))
            avail_dur = max(orig_dur, round(next_start - orig_start - 0.03, 3))
        else:
            avail_dur = max(orig_dur, round(video_dur - orig_start, 3))

        # Convert raw tts_s{idx}.wav to 24kHz mono 16-bit first to measure exact duration
        norm_tts = out_dir / f"_norm_tts_s{idx}.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(tts_wav),
                "-ac", "1", "-ar", str(MASTER_SR), "-c:a", "pcm_s16le",
                str(norm_tts),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if not norm_tts.is_file():
            seg["align_status"] = "error"
            continue

        _, tts_dur = _read_wav_mono_s16(norm_tts)
        seg["tts_duration"] = tts_dur

        ratio = 1.0
        status = "ok"
        if auto_align and tts_dur > 0.1:
            if tts_dur > avail_dur:
                needed = tts_dur / max(0.15, avail_dur)
                if needed > max_atempo:
                    ratio = max_atempo
                    status = "overflow"
                    overflow_count += 1
                else:
                    ratio = needed
                    status = "stretched"
                    stretched_count += 1
            elif tts_dur > orig_dur * 1.08:
                # Fits in silence gap before next sentence, apply gentle speedup halfway toward orig_dur
                target_d = min(avail_dur, (orig_dur + min(avail_dur, tts_dur)) * 0.5)
                ratio = min(max_atempo, max(1.0, tts_dur / max(0.15, target_d)))
                if ratio > 1.04:
                    status = "stretched"
                    stretched_count += 1
            elif tts_dur < orig_dur * 0.72 and orig_dur >= 1.6:
                ratio = max(min_atempo, tts_dur / (orig_dur * 0.82))
                if ratio < 0.96:
                    status = "stretched"
                    stretched_count += 1

        aligned_wav = out_dir / f"aligned_s{idx}.wav"
        if abs(ratio - 1.0) >= 0.03:
            r_at = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(norm_tts),
                    "-af", f"atempo={ratio:.3f}",
                    "-ac", "1", "-ar", str(MASTER_SR), "-c:a", "pcm_s16le",
                    str(aligned_wav),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if r_at.returncode != 0 or not aligned_wav.is_file():
                aligned_wav = norm_tts
                ratio = 1.0
        else:
            aligned_wav = norm_tts
            ratio = 1.0

        samples, aligned_dur = _read_wav_mono_s16(aligned_wav)
        norm_tts.unlink(missing_ok=True)

        place_start = max(orig_start, cursor_sec)
        start_sample = int(round(place_start * MASTER_SR))
        end_sample = min(len(master_pcm), start_sample + len(samples))
        copy_len = max(0, end_sample - start_sample)
        if copy_len > 0:
            master_pcm[start_sample:end_sample] = samples[:copy_len]

        place_end = round(place_start + aligned_dur, 3)
        cursor_sec = place_end + 0.02
        aligned_count += 1

        seg["aligned_start"] = round(place_start, 3)
        seg["aligned_end"] = place_end
        seg["aligned_duration"] = aligned_dur
        seg["atempo_ratio"] = round(ratio, 2)
        seg["align_status"] = status
        seg["tts_clip"] = f"tts_s{idx}.wav"

    # Trim master_pcm to max(video_dur, cursor_sec)
    final_audio_dur = max(video_dur, min(video_dur + 3.0, cursor_sec))
    keep_samples = min(len(master_pcm), int(round(final_audio_dur * MASTER_SR)))
    new_vocals_wav = out_dir / "new_vocals_aligned.wav"
    _write_wav_mono_s16(new_vocals_wav, master_pcm[:keep_samples], MASTER_SR)

    # Mix new_vocals_aligned.wav with bgm.wav
    bgm_wav = out_dir / "bgm.wav"
    final_mixed_wav = out_dir / "final_mixed_audio.wav"
    v_vol = max(0.1, min(2.5, float(vocal_volume)))
    b_vol = max(0.0, min(2.0, float(bgm_volume)))

    if bgm_wav.is_file() and b_vol > 0.005:
        filter_str = (
            f"[0:a]aresample=44100,pan=stereo|c0=c0|c1=c0,volume={v_vol:.2f}[v];"
            f"[1:a]aresample=44100,volume={b_vol:.2f}[b];"
            f"[v][b]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
            f"alimiter=limit=0.95[out]"
        )
        r_mix = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(new_vocals_wav),
                "-i", str(bgm_wav),
                "-filter_complex", filter_str,
                "-map", "[out]",
                "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le",
                str(final_mixed_wav),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r_mix.returncode != 0 or not final_mixed_wav.is_file():
            raise ValueError(f"人声与伴奏混音失败：{(r_mix.stderr or '')[-800:]}")
    else:
        r_mix = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(new_vocals_wav),
                "-af", f"aresample=44100,pan=stereo|c0=c0|c1=c0,volume={v_vol:.2f}",
                "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le",
                str(final_mixed_wav),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r_mix.returncode != 0 or not final_mixed_wav.is_file():
            raise ValueError(f"人声导出失败：{(r_mix.stderr or '')[-800:]}")

    # Generate SRT files
    origin_srt = out_dir / "origin.srt"
    cloned_srt = out_dir / "cloned.srt"
    bilingual_srt = out_dir / "bilingual.srt"
    origin_srt.write_text(build_segments_srt(segments, use_orig=True), encoding="utf-8")
    cloned_srt.write_text(build_segments_srt(segments, use_orig=False, bilingual=False), encoding="utf-8")
    bilingual_srt.write_text(build_segments_srt(segments, use_orig=False, bilingual=True), encoding="utf-8")

    # Mux final_cloned.mp4
    final_mp4 = out_dir / "final_cloned.mp4"
    sub_file_name = "bilingual.srt" if bilingual_subtitles else "cloned.srt"

    if burn_subtitles and (out_dir / sub_file_name).is_file():
        # Run with cwd=out_dir so relative path subtitles=cloned.srt avoids Windows drive colon issues
        mux_cmd = [
            "ffmpeg", "-y",
            "-i", "video_std.mp4",
            "-i", "final_mixed_audio.wav",
            "-vf", f"subtitles={sub_file_name}:force_style='FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1,Shadow=0,MarginV=18'",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart",
            "final_cloned.mp4",
        ]
        r_mux = subprocess.run(mux_cmd, cwd=str(out_dir), capture_output=True, text=True, timeout=300)
        if r_mux.returncode != 0 or not final_mp4.is_file():
            # Fallback to stream copy without burning subtitles if font/libass fails
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(video_std),
                    "-i", str(final_mixed_wav),
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart",
                    str(final_mp4),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
    else:
        r_mux = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_std),
                "-i", str(final_mixed_wav),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest", "-movflags", "+faststart",
                str(final_mp4),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r_mux.returncode != 0 or not final_mp4.is_file():
            raise ValueError(f"合成最终视频失败：{(r_mux.stderr or '')[-800:]}")

    return {
        "final_mp4": final_mp4.name,
        "new_vocals_wav": new_vocals_wav.name,
        "final_mixed_wav": final_mixed_wav.name,
        "origin_srt": origin_srt.name,
        "cloned_srt": cloned_srt.name,
        "bilingual_srt": bilingual_srt.name,
        "video_duration_sec": round(video_dur, 2),
        "aligned_count": aligned_count,
        "stretched_count": stretched_count,
        "overflow_count": overflow_count,
        "auto_align": auto_align,
        "vocal_volume": v_vol,
        "bgm_volume": b_vol,
        "burn_subtitles": burn_subtitles,
    }


async def compose_cloned_video(
    out_dir: Path,
    segments: list[dict[str, Any]],
    *,
    auto_align: bool = True,
    max_atempo: float = 1.35,
    min_atempo: float = 0.88,
    vocal_volume: float = 1.0,
    bgm_volume: float = 0.65,
    burn_subtitles: bool = False,
    bilingual_subtitles: bool = False,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _sync_compose_cloned_video,
        out_dir,
        segments,
        auto_align=auto_align,
        max_atempo=max_atempo,
        min_atempo=min_atempo,
        vocal_volume=vocal_volume,
        bgm_volume=bgm_volume,
        burn_subtitles=burn_subtitles,
        bilingual_subtitles=bilingual_subtitles,
    )
