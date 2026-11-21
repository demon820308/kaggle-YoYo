"""Kaggle GPU kernel for Video Clone Stage 1 (audio-separator UVR5 + Faster-Whisper ASR)
and optional One-Click Stage 2 (CosyVoice 2 zero-shot / cross-lingual / instruct synthesis).
"""

import base64
import ctypes
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
import types
import urllib.request
from pathlib import Path

os.environ["HF_HOME"] = "/tmp/hf_cache"

WORK = Path("/kaggle/working")
TMP = Path("/tmp/vclone_work")
TMP.mkdir(parents=True, exist_ok=True)

# __VCLONE_PREP_PARAMS_BEGIN__
PARAMS = json.loads(__PARAMS_JSON_STR__)
AUDIO_EXT = PARAMS.get("audio_ext", ".flac")
AUDIO_B64_PARTS = PARAMS.get("audio_b64_parts", [])
INPUT_FILENAME = PARAMS.get("input_filename") or ""
SEPARATOR_MODEL = PARAMS.get("separator_model", "UVR-MDX-NET-Inst_HQ_3.onnx")
WHISPER_MODEL = PARAMS.get("whisper_model", "large-v3-turbo")
SOURCE_LANGUAGE = PARAMS.get("source_language") or None
ONE_CLICK_TTS = bool(PARAMS.get("one_click_tts", False))
COSY_MODE = PARAMS.get("cosy_mode", "zero_shot")
COSY_INSTRUCT = PARAMS.get("cosy_instruct") or ""
COSY_SPEED = float(PARAMS.get("cosy_speed", 1.0) or 1.0)
# __VCLONE_PREP_PARAMS_END__

timings = {}
result = {
    "platform": "kaggle",
    "job": "yoyo-vclone-prep",
    "separator_model": SEPARATOR_MODEL,
    "whisper_model": WHISPER_MODEL,
    "source_language": SOURCE_LANGUAGE,
    "one_click_tts": ONE_CLICK_TTS,
    "used_cached_env": False,
    "segments": [],
    "tts_sentences": [],
    "failed_tts": [],
    "timings": timings,
}


def run(cmd, cwd=None):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)


def write_result():
    (WORK / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("RESULT_BEGIN", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("RESULT_END", flush=True)


def fail(verdict, **extra):
    result["verdict"] = verdict
    result.update(extra)
    write_result()
    print("DONE", verdict, flush=True)
    sys.exit(0)


def get_nvidia_lib_dirs():
    import site
    dirs = []
    for root in [Path(p) for p in site.getsitepackages()]:
        for sub in root.glob("nvidia/*/lib"):
            if sub.is_dir():
                dirs.append(str(sub))
        tlib = root / "torch" / "lib"
        if tlib.is_dir():
            dirs.append(str(tlib))
    for sys_dir in ("/usr/local/cuda/lib64", "/usr/local/nvidia/lib64"):
        if Path(sys_dir).is_dir():
            dirs.append(sys_dir)
    return dirs


def preload_nvidia_libs():
    """Preload bundled cuDNN/cuBLAS from PyTorch/nvidia packages so CTranslate2 and ONNXRuntime find them."""
    try:
        import site
        for root in [Path(p) for p in site.getsitepackages()]:
            for pattern in ("nvidia/*/lib/*.so*", "torch/lib/*.so*"):
                for so_path in sorted(root.glob(pattern)):
                    name = so_path.name
                    if any(k in name for k in ("libcudnn", "libcublas", "libnvrtc", "libcufft", "libcudart")):
                        try:
                            ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
                        except Exception:
                            pass
    except Exception as exc:
        print(f"WARN: preload_nvidia_libs: {exc}", flush=True)


# 1) GPU probe
smi = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
result["gpu"] = (smi.stdout or smi.stderr).strip()

# 2) Discover mounted yoyo-vclone-env cache if available (supports v2 .tar archives & v1 unpacked layout)
env_root = None
env_manifest = None
prep_cache_root = None
tts_cache_root = None
input_root = Path("/kaggle/input")
if input_root.is_dir():
    for mf in input_root.rglob("vclone_env_manifest.json"):
        try:
            env_manifest = json.loads(mf.read_text(encoding="utf-8"))
            env_root = mf.parent
            result["used_cached_env"] = True
            print(f"Found cached vclone-env v{env_manifest.get('version', 1)} at {env_root}", flush=True)
            break
        except Exception:
            pass

if env_root and env_manifest:
    import tarfile
    t_untar = time.time()
    prep_tar_rel = env_manifest.get("prep_tar_rel")
    if prep_tar_rel and (env_root / prep_tar_rel).is_file():
        prep_cache_root = TMP / "env_prep"
        prep_cache_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(env_root / prep_tar_rel, "r") as tf:
            tf.extractall(prep_cache_root)
    else:
        prep_cache_root = env_root

    if ONE_CLICK_TTS:
        tts_tar_rel = env_manifest.get("tts_tar_rel")
        if tts_tar_rel and (env_root / tts_tar_rel).is_file():
            tts_cache_root = TMP / "env_tts"
            tts_cache_root.mkdir(parents=True, exist_ok=True)
            with tarfile.open(env_root / tts_tar_rel, "r") as tf:
                tf.extractall(tts_cache_root)
        else:
            tts_cache_root = env_root
    timings["untar_env_sec"] = round(time.time() - t_untar, 2)

# 3) Install packages (offline wheels first if cached, then online fallback)
t0 = time.time()
needed_pkgs = ["faster-whisper", "audio-separator", "onnxruntime-gpu", "soundfile", "huggingface_hub"]
if ONE_CLICK_TTS:
    needed_pkgs.extend([
        "HyperPyYAML", "conformer", "diffusers", "hydra-core", "omegaconf",
        "inflect", "unidecode", "eng_to_ipa", "wetext", "modelscope", "einops",
    ])

if prep_cache_root and env_manifest:
    wheels_dir = prep_cache_root / env_manifest.get("wheels_rel", "vclone_cache/wheels")
    if wheels_dir.is_dir():
        r_off = run([
            sys.executable, "-m", "pip", "install", "-q",
            "--no-index", "--find-links", str(wheels_dir),
        ] + needed_pkgs)
        if r_off.returncode != 0:
            print("Offline wheel install partial, supplementing online...", flush=True)
            run([sys.executable, "-m", "pip", "install", "-q"] + needed_pkgs)
    else:
        run([sys.executable, "-m", "pip", "install", "-q"] + needed_pkgs)
else:
    r_on = run([sys.executable, "-m", "pip", "install", "-q"] + needed_pkgs)
    if r_on.returncode != 0:
        fail("PIP_INSTALL_FAIL", stderr=(r_on.stderr or "")[-2000:])

timings["pip_install_sec"] = round(time.time() - t0, 1)
preload_nvidia_libs()

import numpy as np
import soundfile as sf

# 4) Locate mounted input audio (dataset) or decode inline base64 audio, then convert to 44.1kHz WAV
t0 = time.time()
raw_in = None
if INPUT_FILENAME and input_root.is_dir():
    for cand in input_root.rglob(INPUT_FILENAME):
        if cand.is_file():
            raw_in = cand
            print(f"Found mounted input audio: {raw_in} ({raw_in.stat().st_size} bytes)", flush=True)
            break

if raw_in is None:
    if not AUDIO_B64_PARTS:
        fail("MISSING_INPUT_AUDIO")
    raw_bytes = base64.b64decode("".join(AUDIO_B64_PARTS))
    ext = AUDIO_EXT if AUDIO_EXT.startswith(".") else f".{AUDIO_EXT}"
    raw_in = TMP / f"input{ext}"
    raw_in.write_bytes(raw_bytes)

input_wav = TMP / "input_44k.wav"
r_ff = run(["ffmpeg", "-y", "-i", str(raw_in), "-ac", "2", "-ar", "44100", str(input_wav)])
if r_ff.returncode != 0 or not input_wav.is_file():
    fail("FFMPEG_DECODE_FAIL", stderr=(r_ff.stderr or "")[-1500:])

in_info = sf.info(str(input_wav))
result["input_duration_sec"] = round(in_info.frames / in_info.samplerate, 3)
timings["decode_sec"] = round(time.time() - t0, 2)

# 5) Run audio-separator (UVR5) to split vocals & BGM
t0 = time.time()
uvr_model_dir = TMP / "uvr_models"
uvr_model_dir.mkdir(parents=True, exist_ok=True)

if prep_cache_root and env_manifest:
    cached_uvr = prep_cache_root / env_manifest.get("uvr_dir_rel", "vclone_cache/models/uvr")
    if cached_uvr.is_dir():
        for f in cached_uvr.iterdir():
            if f.is_file():
                target = uvr_model_dir / f.name
                if not target.exists():
                    try:
                        os.symlink(f, target)
                    except Exception:
                        shutil.copy2(f, target)

model_target = uvr_model_dir / SEPARATOR_MODEL
if not model_target.exists() and SEPARATOR_MODEL == "UVR-MDX-NET-Inst_HQ_3.onnx":
    for mirror in (
        "https://huggingface.co/KimberleyJSN/audio-separator-models/resolve/main/UVR-MDX-NET-Inst_HQ_3.onnx",
        "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx",
    ):
        try:
            print(f"Downloading {SEPARATOR_MODEL} from {mirror} ...", flush=True)
            req = urllib.request.Request(mirror, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as resp, open(model_target, "wb") as fp:
                shutil.copyfileobj(resp, fp)
            if model_target.stat().st_size > 1000:
                break
        except Exception as exc:
            print(f"WARN: mirror {mirror} failed: {exc}", flush=True)

sep_out_dir = TMP / "sep_out"
sep_out_dir.mkdir(parents=True, exist_ok=True)

# Run audio-separator in a fresh subprocess with LD_LIBRARY_PATH pointing to torch/nvidia CUDA libs
# so onnxruntime-gpu reliably loads CUDAExecutionProvider (~10s on T4 instead of ~155s CPU fallback).
sep_runner_script = TMP / "_run_sep.py"
sep_runner_script.write_text(
    f"""import json, os, sys
try:
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    print("ORT providers:", ort.get_available_providers(), flush=True)
except Exception as e:
    print("ORT probe warn:", e, flush=True)
from audio_separator.separator import Separator
sep = Separator(
    output_dir={repr(str(sep_out_dir))},
    model_file_dir={repr(str(uvr_model_dir))},
    output_format="WAV",
)
sep.load_model(model_filename={repr(SEPARATOR_MODEL)})
files = sep.separate({repr(str(input_wav))})
print("SEP_FILES_JSON=" + json.dumps([str(x) for x in files]), flush=True)
""",
    encoding="utf-8",
)

sep_env = os.environ.copy()
lib_dirs = get_nvidia_lib_dirs()
if lib_dirs:
    existing_ld = sep_env.get("LD_LIBRARY_PATH", "")
    sep_env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing_ld] if existing_ld else []))

r_sep = subprocess.run([sys.executable, str(sep_runner_script)], env=sep_env, capture_output=True, text=True)
print((r_sep.stdout or "")[-1200:], flush=True)
if r_sep.stderr:
    print((r_sep.stderr or "")[-800:], flush=True)

out_files = []
for line in (r_sep.stdout or "").splitlines():
    if line.startswith("ORT providers:"):
        result["ort_providers"] = line.split(":", 1)[1].strip()
    elif line.startswith("SEP_FILES_JSON="):
        out_files = json.loads(line.split("=", 1)[1])

if r_sep.returncode != 0 or not out_files:
    fail("SEPARATOR_FAIL", error=(r_sep.stderr or r_sep.stdout or "Separator subprocess failed")[-1500:])

vocal_raw = None
bgm_raw = None
for f in out_files:
    p = Path(f) if Path(f).is_absolute() else (sep_out_dir / Path(f).name)
    if not p.is_file():
        candidates = list(sep_out_dir.glob(f"*{Path(f).name}*"))
        if candidates:
            p = candidates[0]
    low = p.name.lower()
    if "(vocals)" in low or ("vocal" in low and "no_vocal" not in low and "inst" not in low):
        vocal_raw = p
    elif "(instrumental)" in low or "inst" in low or "no_vocal" in low or "other" in low:
        bgm_raw = p

if vocal_raw is None:
    all_wavs = sorted(sep_out_dir.glob("*.wav"))
    for p in all_wavs:
        if "vocal" in p.name.lower() and "inst" not in p.name.lower():
            vocal_raw = p
        else:
            bgm_raw = p

if vocal_raw is None or not vocal_raw.is_file():
    fail("SEPARATOR_OUTPUT_MISSING", files=[str(x) for x in out_files])

vocals_wav = WORK / "vocals.wav"
bgm_wav = WORK / "bgm.wav"

# Convert vocals to 24kHz mono 16-bit PCM WAV (compact & clean for Whisper + CosyVoice 2)
run(["ffmpeg", "-y", "-i", str(vocal_raw), "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(vocals_wav)])
if bgm_raw and bgm_raw.is_file():
    run(["ffmpeg", "-y", "-i", str(bgm_raw), "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(bgm_wav)])
else:
    # Create silent BGM if source had no instrumental track
    run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", str(result["input_duration_sec"]), "-c:a", "pcm_s16le", str(bgm_wav),
    ])

timings["separator_sec"] = round(time.time() - t0, 1)
print(f"UVR separation done in {timings['separator_sec']}s", flush=True)

# 6) Faster-Whisper ASR on vocals.wav
t0 = time.time()
whisper_path_or_id = WHISPER_MODEL
if WHISPER_MODEL == "large-v3-turbo":
    whisper_path_or_id = "deepdml/faster-whisper-large-v3-turbo-ct2"
    if prep_cache_root and env_manifest:
        cached_w = prep_cache_root / env_manifest.get("whisper_dir_rel", "vclone_cache/models/whisper_large_v3_turbo")
        if (cached_w / "model.bin").is_file():
            whisper_path_or_id = str(cached_w)

raw_segments = []
detected_lang = SOURCE_LANGUAGE or "zh"
try:
    # Convert vocals to 16kHz mono float32 numpy array directly via ffmpeg+soundfile
    # so faster-whisper bypasses PyAV (av.open metadata_errors incompatibility in some images)
    vocals_16k = TMP / "vocals_16k.wav"
    r_w16 = run(["ffmpeg", "-y", "-i", str(vocals_wav), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(vocals_16k)])
    if r_w16.returncode != 0 or not vocals_16k.is_file():
        raise RuntimeError(f"ffmpeg 16k convert failed: {(r_w16.stderr or '')[-500:]}")
    w_audio, _w_sr = sf.read(str(vocals_16k), dtype="float32")
    if w_audio.ndim > 1:
        w_audio = w_audio.mean(axis=-1)
    w_audio = np.ascontiguousarray(w_audio, dtype=np.float32)

    from faster_whisper import WhisperModel
    try:
        w_model = WhisperModel(whisper_path_or_id, device="cuda", compute_type="float16")
    except Exception as cuda_exc:
        print(f"WARN: Faster-Whisper float16 CUDA init failed ({cuda_exc}), trying cpu int8...", flush=True)
        w_model = WhisperModel(whisper_path_or_id, device="cpu", compute_type="int8")

    segs_iter, info = w_model.transcribe(
        w_audio,
        language=SOURCE_LANGUAGE,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200),
        word_timestamps=True,
    )
    detected_lang = getattr(info, "language", None) or detected_lang
    for seg in segs_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        words = []
        for w in (getattr(seg, "words", None) or []):
            words.append({"start": round(float(w.start), 3), "end": round(float(w.end), 3), "word": w.word})
        raw_segments.append({
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "text": text,
            "words": words,
        })
    del w_model
except Exception as w_exc:
    fail("WHISPER_FAIL", error=str(w_exc))

# Split overly long segments (> 10s or > 42 chars) at punctuation using word timestamps
PUNCT = set("。！？!?；;，,")
final_segments = []
for seg in raw_segments:
    dur = seg["end"] - seg["start"]
    words = seg.get("words") or []
    if (dur <= 10.0 and len(seg["text"]) <= 42) or len(words) < 4:
        final_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "orig_text": seg["text"],
            "text": seg["text"],
        })
        continue
    cur_words = []
    cur_start = words[0]["start"]
    for i, w in enumerate(words):
        cur_words.append(w)
        w_txt = (w["word"] or "").strip()
        cur_dur = w["end"] - cur_start
        cur_len = sum(len(x["word"]) for x in cur_words)
        is_last = i == len(words) - 1
        has_punct = any(ch in PUNCT for ch in w_txt)
        if is_last or (has_punct and (cur_dur >= 2.2 or cur_len >= 14)) or cur_dur >= 9.5 or cur_len >= 38:
            chunk_txt = "".join(x["word"] for x in cur_words).strip()
            if chunk_txt:
                final_segments.append({
                    "start": round(cur_start, 3),
                    "end": round(w["end"], 3),
                    "orig_text": chunk_txt,
                    "text": chunk_txt,
                })
            if not is_last:
                cur_words = []
                cur_start = words[i + 1]["start"]

for idx, s in enumerate(final_segments, 1):
    s["index"] = idx
    s["duration"] = round(max(0.1, s["end"] - s["start"]), 3)

result["detected_language"] = detected_lang
result["segments"] = final_segments
timings["whisper_sec"] = round(time.time() - t0, 1)
print(f"Faster-Whisper recognized {len(final_segments)} segments in {timings['whisper_sec']}s (lang={detected_lang})", flush=True)

if not final_segments:
    fail("WHISPER_NO_SPEECH")

# 7) Auto-select best 3.5~11s reference vocal clip from vocals.wav
voc_data, voc_sr = sf.read(str(vocals_wav), dtype="float32")
if voc_data.ndim > 1:
    voc_data = voc_data.mean(axis=-1)

best_ref = None
best_score = -1.0

# Consider single segments and adjacent 2-segment pairs to hit the 4.0~10.0s sweet spot
candidates = []
for s in final_segments:
    candidates.append((s["start"], s["end"], s["orig_text"], [s["index"]]))
for i in range(len(final_segments) - 1):
    s1, s2 = final_segments[i], final_segments[i + 1]
    if s2["start"] - s1["end"] <= 0.8:
        join_sep = " " if detected_lang == "en" else "，"
        candidates.append((s1["start"], s2["end"], f"{s1['orig_text']}{join_sep}{s2['orig_text']}", [s1["index"], s2["index"]]))

for c_start, c_end, c_text, c_idxs in candidates:
    c_dur = round(c_end - c_start, 3)
    if c_dur < 1.5 or c_dur > 14.0:
        continue
    i0 = max(0, int(c_start * voc_sr))
    i1 = min(len(voc_data), int(c_end * voc_sr))
    clip = voc_data[i0:i1]
    if len(clip) < voc_sr:
        continue
    rms = float(np.sqrt(np.mean(clip ** 2)))
    if rms < 0.005:
        continue
    # Sweet spot around 5.0 - 8.5 seconds
    dur_bonus = 1.0 if 3.5 <= c_dur <= 9.5 else (0.6 if 2.5 <= c_dur <= 12.0 else 0.3)
    score = dur_bonus * min(rms, 0.25) * min(len(c_text), 40)
    if score > best_score:
        best_score = score
        best_ref = {
            "start": round(c_start, 3),
            "end": round(c_end, 3),
            "duration": c_dur,
            "ref_text": c_text,
            "segment_indices": c_idxs,
            "rms": round(rms, 4),
        }

if best_ref is None:
    s0 = final_segments[0]
    best_ref = {
        "start": s0["start"],
        "end": min(s0["end"], s0["start"] + 10.0),
        "duration": round(min(s0["duration"], 10.0), 3),
        "ref_text": s0["orig_text"],
        "segment_indices": [s0["index"]],
        "rms": 0.05,
    }

ref_clip_wav = WORK / "ref_clip.wav"
r_i0 = max(0, int(best_ref["start"] * voc_sr))
r_i1 = min(len(voc_data), int(best_ref["end"] * voc_sr))
sf.write(str(ref_clip_wav), voc_data[r_i0:r_i1], voc_sr)
result["best_ref"] = best_ref
print(f"Selected reference clip: {best_ref['start']}s~{best_ref['end']}s ({best_ref['duration']}s): {best_ref['ref_text']}", flush=True)

# 8) Optional One-Click CosyVoice 2 TTS inside the same GPU kernel
if ONE_CLICK_TTS:
    t_tts0 = time.time()
    import torch
    torch.cuda.empty_cache()
    result["torch_cuda"] = bool(torch.cuda.is_available())
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        result["ort_providers_tts"] = ort.get_available_providers()
    except Exception as ort_exc:
        result["ort_providers_tts"] = [f"err:{ort_exc}"]
    print("One-click CosyVoice2 GPU check -> torch_cuda:", result["torch_cuda"], "ort_providers:", result["ort_providers_tts"], flush=True)

    # Locate or clone CosyVoice repo
    cosy_repo = TMP / "CosyVoice"
    if tts_cache_root and env_manifest:
        cached_repo = tts_cache_root / env_manifest.get("cosyvoice_repo_rel", "vclone_cache/CosyVoice")
        if cached_repo.is_dir():
            cosy_repo = cached_repo
    if not (cosy_repo / "cosyvoice").is_dir():
        run([
            "git", "clone", "--depth", "1", "--recursive",
            "https://github.com/FunAudioLLM/CosyVoice.git",
            str(cosy_repo),
        ])

    sys.path.insert(0, str(cosy_repo))
    sys.path.insert(0, str(cosy_repo / "third_party" / "Matcha-TTS"))

    # Ensure text normalizer fallback if pynini/tn is not installed
    try:
        from tn.chinese.normalizer import Normalizer as _ZhNorm  # noqa: F401
    except Exception:
        class _FallbackNormalizer:
            def __init__(self, *args, **kwargs):
                self._wt = None
                try:
                    from wetext import Normalizer as WtNorm
                    self._wt = WtNorm()
                except Exception:
                    pass

            def normalize(self, text):
                if self._wt is not None:
                    try:
                        return self._wt.normalize(text)
                    except Exception:
                        pass
                return text

        for mod_name in ("tn", "tn.chinese", "tn.chinese.normalizer", "tn.english", "tn.english.normalizer"):
            if mod_name not in sys.modules:
                sys.modules[mod_name] = types.ModuleType(mod_name)
        sys.modules["tn.chinese.normalizer"].Normalizer = _FallbackNormalizer
        sys.modules["tn.english.normalizer"].Normalizer = _FallbackNormalizer

    # Locate or download CosyVoice2-0.5B weights
    cosy_model_dir = TMP / "CosyVoice2-0.5B"
    if tts_cache_root and env_manifest:
        cached_cm = tts_cache_root / env_manifest.get("cosyvoice_model_rel", "vclone_cache/models/CosyVoice2-0.5B")
        if (cached_cm / "cosyvoice2.yaml").is_file() or (cached_cm / "cosyvoice.yaml").is_file():
            cosy_model_dir = cached_cm

    if not ((cosy_model_dir / "cosyvoice2.yaml").is_file() or (cosy_model_dir / "cosyvoice.yaml").is_file()):
        from huggingface_hub import snapshot_download
        try:
            snapshot_download("FunAudioLLM/CosyVoice2-0.5B", local_dir=str(cosy_model_dir), local_dir_use_symlinks=False)
        except Exception:
            from modelscope import snapshot_download as ms_download
            ms_download("iic/CosyVoice2-0.5B", local_dir=str(cosy_model_dir))

    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    cosy = CosyVoice2(str(cosy_model_dir), load_jit=False, load_trt=False, fp16=True)
    cosy_sr = int(getattr(cosy, "sample_rate", 24000))
    result["tts_sample_rate"] = cosy_sr

    ref_16k_wav = TMP / "ref_16k.wav"
    run(["ffmpeg", "-y", "-i", str(ref_clip_wav), "-ac", "1", "-ar", "16000", str(ref_16k_wav)])
    prompt_speech_16k = load_wav(str(ref_16k_wav), 16000)

    def run_cosy_segment(text):
        for prompt_arg in (prompt_speech_16k, str(ref_16k_wav)):
            try:
                if COSY_MODE == "instruct2" and COSY_INSTRUCT:
                    gen = cosy.inference_instruct2(text, COSY_INSTRUCT, prompt_arg, stream=False, speed=COSY_SPEED)
                elif COSY_MODE == "cross_lingual":
                    gen = cosy.inference_cross_lingual(text, prompt_arg, stream=False, speed=COSY_SPEED)
                else:
                    gen = cosy.inference_zero_shot(text, best_ref["ref_text"], prompt_arg, stream=False, speed=COSY_SPEED)
                chunks = []
                for item in gen:
                    sp = item["tts_speech"]
                    if hasattr(sp, "detach"):
                        sp = sp.detach().cpu().numpy()
                    chunks.append(np.asarray(sp, dtype=np.float32).squeeze())
                if chunks:
                    return np.concatenate(chunks, axis=-1) if len(chunks) > 1 else chunks[0]
            except TypeError:
                continue
        raise RuntimeError("CosyVoice2 inference returned no audio chunks")

    for seg in final_segments:
        idx = seg["index"]
        txt = seg["text"]
        t_s0 = time.time()
        try:
            wav_arr = run_cosy_segment(txt)
            out_wav = WORK / f"tts_s{idx}.wav"
            sf.write(str(out_wav), wav_arr, cosy_sr)
            dur = round(len(wav_arr) / cosy_sr, 3)
            dt = round(time.time() - t_s0, 2)
            result["tts_sentences"].append({
                "index": idx,
                "text": txt,
                "duration": dur,
                "infer_sec": dt,
                "filename": f"tts_s{idx}.wav",
            })
            print(f"tts_s{idx} done: dur={dur}s, infer={dt}s", flush=True)
        except Exception as exc:
            result["failed_tts"].append({"index": idx, "text": txt, "error": str(exc)})
            print(f"tts_s{idx} FAIL: {exc}", flush=True)

    timings["one_click_tts_sec"] = round(time.time() - t_tts0, 1)

result["verdict"] = "PREP_AND_TTS_OK" if ONE_CLICK_TTS else "PREP_OK"
write_result()
print("DONE", result["verdict"], flush=True)
