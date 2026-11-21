"""Kaggle GPU kernel for Video Clone Stage 2: CosyVoice 2 (CosyVoice2-0.5B) voice cloning synthesis."""

import base64
import ctypes
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

os.environ["HF_HOME"] = "/tmp/hf_cache"

WORK = Path("/kaggle/working")
TMP = Path("/tmp/vclone_tts_work")
TMP.mkdir(parents=True, exist_ok=True)

# __VCLONE_TTS_PARAMS_BEGIN__
PARAMS = json.loads(__PARAMS_JSON_STR__)
REF_EXT = PARAMS.get("ref_ext", ".wav")
REF_B64_PARTS = PARAMS.get("ref_b64_parts", [])
REF_TEXT = PARAMS.get("ref_text", "")
SEGMENTS = PARAMS.get("segments", [])
COSY_MODE = PARAMS.get("cosy_mode", "zero_shot")
COSY_INSTRUCT = PARAMS.get("cosy_instruct") or ""
COSY_SPEED = float(PARAMS.get("cosy_speed", 1.0) or 1.0)
# __VCLONE_TTS_PARAMS_END__

timings = {}
result = {
    "platform": "kaggle",
    "job": "yoyo-vclone-tts",
    "cosy_mode": COSY_MODE,
    "cosy_instruct": COSY_INSTRUCT,
    "cosy_speed": COSY_SPEED,
    "ref_text": REF_TEXT,
    "used_cached_env": False,
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


def preload_nvidia_libs():
    try:
        import site
        for root in [Path(p) for p in site.getsitepackages()]:
            for pattern in ("nvidia/*/lib/*.so*", "torch/lib/*.so*"):
                for so_path in sorted(root.glob(pattern)):
                    if any(k in so_path.name for k in ("libcudnn", "libcublas", "libnvrtc", "libcufft")):
                        try:
                            ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
                        except Exception:
                            pass
    except Exception:
        pass


# 1) GPU probe
smi = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
result["gpu"] = (smi.stdout or smi.stderr).strip()

# 2) Discover mounted yoyo-vclone-env cache (supports v2 .tar archives & v1 unpacked layout)
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

    tts_tar_rel = env_manifest.get("tts_tar_rel")
    if tts_tar_rel and (env_root / tts_tar_rel).is_file():
        tts_cache_root = TMP / "env_tts"
        tts_cache_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(env_root / tts_tar_rel, "r") as tf:
            tf.extractall(tts_cache_root)
    else:
        tts_cache_root = env_root
    timings["untar_env_sec"] = round(time.time() - t_untar, 2)

# 3) Install CosyVoice 2 runtime dependencies
t0 = time.time()
needed_pkgs = [
    "onnxruntime-gpu", "soundfile", "huggingface_hub", "modelscope",
    "HyperPyYAML", "conformer", "diffusers", "hydra-core", "omegaconf",
    "inflect", "unidecode", "eng_to_ipa", "wetext", "einops",
]

if prep_cache_root and env_manifest:
    wheels_dir = prep_cache_root / env_manifest.get("wheels_rel", "vclone_cache/wheels")
    if wheels_dir.is_dir():
        r_off = run([
            sys.executable, "-m", "pip", "install", "-q",
            "--no-index", "--find-links", str(wheels_dir),
        ] + needed_pkgs)
        if r_off.returncode != 0:
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
import torch

result["torch_cuda"] = bool(torch.cuda.is_available())
try:
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    result["ort_providers"] = ort.get_available_providers()
except Exception as ort_exc:
    result["ort_providers"] = [f"err:{ort_exc}"]
print("CosyVoice2 GPU check -> torch_cuda:", result["torch_cuda"], "ort_providers:", result["ort_providers"], flush=True)

# 4) Decode reference audio clip
if not REF_B64_PARTS:
    fail("MISSING_REF_AUDIO")
if not SEGMENTS:
    fail("MISSING_SEGMENTS")

raw_ref = base64.b64decode("".join(REF_B64_PARTS))
ext = REF_EXT if REF_EXT.startswith(".") else f".{REF_EXT}"
ref_in = TMP / f"ref_in{ext}"
ref_in.write_bytes(raw_ref)

ref_16k_wav = TMP / "ref_16k.wav"
r_ff = run(["ffmpeg", "-y", "-i", str(ref_in), "-ac", "1", "-ar", "16000", str(ref_16k_wav)])
if r_ff.returncode != 0 or not ref_16k_wav.is_file():
    fail("REF_FFMPEG_FAIL", stderr=(r_ff.stderr or "")[-1500:])

# 5) Setup CosyVoice repo & text normalizer fallback
t0 = time.time()
cosy_repo = TMP / "CosyVoice"
if tts_cache_root and env_manifest:
    cached_repo = tts_cache_root / env_manifest.get("cosyvoice_repo_rel", "vclone_cache/CosyVoice")
    if (cached_repo / "cosyvoice").is_dir():
        cosy_repo = cached_repo

if not (cosy_repo / "cosyvoice").is_dir():
    r_git = run([
        "git", "clone", "--depth", "1", "--recursive",
        "https://github.com/FunAudioLLM/CosyVoice.git",
        str(cosy_repo),
    ])
    if r_git.returncode != 0:
        fail("COSYVOICE_GIT_FAIL", stderr=(r_git.stderr or "")[-1500:])

sys.path.insert(0, str(cosy_repo))
sys.path.insert(0, str(cosy_repo / "third_party" / "Matcha-TTS"))

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

# 6) Load CosyVoice2-0.5B model
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

try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    cosy = CosyVoice2(str(cosy_model_dir), load_jit=False, load_trt=False, fp16=True)
except Exception as load_exc:
    fail("COSYVOICE_LOAD_FAIL", error=str(load_exc))

cosy_sr = int(getattr(cosy, "sample_rate", 24000))
result["sample_rate"] = cosy_sr
timings["model_load_sec"] = round(time.time() - t0, 1)
print(f"CosyVoice2-0.5B loaded in {timings['model_load_sec']}s (sr={cosy_sr})", flush=True)

prompt_speech_16k = load_wav(str(ref_16k_wav), 16000)


def synthesize_one(text):
    for prompt_arg in (prompt_speech_16k, str(ref_16k_wav)):
        try:
            if COSY_MODE == "instruct2" and COSY_INSTRUCT:
                gen = cosy.inference_instruct2(text, COSY_INSTRUCT, prompt_arg, stream=False, speed=COSY_SPEED)
            elif COSY_MODE == "cross_lingual":
                gen = cosy.inference_cross_lingual(text, prompt_arg, stream=False, speed=COSY_SPEED)
            else:
                gen = cosy.inference_zero_shot(text, REF_TEXT or "你好", prompt_arg, stream=False, speed=COSY_SPEED)
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


# 7) Synthesize each segment
t_all0 = time.time()
all_parts = []
sil = np.zeros(int(cosy_sr * 0.25), dtype=np.float32)

for item in SEGMENTS:
    idx = int(item["index"])
    txt = (item.get("text") or "").strip()
    if not txt:
        continue
    t_s0 = time.time()
    try:
        wav_arr = synthesize_one(txt)
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
        all_parts.append(wav_arr)
        print(f"tts_s{idx} done: dur={dur}s, infer={dt}s", flush=True)
    except Exception as exc:
        dt = round(time.time() - t_s0, 2)
        result["failed_tts"].append({"index": idx, "text": txt, "error": str(exc)})
        print(f"tts_s{idx} FAIL ({dt}s): {exc}", flush=True)

timings["total_infer_sec"] = round(time.time() - t_all0, 2)

if not all_parts:
    fail("TTS_FAIL_ALL")

concat_parts = []
for i, p in enumerate(all_parts):
    concat_parts.append(p)
    if i < len(all_parts) - 1:
        concat_parts.append(sil)
full_wav = np.concatenate(concat_parts)
sf.write(str(WORK / "tts_full.wav"), full_wav, cosy_sr)
result["full_duration_sec"] = round(len(full_wav) / cosy_sr, 3)
result["verdict"] = "TTS_OK" if not result["failed_tts"] else "TTS_PARTIAL_OK"
write_result()
print("DONE", result["verdict"], flush=True)
