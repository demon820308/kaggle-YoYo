"""Kaggle CPU kernel to build and cache yoyo-vclone-env (wheels + UVR5 + Faster-Whisper + CosyVoice2-0.5B).
Consumes 0 GPU quota (runs with enableGpu=False). Output in /kaggle/working is mounted via kernelDataSources.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

os.environ["HF_HOME"] = "/tmp/hf_cache"

WORK = Path("/kaggle/working")
CACHE = WORK / "vclone_cache"
WHEELS_DIR = CACHE / "wheels"
MODELS_DIR = CACHE / "models"
UVR_DIR = MODELS_DIR / "uvr"
WHISPER_DIR = MODELS_DIR / "whisper_large_v3_turbo"
COSY_MODEL_DIR = MODELS_DIR / "CosyVoice2-0.5B"
COSY_REPO_DIR = CACHE / "CosyVoice"

# __VCLONE_ENV_PARAMS_BEGIN__
PARAMS = json.loads(__PARAMS_JSON_STR__)
INCLUDE_ROFORMER = bool(PARAMS.get("include_roformer", False))
# __VCLONE_ENV_PARAMS_END__

timings = {}
result = {
    "platform": "kaggle",
    "job": "yoyo-vclone-env",
    "python_version": sys.version.split()[0],
    "include_roformer": INCLUDE_ROFORMER,
    "timings": timings,
    "cached_items": {},
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
    sys.exit(0)


for d in (WHEELS_DIR, UVR_DIR, WHISPER_DIR, COSY_MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# 1) Download lightweight wheels for fast offline installation on GPU kernels
t0 = time.time()
pkgs = [
    "faster-whisper",
    "ctranslate2",
    "onnxruntime-gpu",
    "audio-separator",
    "rotary-embedding-torch",
    "beartype",
    "julius",
    "ml-collections",
    "HyperPyYAML",
    "conformer",
    "diffusers",
    "hydra-core",
    "omegaconf",
    "inflect",
    "unidecode",
    "eng_to_ipa",
    "wetext",
    "modelscope",
    "huggingface_hub",
    "soundfile",
    "einops",
]
# Download wheels without re-downloading torch/torchaudio/torchvision/scipy/numpy already in Kaggle image
dl_cmd = [
    sys.executable, "-m", "pip", "download", "-q",
    "-d", str(WHEELS_DIR),
] + pkgs
r_dl = run(dl_cmd)
if r_dl.returncode != 0:
    print("WARN: batch pip download had non-zero exit, downloading package-by-package...", flush=True)
    for p in pkgs:
        run([sys.executable, "-m", "pip", "download", "-q", "-d", str(WHEELS_DIR), p])

# Remove huge torch/triton/nvidia wheels if pip pulled them into WHEELS_DIR, because Kaggle GPU already has them
for whl in list(WHEELS_DIR.glob("*.whl")):
    low = whl.name.lower()
    if low.startswith(("torch-", "torchaudio-", "torchvision-", "triton-", "nvidia_")):
        whl.unlink(missing_ok=True)

wheel_files = list(WHEELS_DIR.glob("*"))
timings["wheels_download_sec"] = round(time.time() - t0, 1)
result["cached_items"]["wheels_count"] = len(wheel_files)
print(f"Cached {len(wheel_files)} wheels in {timings['wheels_download_sec']}s", flush=True)

# 2) Clone CosyVoice repo + Matcha-TTS submodule
t0 = time.time()
if COSY_REPO_DIR.exists():
    shutil.rmtree(COSY_REPO_DIR, ignore_errors=True)
r_git = run([
    "git", "clone", "--depth", "1", "--recursive",
    "https://github.com/FunAudioLLM/CosyVoice.git",
    str(COSY_REPO_DIR),
])
if r_git.returncode != 0:
    fail("GIT_CLONE_COSYVOICE_FAIL", stderr=(r_git.stderr or "")[-1500:])
# Strip .git metadata to save space and file count
for git_dir in list(COSY_REPO_DIR.rglob(".git")):
    if git_dir.is_dir():
        shutil.rmtree(git_dir, ignore_errors=True)
    elif git_dir.is_file():
        git_dir.unlink(missing_ok=True)
timings["cosyvoice_repo_sec"] = round(time.time() - t0, 1)
result["cached_items"]["cosyvoice_repo"] = True

# 3) Download UVR5 model(s)
t0 = time.time()
uvr_urls = [
    (
        "UVR-MDX-NET-Inst_HQ_3.onnx",
        [
            "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx",
            "https://huggingface.co/KimberleyJSN/audio-separator-models/resolve/main/UVR-MDX-NET-Inst_HQ_3.onnx",
        ],
    ),
]
if INCLUDE_ROFORMER:
    uvr_urls.extend([
        (
            "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            [
                "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            ],
        ),
        (
            "model_bs_roformer_ep_317_sdr_12.9755.yaml",
            [
                "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_317_sdr_12.9755.yaml",
            ],
        ),
    ])

for fname, urls in uvr_urls:
    dest = UVR_DIR / fname
    ok = False
    for u in urls:
        try:
            print(f"Downloading {fname} from {u} ...", flush=True)
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as fp:
                shutil.copyfileobj(resp, fp)
            if dest.stat().st_size > 1000:
                ok = True
                break
        except Exception as exc:
            print(f"WARN: {u} failed: {exc}", flush=True)
    if not ok and fname == "UVR-MDX-NET-Inst_HQ_3.onnx":
        fail("UVR_DOWNLOAD_FAIL", file=fname)

timings["uvr_download_sec"] = round(time.time() - t0, 1)
result["cached_items"]["uvr_files"] = [p.name for p in UVR_DIR.iterdir() if p.is_file()]

# 4) Download Faster-Whisper large-v3-turbo CT2 weights
t0 = time.time()
run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "modelscope"])
from huggingface_hub import snapshot_download

try:
    snapshot_download(
        repo_id="deepdml/faster-whisper-large-v3-turbo-ct2",
        local_dir=str(WHISPER_DIR),
        local_dir_use_symlinks=False,
    )
    result["cached_items"]["whisper_model"] = "deepdml/faster-whisper-large-v3-turbo-ct2"
except Exception as exc:
    fail("WHISPER_DOWNLOAD_FAIL", error=str(exc))
timings["whisper_download_sec"] = round(time.time() - t0, 1)

# 5) Download FunAudioLLM/CosyVoice2-0.5B weights
t0 = time.time()
cosy_ok = False
try:
    snapshot_download(
        repo_id="FunAudioLLM/CosyVoice2-0.5B",
        local_dir=str(COSY_MODEL_DIR),
        local_dir_use_symlinks=False,
    )
    cosy_ok = (COSY_MODEL_DIR / "cosyvoice2.yaml").is_file() or (COSY_MODEL_DIR / "cosyvoice.yaml").is_file()
except Exception as exc:
    print(f"HF download of CosyVoice2-0.5B failed ({exc}), trying ModelScope...", flush=True)

if not cosy_ok:
    try:
        from modelscope import snapshot_download as ms_download
        ms_download("iic/CosyVoice2-0.5B", local_dir=str(COSY_MODEL_DIR))
        cosy_ok = True
    except Exception as ms_exc:
        fail("COSYVOICE_MODEL_DOWNLOAD_FAIL", error=str(ms_exc))

timings["cosyvoice_model_sec"] = round(time.time() - t0, 1)
result["cached_items"]["cosyvoice_model"] = "CosyVoice2-0.5B"

# Write manifest for GPU kernels to discover automatically under /kaggle/input
manifest = {
    "version": 1,
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "python_version": sys.version.split()[0],
    "wheels_rel": "vclone_cache/wheels",
    "cosyvoice_repo_rel": "vclone_cache/CosyVoice",
    "uvr_dir_rel": "vclone_cache/models/uvr",
    "whisper_dir_rel": "vclone_cache/models/whisper_large_v3_turbo",
    "cosyvoice_model_rel": "vclone_cache/models/CosyVoice2-0.5B",
}
(WORK / "vclone_env_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)

total_bytes = sum(p.stat().st_size for p in CACHE.rglob("*") if p.is_file())
result["total_cache_gb"] = round(total_bytes / (1024 ** 3), 2)
result["verdict"] = "ENV_OK"
write_result()
print("DONE ENV_OK", flush=True)
