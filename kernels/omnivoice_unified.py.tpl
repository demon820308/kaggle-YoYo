"""Unified OmniVoice Kaggle kernel (design / clone / auto, single model load)."""

import base64
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
SR_FALLBACK = 24000

# __OMNIVOICE_PARAMS_BEGIN__
PARAMS = json.loads(__PARAMS_JSON_STR__)
MODE = PARAMS["mode"]
SENTENCES = PARAMS["sentences"]
INSTRUCT = PARAMS["instruct"]
LANGUAGE = PARAMS["language"]
SPEED = PARAMS["speed"]
DURATION = PARAMS["duration"]
NUM_STEP = PARAMS["num_step"]
GUIDANCE_SCALE = PARAMS["guidance_scale"]
DENOISE = PARAMS["denoise"]
NORMALIZE_TEXT = PARAMS["normalize_text"]
POSTPROCESS_OUTPUT = PARAMS["postprocess_output"]
PAUSE_SEC = PARAMS["pause_sec"]
FULL_WAV_NAME = PARAMS["full_wav_name"]
REF_EXT = PARAMS["ref_ext"]
REF_TEXT = PARAMS["ref_text"]
REF_B64_PARTS = PARAMS["ref_b64_parts"]
PROMPT_B64_PARTS = PARAMS["prompt_b64_parts"]
# __OMNIVOICE_PARAMS_END__

timings = {}
result = {
    "platform": "kaggle",
    "job": "omnivoice-unified",
    "mode": MODE,
    "language": LANGUAGE,
    "instruct": INSTRUCT if MODE == "design" else None,
    "speed": SPEED,
    "duration": DURATION,
    "num_step": NUM_STEP,
    "guidance_scale": GUIDANCE_SCALE,
    "denoise": DENOISE,
    "normalize_text": NORMALIZE_TEXT,
    "postprocess_output": POSTPROCESS_OUTPUT,
    "pause_sec": PAUSE_SEC,
    "sentences": [],
    "failed_sentences": [],
    "unsupported_params": [],
    "tn_fallback": False,
    "timings": timings,
}


def run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True)


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


# 1) GPU probe
smi = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
result["gpu"] = (smi.stdout or smi.stderr).strip()

# 2) Pip install (with optional [tn] fallback)
t0 = time.time()
pkg_spec = "omnivoice[tn]" if NORMALIZE_TEXT else "omnivoice"
inst = run([sys.executable, "-m", "pip", "install", "-q", pkg_spec, "soundfile"])
if inst.returncode != 0 and NORMALIZE_TEXT:
    print("WARN: omnivoice[tn] install failed, falling back to omnivoice without [tn]", flush=True)
    print((inst.stderr or "")[-1000:], flush=True)
    result["tn_fallback"] = True
    NORMALIZE_TEXT = False
    result["normalize_text"] = False
    inst = run([sys.executable, "-m", "pip", "install", "-q", "omnivoice", "soundfile"])

timings["pip_install_sec"] = round(time.time() - t0, 1)
print((inst.stdout or "")[-1200:], flush=True)
print((inst.stderr or "")[-1200:], flush=True)
if inst.returncode != 0:
    fail("PIP_INSTALL_FAIL", stderr=(inst.stderr or "")[-2000:])

import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoice

# 3) Load model once
t0 = time.time()
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16,
)
timings["model_load_sec"] = round(time.time() - t0, 1)
SR = int(getattr(model, "sampling_rate", SR_FALLBACK))
result["sample_rate"] = SR
print(f"Model loaded in {timings['model_load_sec']}s, sampling_rate={SR}", flush=True)

# Probe supported kwargs
gen_sig = inspect.signature(model.generate)
has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in gen_sig.parameters.values())
explicit_params = set(gen_sig.parameters.keys())
config_fields = set()
try:
    from omnivoice import OmniVoiceGenerationConfig
    config_fields = set(getattr(OmniVoiceGenerationConfig, "__dataclass_fields__", {}).keys())
except Exception:
    pass
supported_params = explicit_params | (config_fields if has_var_kw else set())


def put_kw(kw, name, val):
    if val is None:
        return
    if name in supported_params:
        kw[name] = val
    else:
        result["unsupported_params"].append(name)


# 4) Clone mode prompt preparation (either saved .pt or raw audio)
voice_prompt = None
fallback_used = False
ref_wav_path = WORK / "ref.wav"

if MODE == "clone":
    t_prompt0 = time.time()
    if PROMPT_B64_PARTS:
        from omnivoice import VoiceClonePrompt

        prompt_path = WORK / "input_prompt.pt"
        prompt_path.write_bytes(base64.b64decode("".join(PROMPT_B64_PARTS)))
        voice_prompt = VoiceClonePrompt.load(str(prompt_path))
        result["used_saved_prompt"] = True
        result["resolved_ref_text"] = getattr(voice_prompt, "ref_text", None)
        print("Loaded saved VoiceClonePrompt, skipped ffmpeg & Whisper.", flush=True)
    elif REF_B64_PARTS:
        raw = base64.b64decode("".join(REF_B64_PARTS))
        ext = REF_EXT if REF_EXT.startswith(".") else f".{REF_EXT}"
        ref_in = WORK / f"ref_in{ext}"
        ref_in.write_bytes(raw)
        r = run(["ffmpeg", "-y", "-i", ref_in, "-ac", "1", "-ar", str(SR), ref_wav_path])
        if r.returncode != 0 or not ref_wav_path.exists():
            fail("REF_CONVERT_FAIL", stderr=(r.stderr or "")[-2000:])

        info = sf.info(str(ref_wav_path))
        result["ref_wav"] = {
            "duration": round(info.frames / info.samplerate, 3),
            "sample_rate": info.samplerate,
        }
        print("ref_wav:", result["ref_wav"], flush=True)

        if hasattr(model, "create_voice_clone_prompt"):
            try:
                voice_prompt = model.create_voice_clone_prompt(
                    ref_audio=str(ref_wav_path),
                    ref_text=(REF_TEXT or None),
                )
            except Exception as exc:
                if REF_TEXT:
                    fail("CLONE_PROMPT_FAIL", error=str(exc))
                print(f"Auto-ASR prompt creation failed ({exc}), trying Whisper-small fallback...", flush=True)
                r_w = run([sys.executable, "-m", "pip", "install", "-q", "openai-whisper"])
                if r_w.returncode != 0:
                    fail("CLONE_FAIL", stage="whisper_pip_install", stderr=(r_w.stderr or "")[-1500:])
                try:
                    import whisper
                    w_model = whisper.load_model("small")
                    tr = w_model.transcribe(str(ref_wav_path))
                    fb_text = (tr.get("text") or "").strip()
                except Exception as w_exc:
                    fail("CLONE_FAIL", stage="whisper_transcribe", error=str(w_exc))
                if not fb_text:
                    fail("CLONE_FAIL", stage="whisper_empty_text")
                fallback_used = True
                result["fallback_ref_text"] = fb_text
                voice_prompt = model.create_voice_clone_prompt(
                    ref_audio=str(ref_wav_path),
                    ref_text=fb_text,
                )

            result["resolved_ref_text"] = getattr(voice_prompt, "ref_text", None) or REF_TEXT
            try:
                voice_prompt.save(str(WORK / "voice_prompt.pt"))
                result["saved_voice_prompt"] = True
            except Exception as save_exc:
                print(f"WARN: could not save voice_prompt.pt: {save_exc}", flush=True)
                result["saved_voice_prompt"] = False
    else:
        fail("CLONE_MISSING_REF")
    timings["prompt_encode_sec"] = round(time.time() - t_prompt0, 1)

# 5) Shared generation kwargs
base_kw = {}
if LANGUAGE:
    put_kw(base_kw, "language", LANGUAGE)

if DURATION is not None and float(DURATION) > 0:
    put_kw(base_kw, "duration", float(DURATION))
elif SPEED is not None:
    put_kw(base_kw, "speed", float(SPEED))

put_kw(base_kw, "num_step", int(NUM_STEP))
put_kw(base_kw, "guidance_scale", float(GUIDANCE_SCALE))
put_kw(base_kw, "denoise", bool(DENOISE))
put_kw(base_kw, "postprocess_output", bool(POSTPROCESS_OUTPUT))
if NORMALIZE_TEXT:
    put_kw(base_kw, "normalize_text", True)

if MODE == "design":
    put_kw(base_kw, "instruct", INSTRUCT)
elif MODE == "clone":
    if voice_prompt is not None and "voice_clone_prompt" in explicit_params:
        base_kw["voice_clone_prompt"] = voice_prompt
    else:
        base_kw["ref_audio"] = str(ref_wav_path)
        if REF_TEXT or result.get("fallback_ref_text"):
            base_kw["ref_text"] = REF_TEXT or result.get("fallback_ref_text")

result["effective_params"] = {
    k: ("<VoiceClonePrompt>" if k == "voice_clone_prompt" else v) for k, v in base_kw.items()
}

# 6) Generate per sentence using single loaded model
infer_secs = []
audio_secs = []
succeeded_parts = []
sil = np.zeros(int(SR * float(PAUSE_SEC)), dtype=np.float32) if float(PAUSE_SEC) > 0 else None

t_all0 = time.time()
for idx, text in enumerate(SENTENCES, 1):
    out_path = WORK / f"s{idx}.wav"
    t_s0 = time.time()
    try:
        try:
            audios = model.generate(text=text, **base_kw)
        except Exception as gen_exc:
            if base_kw.get("normalize_text"):
                print(f"WARN: s{idx} failed with normalize_text=True ({gen_exc}), retrying without normalize_text", flush=True)
                retry_kw = {k: v for k, v in base_kw.items() if k != "normalize_text"}
                audios = model.generate(text=text, **retry_kw)
                result["tn_fallback"] = True
            else:
                raise
        a0 = audios[0]
        if hasattr(a0, "detach"):
            a0 = a0.detach().cpu().numpy()
        wav = np.asarray(a0, dtype=np.float32).squeeze()
        if wav.ndim > 1:
            wav = wav.mean(axis=0 if wav.shape[0] <= 8 else -1)
        sf.write(str(out_path), wav, SR)
        dt = round(time.time() - t_s0, 2)
        dur = round(len(wav) / SR, 3)
        infer_secs.append(dt)
        audio_secs.append(dur)
        result["sentences"].append({
            "index": idx,
            "text": text,
            "duration": dur,
            "sample_rate": SR,
            "infer_sec": dt,
        })
        succeeded_parts.append(wav)
        print(f"s{idx} done: dur={dur}s, infer={dt}s", flush=True)
    except Exception as exc:
        dt = round(time.time() - t_s0, 2)
        err_msg = str(exc)
        print(f"s{idx} FAIL ({dt}s): {err_msg}", flush=True)
        result["failed_sentences"].append({"index": idx, "text": text, "error": err_msg})

timings["infer_sec_per_sentence"] = infer_secs
timings["total_infer_sec"] = round(time.time() - t_all0, 2)
result["audio_sec_per_sentence"] = audio_secs

if not succeeded_parts:
    fail("INFER_FAIL_ALL")

# 7) Concatenate succeeded segments
concat_list = []
for i, part in enumerate(succeeded_parts):
    concat_list.append(part)
    if sil is not None and i < len(succeeded_parts) - 1:
        concat_list.append(sil)
full = np.concatenate(concat_list)
sf.write(str(WORK / FULL_WAV_NAME), full, SR)

full_dur = round(len(full) / SR, 3)
result["full_audio_sec"] = full_dur
result["full_duration"] = full_dur

prefix = "CLONE" if MODE == "clone" else ("AUTO" if MODE == "auto" else "OMNIVOICE")
if result["failed_sentences"]:
    verdict = f"{prefix}_PARTIAL_OK"
elif fallback_used:
    verdict = f"{prefix}_OK_FALLBACK_REF_TEXT"
else:
    verdict = f"{prefix}_OK"

result["verdict"] = verdict
write_result()
print("DONE", verdict, flush=True)
