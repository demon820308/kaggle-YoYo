"""YoYo OmniVoice on Kaggle: install, then generate test lines with the
user's standard teaching voice (OmniVoice, female/young/mid pitch,
speed 0.85, per-sentence generation, 0.5s silence between sentences).
Outputs land in /kaggle/working: s1..sN.wav, test_full.wav, result.json.
"""
import json
import subprocess
import sys
import time

WORKING = "/kaggle/working"
result = {"platform": "kaggle", "job": "omnivoice-deploy-test"}
timings = {}


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


# --- environment -----------------------------------------------------------
t0 = time.time()
smi = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
result["gpu"] = (smi.stdout or smi.stderr).strip()

# --- install ---------------------------------------------------------------
t0 = time.time()
inst = run([sys.executable, "-m", "pip", "install", "-q", "omnivoice", "soundfile"])
timings["pip_install_sec"] = round(time.time() - t0, 1)
print((inst.stdout or "")[-1500:], flush=True)
print((inst.stderr or "")[-1500:], flush=True)
if inst.returncode != 0:
    result["verdict"] = "PIP_INSTALL_FAIL"
    print(json.dumps(result, ensure_ascii=False), flush=True)
    with open(f"{WORKING}/result.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    sys.exit(0)

# --- inspect CLI ------------------------------------------------------------
help_out = run(["omnivoice-infer", "--help"])
help_text = (help_out.stdout or "") + (help_out.stderr or "")
print(help_text[:3000], flush=True)
has_speed = "--speed" in help_text
result["cli_has_speed_flag"] = has_speed

# --- inference, one call per sentence (user standard) -----------------------
# __OMNIVOICE_PARAMS_BEGIN__
SENTENCES = __SENTENCES_JSON__
INSTRUCT = __INSTRUCT_JSON__
SPEED = __SPEED_STR__
PAUSE_SEC = __PAUSE_SEC__
# __OMNIVOICE_PARAMS_END__

wavs = []
infer_secs = []
for i, text in enumerate(SENTENCES, 1):
    out = f"{WORKING}/s{i}.wav"
    cmd = [
        "omnivoice-infer",
        "--text", text,
        "--output", out,
        "--language", "Chinese",
        "--instruct", INSTRUCT,
    ]
    if has_speed:
        cmd += ["--speed", str(SPEED)]
    t0 = time.time()
    r = run(cmd)
    dt = round(time.time() - t0, 1)
    infer_secs.append(dt)
    print((r.stdout or "")[-1200:], flush=True)
    print((r.stderr or "")[-1200:], flush=True)
    if r.returncode != 0:
        result["verdict"] = f"INFER_FAIL_S{i}"
        result["timings"] = timings
        with open(f"{WORKING}/result.json", "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        sys.exit(0)
    wavs.append(out)

timings["infer_sec_per_sentence"] = infer_secs

# --- concat with 0.5s silence between sentences (user standard) -------------
import numpy as np
import soundfile as sf

parts = []
sr_seen = None
audio_secs = []
for p in wavs:
    data, sr = sf.read(p, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    sr_seen = sr
    audio_secs.append(round(len(data) / sr, 2))
    parts.append(data)
    parts.append(np.zeros(int(sr * PAUSE_SEC), dtype="float32"))
full = np.concatenate(parts[:-1])  # drop trailing silence
sf.write(f"{WORKING}/test_full.wav", full, sr_seen)
result["sample_rate"] = sr_seen
result["audio_sec_per_sentence"] = audio_secs
result["full_audio_sec"] = round(len(full) / sr_seen, 2)
result["timings"] = timings
result["instruct"] = INSTRUCT
result["speed_applied"] = float(SPEED) if has_speed else None
result["verdict"] = "OMNIVOICE_OK"
print("RESULT_BEGIN", flush=True)
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
print("RESULT_END", flush=True)
with open(f"{WORKING}/result.json", "w") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
