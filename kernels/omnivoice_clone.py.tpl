# OmniVoice clone kernel — built per 定版文档附录 B skeleton (order fixed).
# Sample is the user's own voice, embedded as base64; use only for this task.
import base64, json, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
SR = 24000
# __OMNIVOICE_PARAMS_BEGIN__
SPEED = __SPEED_STR__
PAUSE_SEC = __PAUSE_SEC__
SENTENCES = __SENTENCES_JSON__
REF_B64_PARTS = __REF_B64_PARTS_JSON__
# __OMNIVOICE_PARAMS_END__

result = {"mode": "clone", "speed": 1.0, "pause_sec": PAUSE_SEC, "sentences": []}


def run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True)


def write_result():
    (WORK / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RESULT_JSON:", json.dumps(result, ensure_ascii=False), flush=True)


def fail(verdict, **extra):
    result["verdict"] = verdict
    result.update(extra)
    write_result()
    print("DONE", verdict, flush=True)
    sys.exit(0)


# 1) decode embedded sample -> ref.m4a
raw = base64.b64decode("".join(REF_B64_PARTS))
(WORK / "ref.m4a").write_bytes(raw)
print("ref.m4a bytes:", len(raw), flush=True)

# 2) ffmpeg -> 24kHz mono ref.wav
r = run(["ffmpeg", "-y", "-i", WORK / "ref.m4a", "-ac", "1", "-ar", str(SR), WORK / "ref.wav"])
if r.returncode != 0 or not (WORK / "ref.wav").exists():
    fail("REF_CONVERT_FAIL", stderr=(r.stderr or "")[-2000:])

# 3) pip install omnivoice + soundfile
r = run([sys.executable, "-m", "pip", "install", "-q", "omnivoice", "soundfile"])
if r.returncode != 0:
    fail("PIP_INSTALL_FAIL", stderr=(r.stderr or "")[-2000:])

import soundfile as sf

info = sf.info(str(WORK / "ref.wav"))
result["ref_wav"] = {
    "duration": round(info.frames / info.samplerate, 3),
    "sample_rate": info.samplerate,
}
print("ref_wav:", result["ref_wav"], flush=True)

# 4) CLI self-check: --ref_audio must exist
r = run(["omnivoice-infer", "--help"])
help_text = (r.stdout or "") + (r.stderr or "")
result["cli_flags"] = {
    "ref_audio": "ref_audio" in help_text,
    "ref_text": "ref_text" in help_text,
    "speed": "speed" in help_text,
}
print("cli_flags:", result["cli_flags"], flush=True)
if "ref_audio" not in help_text:
    fail("CLI_NO_REF_AUDIO", help_tail=help_text[-1500:])


# 5) clone run; ref_text=None first (built-in Whisper auto-transcribes)
def run_all(ref_text):
    durs = []
    for i, text in enumerate(SENTENCES, 1):
        out = WORK / f"s{i}.wav"
        cmd = ["omnivoice-infer", "--text", text, "--output", out,
               "--language", "Chinese", "--ref_audio", WORK / "ref.wav",
               "--speed", SPEED]
        if ref_text:
            cmd += ["--ref_text", ref_text]
        t0 = time.time()
        r = run(cmd)
        if r.returncode != 0 or not out.exists():
            print("INFER FAIL s%d stdout tail: %s" % (i, (r.stdout or "")[-800:]), flush=True)
            print("INFER FAIL s%d stderr tail: %s" % (i, (r.stderr or "")[-1500:]), flush=True)
            return False, i, durs
        inf = sf.info(str(out))
        durs.append({
            "index": i, "text": text,
            "duration": round(inf.frames / inf.samplerate, 3),
            "sample_rate": inf.samplerate,
            "infer_sec": round(time.time() - t0, 1),
        })
        print("s%d done:" % i, durs[-1], flush=True)
    return True, None, durs


ok, failed_at, durs = run_all(None)
verdict = "CLONE_OK"
if not ok:
    # 3.3 fallback: explicit whisper-small transcription, rerun with --ref_text
    print("primary path failed at sentence", failed_at, "-> fallback", flush=True)
    r = run([sys.executable, "-m", "pip", "install", "-q", "openai-whisper"])
    if r.returncode != 0:
        fail("CLONE_FAIL", stage="whisper_pip_install",
             first_fail_sentence=failed_at, stderr=(r.stderr or "")[-1500:])
    try:
        import whisper
        model = whisper.load_model("small")
        tr = model.transcribe(str(WORK / "ref.wav"), language="zh")
        ref_text = (tr.get("text") or "").strip()
    except Exception as e:
        fail("CLONE_FAIL", stage="whisper_transcribe",
             first_fail_sentence=failed_at, error=str(e))
    if not ref_text:
        fail("CLONE_FAIL", stage="whisper_empty_text", first_fail_sentence=failed_at)
    result["fallback_ref_text"] = ref_text
    print("fallback ref_text:", ref_text, flush=True)
    ok, failed_at, durs = run_all(ref_text)
    if not ok:
        fail("CLONE_FAIL", stage="fallback_infer",
             first_fail_sentence=failed_at, ref_text=ref_text)
    verdict = "CLONE_OK_FALLBACK_REF_TEXT"

# 6) concat with 0.4s silence gaps -> clone_full.wav, write result.json
import numpy as np

parts = []
sil = np.zeros(int(SR * PAUSE_SEC), dtype=np.float32)
for i in range(1, len(SENTENCES) + 1):
    data, sr_i = sf.read(str(WORK / f"s{i}.wav"), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    parts.append(data)
    if i < len(SENTENCES):
        parts.append(sil)
full = np.concatenate(parts)
sf.write(str(WORK / "clone_full.wav"), full, SR)

result["verdict"] = verdict
result["sentences"] = durs
result["full_duration"] = round(len(full) / SR, 3)
write_result()
print("DONE", verdict, flush=True)
