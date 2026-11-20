#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
h3_model_mover.py — Kaggle script-type kernel

Moves the MiniMax H3 model set from Hugging Face into three PRIVATE Kaggle
datasets owned by demon820308, preserving the ComfyUI models/ subtree layout
inside each dataset:

    h3-models-shared     text_encoders/…, vae/…, latent_upscale_models/…,
                         upscale_models/…, vae_approx/…   (~32.5GB)
    h3-models-pika       diffusion_models/Cobijada…, loras/pika…  (~22GB)
    h3-models-hyperflow  diffusion_models/…flash…                 (16.4GB)

STREAMING PASS-THROUGH — NO FILE EVER LANDS ON DISK.
The user ruled out staging models on /tmp (ephemeral, wiped at session end).
Each file is streamed from its HF download URL straight into the Kaggle
dataset upload session, holding ONE 32MiB chunk in memory. Byte counts and
SHA-256 are computed incrementally over the bytes the server confirms as
persisted (so the Tr1dae upscaler's SHA-256 check is still exact). The only
file this kernel writes is /kaggle/working/mover_result.json.

Auth: the Kaggle API token is read from Kaggle Secrets
(UserSecretsClient.get_secret("KAGGLE_API_TOKEN")), falling back to the
KAGGLE_API_TOKEN environment variable. It is never printed, never written
to disk, and is scrubbed from every log line / error message.

Verification ladder (mandatory order — big files are touched only after
the whole upload chain is proven with small ones):
  1a. create throwaway private dataset h3-mover-probe with one tiny text
      file; verify via the dataset list/get API that it exists with the
      right size;
  1b. probe dataset-version semantics (does a version ADD files or
      REPLACE the file set?) with a second tiny file — the mover's
      resume strategy depends on the answer;
  2.  stream the 0.6GB audio VAE into h3-models-shared; verify its
      listed size;
  3.  only then stream the big files (26.4GB encoder, ~20GB Cobijada, …).
Any ladder failure writes mover_result.json with verdict FAIL_UPLOAD_CHAIN
plus diagnostics and stops. Per-file sizes are asserted against the
dataset listing after each dataset is finalized; files already present
with the correct size are skipped on rerun (resume).

!!! KAGGLE DATASET API CONTRACT — UNVERIFIED !!!
The blob-upload flow below mirrors the official kaggle CLI (request an
upload token/URL per file -> PUT the bytes -> create dataset/version
referencing the tokens), but it has NOT been exercised against the live
API. Every contract assumption is centralized in KaggleClient and the
low-level PUT helpers, response shapes are validated at each step, and
any mismatch raises ContractError with the raw response keys/body. The
ladder exists precisely to surface such a mismatch on a 15-byte file
instead of on a 26.4GB one. If the upload endpoint cannot accept a
streamed upload at all, upload_stream raises StreamingUploadUnsupported
— there is deliberately NO disk-staging fallback.
"""

import hashlib
import http.client
import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Hard rule (harmless here — downloads are raw streams, no HF cache is
# used): never let an HF cache land on the 20GB volume.
os.environ["HF_HOME"] = "/tmp/hf_cache"

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
OWNER = "demon820308"
SECRET_NAME = "KAGGLE_API_TOKEN"
API_BASE = "https://www.kaggle.com/api/v1"
WORKING = Path("/kaggle/working")
UA = "h3-model-mover/1.0"

CHUNK = 32 * 1024 * 1024          # upload chunk; multiple of GCS's 256KiB granule
GCS_GRANULE = 262144
assert CHUNK % GCS_GRANULE == 0
PUT_TIMEOUT_S = 900               # per socket operation on big PUTs

PROBE_SLUG = "h3-mover-probe"
UPSCALER_SHA256 = "984afb58f11d01274b90d880596ce2f93bc9c512db66fdeecd4e2d99b371d3e4"

DATASET_TITLES = {
    "h3-models-shared": "H3 Models — Shared (MiniMax H3)",
    "h3-models-pika": "H3 Models — Pika (MiniMax H3)",
    "h3-models-hyperflow": "H3 Models — HyperFlow (MiniMax H3)",
}

# (dataset_slug, path_in_dataset, [hf repos in preference order],
#  basename_in_repo, sha256_or_None, optional)
# path_in_dataset preserves the ComfyUI models/ subtree; the encoder is
# uploaded under its RENAMED filename (the name the workflows expect).
FILE_JOBS = [
    ("h3-models-shared",
     "text_encoders/qwen3vl_32b_minimax_h3_ultra_uncensored_heretic_int8_convrot.safetensors",
     ["ethanfel/Qwen3-VL-32B-Ultra-Heretic-MiniMax-H3-ComfyUI-INT8-ConvRot"],
     "qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors", None, False),
    ("h3-models-shared", "vae/minimax_h3_video_vae_fp16.safetensors",
     ["Comfy-Org/MiniMax-H3"], "minimax_h3_video_vae_fp16.safetensors", None, False),
    ("h3-models-shared", "vae/minimax_h3_audio_vae_fp32.safetensors",
     ["Comfy-Org/MiniMax-H3"], "minimax_h3_audio_vae_fp32.safetensors", None, False),
    ("h3-models-shared", "latent_upscale_models/h3_clean_latent_upscaler_film_epoch200.safetensors",
     ["Tridae/H3LatentUpscaler"], "h3_clean_latent_upscaler_film_epoch200.safetensors",
     UPSCALER_SHA256, False),
    ("h3-models-shared", "vae_approx/taeh3.safetensors",
     ["Kijai/MiniMax-H3-TAE"], "taeh3.safetensors", None, True),
    ("h3-models-pika",
     "diffusion_models/Cobijada_H3_Hybrid_Uncensored_pruned_hybrid_int8_convrot_pruned.safetensors",
     ["Winnougan/Cobijada_Minimax-H3_Hybrid_Pruned_ComfyUI",
      "taurusduan/Cobijada_Minimax-H3_Hybrid_Pruned_ComfyUI"],
     "Cobijada_H3_Hybrid_Uncensored_pruned_hybrid_int8_convrot_pruned.safetensors", None, False),
    ("h3-models-pika", "loras/pika_h3_flex_ref2va_v1_bf16.safetensors",
     ["Comfy-Org/MiniMax-H3"], "pika_h3_flex_ref2va_v1_bf16.safetensors", None, False),
    ("h3-models-hyperflow",
     "diffusion_models/minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors",
     ["VideoRebirthAI/hyperflow-h3-flash"],
     "minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors", None, False),
]
LADDER_VAE_PATH = "vae/minimax_h3_audio_vae_fp32.safetensors"  # ladder step 2

STATE = {"datasets": {}, "ladder": {}, "phases": {}, "notes": []}
CURRENT_PHASE = ["INIT"]
_TOKEN = [None]  # for scrub(); never logged


# --------------------------------------------------------------------------
# Small helpers (mirrors the sibling test kernel's style)
# --------------------------------------------------------------------------
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def note(msg):
    STATE["notes"].append(msg)
    log("NOTE: " + msg)


def scrub(text):
    if _TOKEN[0]:
        return str(text).replace(_TOKEN[0], "<TOKEN>")
    return str(text)


class Phase:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        CURRENT_PHASE[0] = self.name
        self.t0 = time.time()
        log(f"===== PHASE {self.name} =====")
        return self

    def __exit__(self, *exc):
        dt = time.time() - self.t0
        STATE["phases"][self.name] = round(dt, 2)
        log(f"===== PHASE {self.name} done in {dt:.1f}s =====")
        return False


def write_mover_result(verdict, extra=None):
    result = {
        "verdict": verdict,
        "datasets": STATE["datasets"],
        "ladder": STATE["ladder"],
        "timings": STATE["phases"],
        "notes": STATE["notes"],
    }
    if extra:
        result.update(extra)
    WORKING.mkdir(parents=True, exist_ok=True)
    (WORKING / "mover_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))
    log(f"mover_result.json written: verdict={verdict}")
    return result


def fail(verdict, err):
    log(f"FAILURE in phase {CURRENT_PHASE[0]}: {scrub(err)}")
    traceback.print_exc()
    write_mover_result(verdict, {
        "failed_phase": CURRENT_PHASE[0],
        "error": scrub(err),
    })
    print(json.dumps({"verdict": verdict, "failed_phase": CURRENT_PHASE[0],
                      "error": scrub(err)}, indent=2), flush=True)
    sys.exit(1)


class ContractError(RuntimeError):
    """The Kaggle API did not behave the way the CLI-derived contract says."""


class StreamingUploadUnsupported(RuntimeError):
    """The upload endpoint rejected both streamed upload modes.

    NotImplemented-style by design: there is NO disk-staging fallback.
    Escalate (fallback is a local upload by the user)."""


class RangeUploadRejected(RuntimeError):
    """Internal signal: chunked Content-Range PUTs are not accepted here."""


# --------------------------------------------------------------------------
# Auth — Kaggle Secrets first, env var second; token never leaves memory
# --------------------------------------------------------------------------
def get_kaggle_token():
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret(SECRET_NAME)
        if tok:
            tok = tok.strip()
            fp = hashlib.sha256(tok.encode()).hexdigest()[:8]
            log(f"auth: token loaded from Kaggle Secrets (len={len(tok)}, sha256[:8]={fp})")
            return tok
        log("auth: Kaggle Secrets returned an empty value; trying env var")
    except Exception as e:
        log(f"auth: Kaggle Secrets lookup failed ({type(e).__name__}); trying env var")
    tok = os.environ.get(SECRET_NAME)
    if tok:
        tok = tok.strip()
        fp = hashlib.sha256(tok.encode()).hexdigest()[:8]
        log(f"auth: token loaded from environment variable (len={len(tok)}, sha256[:8]={fp})")
        return tok
    raise RuntimeError(
        f"{SECRET_NAME} not available: add it under the kernel's "
        "Settings -> Secrets, or set the env var before pushing")


# --------------------------------------------------------------------------
# HF side — tree-API addressing (pattern copied from h3_hyperflow_test.py)
# plus a Range-capable, re-openable byte source for streaming.
# --------------------------------------------------------------------------
def hf_resolve(repo, basename):
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        tree = json.loads(r.read().decode())
    cands = [e for e in tree if e.get("type") == "file"
             and e["path"].rsplit("/", 1)[-1] == basename]
    if not cands:
        raise RuntimeError(f"{basename} not found in {repo}")
    cands.sort(key=lambda e: len(e["path"]))
    return cands[0]["path"], cands[0].get("size")


def hf_resolve_any(repos, basename):
    errs = []
    for repo in repos:
        try:
            path_in_repo, size = hf_resolve(repo, basename)
            return repo, path_in_repo, size
        except Exception as e:
            errs.append(f"{repo}: {e}")
    raise RuntimeError(f"could not resolve {basename} in any repo: {errs}")


class HFSource:
    """Re-openable byte stream over an HF resolve URL.

    Sequential reads reuse one connection; a read at any other offset
    re-opens the stream with a Range header (HF's CDN honors Range), so a
    resumed upload never needs the file on disk or in memory.
    """

    def __init__(self, resolve_url, total):
        self.resolve_url = resolve_url
        self.total = total
        self.final_url = None
        self.range_ok = None
        self._resp = None
        self._pos = None

    def _ensure_final_url(self):
        if self.final_url is not None:
            return
        req = urllib.request.Request(self.resolve_url,
                                     headers={"Range": "bytes=0-0", "User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r:
            self.final_url = r.geturl()
            self.range_ok = (r.status == 206)
            r.read(4096)  # at most a few bytes of the probe body; then close
        log(f"HF source: range support = {self.range_ok}")

    def _open_at(self, offset):
        self._ensure_final_url()
        headers = {"User-Agent": UA}
        if offset:
            if not self.range_ok:
                raise ContractError(
                    "HF source does not honor Range requests; cannot "
                    "re-position the stream for a resumed upload")
            headers["Range"] = f"bytes={offset}-"
        req = urllib.request.Request(self.final_url or self.resolve_url,
                                     headers=headers)
        resp = urllib.request.urlopen(req, timeout=300)
        if offset and resp.status != 206:
            resp.close()
            raise ContractError(
                f"HF source returned HTTP {resp.status} for Range offset {offset}")
        self._resp = resp
        self._pos = offset

    def read_exact_at(self, offset, n):
        """Return exactly n bytes starting at offset (n may be 0)."""
        if n == 0:
            return b""
        if self._resp is None or self._pos != offset:
            self._open_at(offset)
        buf = bytearray()
        while len(buf) < n:
            piece = self._resp.read(min(n - len(buf), 8 * 1024 * 1024))
            if not piece:
                raise RuntimeError(
                    f"HF stream ended early at {offset + len(buf)} "
                    f"(wanted {n} bytes from {offset}, total {self.total})")
            buf += piece
            self._pos += len(piece)
        return bytes(buf)

    def close(self):
        if self._resp is not None:
            try:
                self._resp.close()
            except Exception:
                pass
            self._resp = None
            self._pos = None


# --------------------------------------------------------------------------
# Low-level upload PUTs (module-level seams; the streamed-upload engine
# below is built only from these + HFSource, so they can be stubbed in
# offline tests). All target the createUrl returned by start_upload.
# --------------------------------------------------------------------------
def _conn_for(url):
    u = urllib.parse.urlsplit(url)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    conn = cls(u.hostname, u.port, timeout=PUT_TIMEOUT_S)
    path = u.path + (("?" + u.query) if u.query else "")
    return conn, path


def put_range_chunk(url, buf, start, total):
    """PUT buf as bytes [start, start+len(buf)) of a resumable session.
    Returns (status, headers_dict, body_snippet)."""
    conn, path = _conn_for(url)
    try:
        conn.request("PUT", path, body=buf, headers={
            "Content-Length": str(len(buf)),
            "Content-Range": f"bytes {start}-{start + len(buf) - 1}/{total}",
        })
        resp = conn.getresponse()
        body = resp.read(65536)
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, body
    finally:
        conn.close()


def query_upload_status(url, total):
    """Bytes the server has confirmed for this session (GCS protocol:
    PUT with Content-Range */total and no body; 308 + Range, or 200/201
    when the upload is already complete)."""
    conn, path = _conn_for(url)
    try:
        conn.request("PUT", path, body=b"", headers={
            "Content-Length": "0",
            "Content-Range": f"bytes */{total}",
        })
        resp = conn.getresponse()
        resp.read(65536)
        if resp.status in (200, 201):
            return total
        if resp.status == 308:
            rng = {k.lower(): v for k, v in resp.getheaders()}.get("range", "")
            # "bytes=0-12345" -> 12346 bytes confirmed
            try:
                return int(rng.rsplit("-", 1)[1]) + 1
            except Exception:
                return 0
        raise ContractError(
            f"upload status query -> HTTP {resp.status}")
    finally:
        conn.close()


def put_bytes_simple(url, data):
    """One-shot PUT of a small in-memory body (ladder probe files).
    Tries a Content-Range PUT first (resumable-session style), then a
    plain PUT; raises ContractError if both fail."""
    errors = []
    try:
        status, _, body = put_range_chunk(url, data, 0, len(data))
        if status in (200, 201):
            return
        errors.append(f"range PUT -> HTTP {status}: {body[:300]!r}")
    except Exception as e:
        errors.append(f"range PUT raised: {e}")
    conn, path = _conn_for(url)
    try:
        conn.request("PUT", path, body=data,
                     headers={"Content-Length": str(len(data))})
        resp = conn.getresponse()
        body = resp.read(65536)
        if resp.status in (200, 201):
            return
        errors.append(f"plain PUT -> HTTP {resp.status}: {body[:300]!r}")
    finally:
        conn.close()
    raise ContractError("tiny-file PUT failed both ways: " + " | ".join(errors))


# --------------------------------------------------------------------------
# Streamed-upload engine — HFSource -> upload session, one chunk in RAM.
# SHA-256 is updated ONLY over byte ranges the server has confirmed as
# persisted, so every byte is hashed exactly once even across resumes.
# --------------------------------------------------------------------------
def stream_upload_resumable(create_url, source, total):
    """Mode A: chunked Content-Range PUTs against a resumable session.
    Returns (sha256_hex, bytes_confirmed)."""
    hasher = hashlib.sha256()
    confirmed = 0
    stalls = 0
    next_log_at = 2 * 2**30
    while confirmed < total:
        want = min(CHUNK, total - confirmed)
        buf = source.read_exact_at(confirmed, want)
        body = b""
        try:
            status, _, body = put_range_chunk(create_url, buf, confirmed, total)
        except Exception as e:
            status = None
            log(f"chunk PUT at {confirmed} raised {type(e).__name__}: {e}; querying status")
        if status in (200, 201):
            if confirmed + len(buf) != total:
                raise ContractError(
                    "upload endpoint reported success before all bytes were "
                    "sent — this URL is not a resumable session")
            server_confirmed = total
        elif status == 308 or status is None or (isinstance(status, int) and status >= 500):
            # 308 = incomplete; a transport error or 5xx may still have
            # persisted a prefix — ask the server what it holds, then resume.
            try:
                server_confirmed = query_upload_status(create_url, total)
            except ContractError:
                if confirmed == 0:
                    raise RangeUploadRejected(
                        "status query not supported on this URL either")
                raise
        elif confirmed == 0 and status in (400, 403, 404, 405, 411):
            raise RangeUploadRejected(
                f"first chunk PUT rejected: HTTP {status}: {body[:500]!r}")
        else:
            raise ContractError(
                f"chunk PUT at {confirmed} -> HTTP {status}: {body[:500]!r}")
        if server_confirmed > confirmed:
            newly = min(server_confirmed, confirmed + len(buf))
            hasher.update(buf[:newly - confirmed])
            confirmed = newly
            stalls = 0
            if confirmed >= next_log_at or confirmed == total:
                log(f"  upload: {confirmed / 2**30:.1f} GiB / {total / 2**30:.1f} GiB confirmed")
                next_log_at = confirmed + 2 * 2**30
        else:
            stalls += 1
            if stalls > 5:
                raise ContractError(
                    f"upload session stalled at {confirmed}/{total} after retries")
            log(f"  upload: no progress at {confirmed}; retrying (stall {stalls})")
            time.sleep(5)
    return hasher.hexdigest(), confirmed


def stream_upload_single_shot(create_url, source, total):
    """Mode B: one PUT with Content-Length = total, body streamed from the
    HF source in 8MiB pieces. Still disk-free; a mid-PUT failure restarts
    the file from byte 0 (the caller may retry the whole file)."""
    conn, path = _conn_for(create_url)
    hasher = hashlib.sha256()
    sent = 0
    try:
        conn.putrequest("PUT", path)
        conn.putheader("Content-Length", str(total))
        conn.endheaders()
        next_log_at = 2 * 2**30
        while sent < total:
            piece = source.read_exact_at(sent, min(8 * 1024 * 1024, total - sent))
            conn.send(piece)
            hasher.update(piece)
            sent += len(piece)
            if sent >= next_log_at or sent == total:
                log(f"  upload (single-shot): {sent / 2**30:.1f} GiB / {total / 2**30:.1f} GiB")
                next_log_at = sent + 2 * 2**30
        resp = conn.getresponse()
        body = resp.read(65536)
        if resp.status not in (200, 201):
            raise ContractError(
                f"single-shot PUT -> HTTP {resp.status}: {body[:500]!r}")
    finally:
        conn.close()
    return hasher.hexdigest(), sent


# --------------------------------------------------------------------------
# Kaggle dataset API client — ALL contract assumptions live here.
# (Mirrors the official kaggle CLI's blob-upload flow; UNVERIFIED against
# the live API — every response shape is validated, mismatches raise
# ContractError with the raw keys/body so the ladder fails loudly.)
# --------------------------------------------------------------------------
class KaggleClient:
    def __init__(self, token):
        self._token = token

    # -- raw request plumbing -------------------------------------------------
    def _raw(self, method, path, body=None, params=None):
        url = API_BASE + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers={
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": UA,
        })
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace") if e.fp else ""
            return e.code, raw

    def _request_json(self, method, path, body=None, params=None):
        status, raw = self._raw(method, path, body=body, params=params)
        if status >= 400:
            raise ContractError(
                f"{method} {path} -> HTTP {status}: {scrub(raw[:1000])}")
        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except Exception:
            raise ContractError(
                f"{method} {path} returned non-JSON: {scrub(raw[:500])}")

    # -- blob upload ----------------------------------------------------------
    def start_upload(self, filename, size, mtime_epoch):
        """Ask Kaggle for an upload slot for one file via /api/v1/blobs/upload.
        Returns (file_token, create_url)."""
        body = {
            "type": "DATASET",
            "name": filename,
            "contentType": "application/octet-stream",
            "contentLength": int(size),
            "lastModifiedEpochSeconds": int(mtime_epoch),
        }
        resp = self._request_json("POST", "/blobs/upload", body=body)
        token = next((resp[k] for k in ("token", "fileToken", "uploadToken")
                      if resp.get(k)), None)
        url = next((resp[k] for k in ("createUrl", "url", "uploadUrl", "signedUrl")
                    if resp.get(k)), None)
        if not token or not url:
            raise ContractError(
                f"start-upload response missing token/url; "
                f"keys={sorted(resp.keys())} body={scrub(str(resp)[:500])}")
        return token, url

    def upload_stream(self, create_url, source, total):
        """Stream `total` bytes from HFSource into the upload slot.
        Returns (sha256_hex, bytes_sent, mode). Raises
        StreamingUploadUnsupported if the endpoint accepts neither
        streamed mode — there is no disk fallback, by design."""
        source._ensure_final_url()
        if source.range_ok:
            try:
                digest, sent = stream_upload_resumable(create_url, source, total)
                return digest, sent, "resumable"
            except RangeUploadRejected as e:
                note(f"resumable upload not accepted ({e}); "
                     "falling back to single-shot streamed PUT")
                source.close()
        else:
            note("HF source has no Range support; using single-shot streamed PUT")
        try:
            digest, sent = stream_upload_single_shot(create_url, source, total)
            return digest, sent, "single-shot"
        except Exception as e:
            raise StreamingUploadUnsupported(
                f"upload endpoint accepted neither resumable chunked PUTs "
                f"nor a single-shot streamed PUT (last error: {scrub(e)}). "
                f"Escalate: the fallback is a LOCAL upload by the user, "
                f"not disk staging.")

    # -- dataset create / version ----------------------------------------------
    @staticmethod
    def _file_infos(tokens):
        return [{"token": t, "description": ""} for t, _ in tokens]

    def _check_api_error(self, op, resp):
        if isinstance(resp, dict) and (resp.get("hasError") or resp.get("error") or resp.get("errorMessage")):
            err = resp.get("error") or resp.get("errorMessage") or str(resp)
            raise ContractError(f"{op} returned error: {scrub(err)}")

    def get_dataset_version(self, slug):
        status, raw = self._raw("GET", f"/datasets/view/{OWNER}/{slug}")
        if status >= 400:
            return 0
        try:
            d = json.loads(raw)
            return int(d.get("currentVersionNumber") or d.get("currentVersionNumberNullable") or 0)
        except Exception:
            return 0

    def wait_dataset_ready(self, slug, min_version=1, timeout_s=600):
        deadline = time.time() + timeout_s
        last = None
        while time.time() < deadline:
            status, raw = self._raw("GET", f"/datasets/status/{OWNER}/{slug}")
            if status == 200:
                val = raw.strip().strip('"').lower()
                last = val
                if val == "ready":
                    cur_ver = self.get_dataset_version(slug)
                    if cur_ver >= min_version:
                        return
                    last = f"ready(ver={cur_ver}<{min_version})"
                elif val in ("error", "failed"):
                    raise ContractError(f"dataset {slug} status={val}")
            time.sleep(3)
        raise ContractError(f"dataset {slug} not ready within {timeout_s}s (last={last})")

    def create_dataset(self, slug, title, tokens):
        body = {
            "ownerSlug": OWNER,
            "slug": slug,
            "title": title,
            "licenseName": "CC0-1.0",
            "isPrivate": True,
            "subtitle": "MiniMax H3 model files for ComfyUI (private)",
            "description": "Streamed from Hugging Face by h3_model_mover.py.",
            "files": self._file_infos(tokens),
            "categoryIds": [],
        }
        resp = self._request_json("POST", "/datasets/create/new", body=body)
        self._check_api_error(f"create_dataset({slug})", resp)
        self.wait_dataset_ready(slug, min_version=1)
        return resp

    def create_version(self, slug, tokens, version_notes):
        old_ver = self.get_dataset_version(slug)
        body = {
            "versionNotes": version_notes,
            "subtitle": "MiniMax H3 model files for ComfyUI (private)",
            "description": "Streamed from Hugging Face by h3_model_mover.py.",
            "files": self._file_infos(tokens),
            "categoryIds": [],
            "deleteOldVersions": False,
        }
        resp = self._request_json(
            "POST", f"/datasets/create/version/{OWNER}/{slug}", body=body)
        self._check_api_error(f"create_version({slug})", resp)
        self.wait_dataset_ready(slug, min_version=old_ver + 1)
        return resp

    # -- listing / verification --------------------------------------------------
    @staticmethod
    def _normalize_files(items):
        out = []
        for it in items:
            if not isinstance(it, dict):
                continue
            name = it.get("name") or it.get("fileName") or it.get("ref")
            size = it.get("size")
            if size is None:
                size = it.get("totalBytes") or it.get("fileSize") or it.get("bytes")
            if name is not None:
                out.append({"name": str(name),
                            "size": int(size) if size is not None else None})
        return out

    def _extract_file_list(self, payload):
        """Hunt for a file list in a view/list response of unknown shape."""
        if isinstance(payload, list):
            return self._normalize_files(payload)
        if isinstance(payload, dict):
            for key in ("files", "datasetFiles", "resources", "dataset_files"):
                if isinstance(payload.get(key), list):
                    return self._normalize_files(payload[key])
            for v in payload.values():
                if isinstance(v, dict):
                    found = self._extract_file_list(v)
                    if found is not None:
                        return found
        return None

    def list_dataset_files(self, slug):
        """[{'name','size'}...] if the dataset exists, None if it does
        not, ContractError if it responds but cannot be understood."""
        status2, raw2 = self._raw("GET", f"/datasets/list/{OWNER}/{slug}")
        if status2 in (403, 404):
            return None
        if status2 < 400:
            try:
                files = self._extract_file_list(json.loads(raw2))
            except Exception:
                files = None
            if files is not None:
                return files
        status, raw = self._raw("GET", f"/datasets/view/{OWNER}/{slug}")
        if status in (403, 404):
            return None
        if status < 400:
            try:
                files = self._extract_file_list(json.loads(raw))
            except Exception:
                files = None
            if files is not None:
                return files
            raise ContractError(
                f"dataset file listing for {slug}: no file list found in "
                f"either response; view keys/body={scrub(raw[:400])} "
                f"list body={scrub(raw2[:400])}")
        raise ContractError(
            f"dataset file listing for {slug}: list -> HTTP {status2}, "
            f"view -> HTTP {status}: {scrub(raw2[:400])}")

    def try_delete_dataset(self, slug):
        """Best-effort cleanup of the probe dataset; never fatal."""
        try:
            status, raw = self._raw("POST", f"/datasets/delete/{OWNER}/{slug}")
            return f"HTTP {status}"
        except Exception as e:
            return f"delete attempt raised: {type(e).__name__}"


# --------------------------------------------------------------------------
# Orchestration — ladder first, then per-dataset streaming + verification
# --------------------------------------------------------------------------
def sizes_match(listed, expected):
    """(exact, rounded) — the API is expected to report exact bytes; a
    rounded report is tolerated but FLAGGED, per the verification spec."""
    if listed is None or expected is None:
        return False, False
    if listed == expected:
        return True, False
    for unit in (1024, 1000):
        if round(expected / unit) * unit == listed or int(expected / unit) * unit == listed:
            return False, True
    if expected > 0 and abs(listed - expected) / expected < 0.001:
        return False, True
    return False, False


def find_in_listing(listing, path_in_dataset):
    base = path_in_dataset.rsplit("/", 1)[-1]
    for f in listing or []:
        if f["name"].rsplit("/", 1)[-1] == base:
            return f
    return None


def dataset_state(slug):
    if slug not in STATE["datasets"]:
        STATE["datasets"][slug] = {"slug": slug, "files": {}}
    return STATE["datasets"][slug]


def record_file(slug, path_in_dataset, size, status, **extra):
    entry = {"name": path_in_dataset, "bytes": size, "status": status}
    entry.update(extra)
    dataset_state(slug)["files"][path_in_dataset] = entry
    return entry


def stream_one_file(client, job):
    """Stream one model file HF -> Kaggle upload slot. Returns its
    file token plus the run record. Nothing touches disk."""
    slug, path_in_ds, repos, basename, sha, optional = job
    repo, path_in_repo, size = hf_resolve_any(repos, basename)
    if not size:
        raise RuntimeError(f"HF tree API gave no size for {repo}/{path_in_repo}")
    src = HFSource(f"https://huggingface.co/{repo}/resolve/main/{path_in_repo}", size)
    t0 = time.time()
    try:
        token, create_url = client.start_upload(path_in_ds, size, int(time.time()))
        digest, sent, mode = client.upload_stream(create_url, src, size)
    finally:
        src.close()
    if sent != size:
        raise ContractError(f"{path_in_ds}: streamed {sent} bytes, expected {size}")
    if sha and digest != sha:
        raise RuntimeError(
            f"SHA-256 mismatch for streamed {basename}: got {digest}, want {sha}")
    dt = time.time() - t0
    log(f"streamed {path_in_ds}: {size / 2**30:.2f} GiB in {dt:.0f}s "
        f"(mode={mode}, sha256={digest[:16]}…)")
    return {"token": token, "bytes": size, "sha256": digest,
            "upload_mode": mode, "seconds": round(dt, 2)}


def ladder_probe(client):
    """Ladder steps 1a/1b — prove the whole upload chain on tiny files.
    Returns the dataset-version semantics: 'additive' or 'replace'."""
    with Phase("LADDER"):
        content = b"h3 mover probe - streaming upload chain verification\n"
        token, create_url = client.start_upload("probe.txt", len(content),
                                                int(time.time()))
        put_bytes_simple(create_url, content)
        if client.list_dataset_files(PROBE_SLUG) is None:
            client.create_dataset(PROBE_SLUG, "H3 Mover Probe (throwaway)",
                                  [(token, "probe.txt")])
        else:
            client.create_version(PROBE_SLUG, [(token, "probe.txt")],
                                  "probe v1 — streaming upload chain verification")
        files = client.list_dataset_files(PROBE_SLUG)
        if files is None:
            raise ContractError(f"probe dataset {PROBE_SLUG} not visible after create")
        hit = find_in_listing(files, "probe.txt")
        if hit is None or hit["size"] != len(content):
            raise ContractError(
                f"probe verification failed: listing={files}, "
                f"expected probe.txt of {len(content)} bytes")
        STATE["ladder"]["probe_create"] = {
            "dataset": PROBE_SLUG, "file": "probe.txt",
            "bytes": len(content), "verified": True}
        log("ladder 1a OK: probe dataset created and verified")

        content2 = b"probe2\n"
        token2, create_url2 = client.start_upload("probe2.txt", len(content2),
                                                  int(time.time()))
        put_bytes_simple(create_url2, content2)
        client.create_version(PROBE_SLUG, [(token2, "probe2.txt")],
                              "probe v2 — version semantics check")
        files2 = client.list_dataset_files(PROBE_SLUG) or []
        names = {f["name"].rsplit("/", 1)[-1] for f in files2}
        if "probe2.txt" not in names:
            raise ContractError(
                f"probe v2 verification failed: probe2.txt not in listing {files2}")
        semantics = "additive" if "probe.txt" in names else "replace"
        STATE["ladder"]["version_semantics"] = semantics
        log(f"ladder 1b OK: dataset versions are '{semantics}' "
            f"({'files accumulate' if semantics == 'additive' else 'each version replaces the file set'})")
        return semantics


def ladder_step2_vae(client):
    """Ladder step 2 — stream the 0.6GB audio VAE into h3-models-shared
    and verify its listed size, before any big file is touched."""
    job = next(j for j in FILE_JOBS if j[1] == LADDER_VAE_PATH)
    slug = job[0]
    log(f"ladder 2: streaming {LADDER_VAE_PATH} into {slug}")
    rec = stream_one_file(client, job)
    if client.list_dataset_files(slug) is None:
        client.create_dataset(slug, DATASET_TITLES[slug],
                              [(rec["token"], LADDER_VAE_PATH)])
    else:
        client.create_version(slug, [(rec["token"], LADDER_VAE_PATH)],
                              "ladder step 2: audio VAE")
    files = client.list_dataset_files(slug)
    hit = find_in_listing(files, LADDER_VAE_PATH)
    exact, rounded = sizes_match(hit["size"] if hit else None, rec["bytes"])
    if hit is None or not (exact or rounded):
        raise ContractError(
            f"ladder step 2 verification failed: listing hit={hit}, "
            f"expected {rec['bytes']} bytes")
    if rounded:
        note("ladder 2: dataset API appears to report ROUNDED file sizes — flagged")
    record_file(slug, LADDER_VAE_PATH, rec["bytes"], "verified (ladder step 2)",
                sha256=rec["sha256"], upload_mode=rec["upload_mode"],
                seconds=rec["seconds"])
    STATE["ladder"]["shared_first_file"] = {
        "file": LADDER_VAE_PATH, "bytes": rec["bytes"], "verified": True}
    log("ladder 2 OK: audio VAE in h3-models-shared with verified size")


def verify_dataset(client, slug, expected):
    """Assert every expected file is listed with the right size.
    expected: {path_in_dataset: bytes}. Returns True when all pass."""
    listing = client.list_dataset_files(slug)
    if listing is None:
        raise ContractError(f"dataset {slug} vanished at verification time")
    all_ok = True
    for path_in_ds, size in expected.items():
        job = next(j for j in FILE_JOBS if j[0] == slug and j[1] == path_in_ds)
        optional = job[5]
        hit = find_in_listing(listing, path_in_ds)
        cur = dataset_state(slug)["files"].get(path_in_ds, {})
        if hit is None:
            if optional:
                record_file(slug, path_in_ds, size, "optional — absent (ok)",
                            **{k: v for k, v in cur.items()
                               if k not in ("name", "bytes", "status")})
                note(f"optional file absent from {slug} (ok): {path_in_ds}")
                continue
            record_file(slug, path_in_ds, size, "VERIFY-FAILED: not in listing")
            all_ok = False
            continue
        exact, rounded = sizes_match(hit["size"], size)
        if exact:
            record_file(slug, path_in_ds, size, "verified",
                        **{k: v for k, v in cur.items()
                           if k not in ("name", "bytes", "status")})
        elif rounded:
            record_file(slug, path_in_ds, size,
                        "verified (API size appears rounded — FLAGGED)",
                        listed_bytes=hit["size"],
                        **{k: v for k, v in cur.items()
                           if k not in ("name", "bytes", "status")})
            note(f"{slug}/{path_in_ds}: listed size {hit['size']} looks like a "
                 f"rounded version of {size} — flagged, accepted")
        else:
            record_file(slug, path_in_ds, size,
                        f"VERIFY-FAILED: listed {hit['size']} bytes",
                        listed_bytes=hit["size"])
            all_ok = False
    return all_ok


def process_dataset(client, slug, semantics):
    """Stream whatever `slug` still needs and finalize it.

    Resume: files already listed with the correct size are skipped. If
    dataset versions turn out to REPLACE the file set, a partial dataset
    is rebuilt with one version call carrying every file's token
    (already-present files are re-streamed in that case — the API gives
    no way to reference them — and the log says so)."""
    jobs = [j for j in FILE_JOBS if j[0] == slug]
    expected = {}
    for job in jobs:
        _, _, size = hf_resolve_any(job[2], job[3])
        if not size:
            raise RuntimeError(f"HF tree API gave no size for {job[3]}")
        expected[job[1]] = size

    listing = client.list_dataset_files(slug)
    exists = listing is not None
    satisfied = {}
    if exists:
        for job in jobs:
            hit = find_in_listing(listing, job[1])
            if hit is not None:
                exact, rounded = sizes_match(hit["size"], expected[job[1]])
                if exact or rounded:
                    satisfied[job[1]] = hit["size"]
                    if job[1] not in dataset_state(slug)["files"]:
                        record_file(slug, job[1], expected[job[1]],
                                    "present — skipped (resume)")

    missing = [j for j in jobs if j[1] not in satisfied]
    if not missing:
        log(f"{slug}: all {len(jobs)} files already present with correct sizes — skipped")
        if not verify_dataset(client, slug, expected):
            fail("FAIL_VERIFY", f"{slug}: pre-existing files failed verification")
        return

    if exists and semantics == "replace":
        note(f"{slug}: versions REPLACE the file set — re-streaming all "
             f"{len(jobs)} files so the final version is complete "
             f"({len(satisfied)} were already present and are re-sent; "
             f"the API offers no token for existing files)")
        to_stream = jobs
    else:
        to_stream = missing

    tokens = []
    for job in to_stream:
        cur = dataset_state(slug)["files"].get(job[1], {})
        if job[1] in satisfied and semantics == "additive":
            continue
        try:
            rec = stream_one_file(client, job)
        except Exception as e:
            if job[5]:  # optional
                note(f"optional file failed to stream (continuing): {job[1]}: {scrub(e)}")
                record_file(slug, job[1], expected[job[1]],
                            f"optional — stream failed: {scrub(e)}")
                continue
            raise
        tokens.append((rec["token"], job[1]))
        record_file(slug, job[1], rec["bytes"], "uploaded (pending finalize)",
                    sha256=rec["sha256"], upload_mode=rec["upload_mode"],
                    seconds=rec["seconds"])

    if not tokens and exists:
        log(f"{slug}: nothing new to finalize")
    elif not exists:
        client.create_dataset(slug, DATASET_TITLES[slug], tokens)
        log(f"{slug}: dataset created with {len(tokens)} files")
    else:
        client.create_version(slug, tokens,
                              f"h3 mover: add {len(tokens)} file(s)")
        log(f"{slug}: new version created with {len(tokens)} file token(s)")

    if not verify_dataset(client, slug, expected):
        fail("FAIL_VERIFY", f"{slug}: post-upload verification failed "
                            f"(see mover_result.json for per-file detail)")
    log(f"{slug}: complete and verified ({len(jobs)} files)")


def main():
    WORKING.mkdir(parents=True, exist_ok=True)
    try:
        with Phase("AUTH"):
            _TOKEN[0] = get_kaggle_token()
        client = KaggleClient(_TOKEN[0])
        semantics = ladder_probe(client)
        with Phase("LADDER2"):
            ladder_step2_vae(client)
        with Phase("DATASETS"):
            for slug in ("h3-models-shared", "h3-models-pika", "h3-models-hyperflow"):
                log(f"----- dataset {slug} -----")
                process_dataset(client, slug, semantics)
        with Phase("CLEANUP"):
            outcome = client.try_delete_dataset(PROBE_SLUG)
            STATE["ladder"]["probe_delete_attempt"] = outcome
            log(f"probe dataset delete attempt: {outcome} "
                f"(best-effort; a leftover 15-byte private dataset is harmless)")
    except SystemExit:
        raise
    except (ContractError, StreamingUploadUnsupported) as e:
        if CURRENT_PHASE[0] in ("LADDER", "LADDER2"):
            fail("FAIL_UPLOAD_CHAIN", e)
        fail(f"FAIL_{CURRENT_PHASE[0]}", e)
    except Exception as e:
        if CURRENT_PHASE[0] in ("LADDER", "LADDER2"):
            fail("FAIL_UPLOAD_CHAIN", e)
        fail(f"FAIL_{CURRENT_PHASE[0]}", e)

    result = write_mover_result("MOVER_OK")
    print("\n================ H3 MODEL MOVER SUMMARY ================", flush=True)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
