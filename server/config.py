"""Configuration, constants, paths, and token management for YoYo Kaggle Console."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ASSETS_DIR = ROOT_DIR / "assets"
KERNELS_DIR = ROOT_DIR / "kernels"
WEB_DIR = ROOT_DIR / "web"

VOICE_OUTPUTS_DIR = DATA_DIR / "voice_outputs"
VOICE_LIBRARY_DIR = DATA_DIR / "voice_library"
VIDEO_OUTPUTS_DIR = DATA_DIR / "outputs"
VCLONE_DIR = DATA_DIR / "vclone"
COOKIES_DIR = DATA_DIR / "cookies"
TOKEN_FILE = DATA_DIR / "kaggle_token"
ACCOUNTS_FILE = DATA_DIR / "kaggle_accounts.json"
SETTINGS_FILE = DATA_DIR / "settings.json"
JOBS_FILE = DATA_DIR / "jobs.json"
RUNS_FILE = DATA_DIR / "runs.json"
VOICE_RUNS_FILE = DATA_DIR / "voice_runs.json"
VOICE_LIBRARY_FILE = DATA_DIR / "voice_library.json"
VCLONE_RUNS_FILE = DATA_DIR / "vclone_runs.json"

HOST = "127.0.0.1"
PORT = 8760

KAGGLE_OWNER = os.environ.get("KAGGLE_USERNAME", "").strip()
KAGGLE_API_BASE = "https://www.kaggle.com/api/v1"
KAGGLE_GPU_WEEKLY_SEC = 108000

OMNIVOICE_DOCKER_IMAGE = (
    "gcr.io/kaggle-private-byod/python@sha256:"
    "de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01"
)
OMNIVOICE_MACHINE_SHAPE = "NvidiaTeslaT4"
OMNIVOICE_USE_UNIFIED_KERNEL = True

OMNIVOICE_DESIGN_KERNEL = {
    "slug": "yoyo-omnivoice",
    "id": 144856842,
    "title": "YoYo OmniVoice",
    "docker_image": OMNIVOICE_DOCKER_IMAGE,
    "machine_shape": OMNIVOICE_MACHINE_SHAPE,
    "pause_sec": 0.5,
    "default_speed": 0.85,
    "full_wav_name": "test_full.wav",
}

OMNIVOICE_CLONE_KERNEL = {
    "slug": "yoyo-omnivoice-clone",
    "id": 145372284,
    "title": "yoyo-omnivoice-clone",
    "docker_image": OMNIVOICE_DOCKER_IMAGE,
    "machine_shape": OMNIVOICE_MACHINE_SHAPE,
    "pause_sec": 0.4,
    "default_speed": 1.0,
    "full_wav_name": "clone_full.wav",
}

OMNIVOICE_CAPABILITIES = {
    "attrs": {
        "gender": [
            {"zh": "女", "en": "female"},
            {"zh": "男", "en": "male"},
        ],
        "age": [
            {"zh": "儿童", "en": "child"},
            {"zh": "少年", "en": "teenager"},
            {"zh": "青年", "en": "young adult"},
            {"zh": "中年", "en": "middle-aged"},
            {"zh": "老年", "en": "elderly"},
        ],
        "pitch": [
            {"zh": "极低音调", "en": "very low pitch"},
            {"zh": "低音调", "en": "low pitch"},
            {"zh": "中音调", "en": "moderate pitch"},
            {"zh": "高音调", "en": "high pitch"},
            {"zh": "极高音调", "en": "very high pitch"},
        ],
        "style": [
            {"zh": "耳语", "en": "whisper"},
        ],
        "dialect": [
            "四川话", "东北话", "陕西话", "河南话", "贵州话", "云南话",
            "桂林话", "济南话", "石家庄话", "甘肃话", "宁夏话", "青岛话",
        ],
        "accent": [
            "american accent", "british accent", "australian accent",
            "canadian accent", "indian accent", "chinese accent",
            "korean accent", "japanese accent", "portuguese accent",
            "russian accent",
        ],
    },
    "presets": [
        {"label": "女·青年·中音调（标准教学）", "instruct": "女，青年，中音调", "language": "Chinese"},
        {"label": "男·青年·低音调", "instruct": "男，青年，低音调", "language": "Chinese"},
        {"label": "女·青年·高音调（活泼）", "instruct": "女，青年，高音调", "language": "Chinese"},
        {"label": "男·中年·极低音调（新闻/旁白）", "instruct": "男，中年，极低音调", "language": "Chinese"},
        {"label": "女·儿童（童声故事）", "instruct": "女，儿童，高音调", "language": "Chinese"},
        {"label": "女·青年·耳语（助眠）", "instruct": "女，青年，耳语", "language": "Chinese"},
        {"label": "女·青年·四川话", "instruct": "女，青年，四川话", "language": "Chinese"},
        {"label": "男·中年·东北话", "instruct": "男，中年，东北话", "language": "Chinese"},
        {"label": "男·老年·陕西话", "instruct": "男，老年，陕西话", "language": "Chinese"},
        {"label": "Female·Young Adult·British Accent", "instruct": "female, young adult, moderate pitch, british accent", "language": "English"},
        {"label": "Male·High Pitch·Indian Accent", "instruct": "male, high pitch, indian accent", "language": "English"},
    ],
    "nonverbal_tags": [
        {"tag": "[laughter]", "label": "笑声"},
        {"tag": "[sigh]", "label": "叹气"},
        {"tag": "[confirmation-en]", "label": "嗯(确认)"},
        {"tag": "[question-en]", "label": "嗯?(疑问)"},
        {"tag": "[question-ah]", "label": "啊?(疑问)"},
        {"tag": "[question-oh]", "label": "哦?(疑问)"},
        {"tag": "[question-ei]", "label": "诶?(疑问)"},
        {"tag": "[question-yi]", "label": "咦?(疑问)"},
        {"tag": "[surprise-ah]", "label": "啊!(惊讶)"},
        {"tag": "[surprise-oh]", "label": "哦!(惊讶)"},
        {"tag": "[surprise-wa]", "label": "哇!(惊叹)"},
        {"tag": "[surprise-yo]", "label": "哟!(惊叹)"},
        {"tag": "[dissatisfaction-hnn]", "label": "哼(不满)"},
    ],
    "languages": [
        {"value": "", "label": "自动识别（不传 language）"},
        {"value": "Chinese", "label": "中文 (Chinese)"},
        {"value": "English", "label": "英文 (English)"},
        {"value": "Cantonese", "label": "粤语 (Cantonese)"},
        {"value": "Japanese", "label": "日语 (Japanese)"},
        {"value": "Korean", "label": "韩语 (Korean)"},
        {"value": "French", "label": "法语 (French)"},
        {"value": "German", "label": "德语 (German)"},
        {"value": "Spanish", "label": "西班牙语 (Spanish)"},
        {"value": "Russian", "label": "俄语 (Russian)"},
        {"value": "Arabic", "label": "阿拉伯语 (Arabic)"},
        {"value": "Portuguese", "label": "葡萄牙语 (Portuguese)"},
        {"value": "Italian", "label": "意大利语 (Italian)"},
        {"value": "Thai", "label": "泰语 (Thai)"},
        {"value": "Vietnamese", "label": "越南语 (Vietnamese)"},
        {"value": "Indonesian", "label": "印尼语 (Indonesian)"},
    ],
}

VCLONE_MAX_DURATION_SEC = 180.0

VCLONE_ENV_KERNEL = {
    "slug": "yoyo-vclone-env",
    "title": "yoyo-vclone-env",
    "machine_shape": "None",
}

VCLONE_PREP_KERNEL = {
    "slug": "yoyo-vclone-prep",
    "title": "yoyo-vclone-prep",
    "machine_shape": OMNIVOICE_MACHINE_SHAPE,
}

VCLONE_TTS_KERNEL = {
    "slug": "yoyo-vclone-tts",
    "title": "yoyo-vclone-tts",
    "machine_shape": OMNIVOICE_MACHINE_SHAPE,
}

VCLONE_CAPABILITIES = {
    "max_duration_sec": VCLONE_MAX_DURATION_SEC,
    "platforms": [
        {"id": "douyin", "label": "抖音 (Douyin)"},
        {"id": "bilibili", "label": "哔哩哔哩 (Bilibili)"},
        {"id": "tiktok", "label": "TikTok"},
        {"id": "youtube", "label": "YouTube"},
        {"id": "generic", "label": "其他平台 / 通用链接"},
    ],
    "separator_models": [
        {
            "value": "UVR-MDX-NET-Inst_HQ_3.onnx",
            "label": "UVR-MDX-NET-Inst_HQ_3 (推荐·速度快·人声干净)",
        },
        {
            "value": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "label": "BS-Roformer SDR 12.97 (高精分离·耗时较长)",
        },
    ],
    "whisper_models": [
        {"value": "large-v3-turbo", "label": "Faster-Whisper large-v3-turbo (推荐·快且准)"},
        {"value": "large-v3", "label": "Faster-Whisper large-v3 (最高精度)"},
        {"value": "medium", "label": "Faster-Whisper medium (轻量快速)"},
    ],
    "cosyvoice_modes": [
        {"value": "zero_shot", "label": "标准零样本克隆 (3~12秒复刻音色与韵律)"},
        {"value": "cross_lingual", "label": "跨语种翻译克隆 (保留原声线跨语言配音)"},
        {"value": "instruct2", "label": "情感/风格指令克隆 (支持语气与方言控制)"},
    ],
    "instruct_presets": [
        {"label": "自然口播（不加额外指令）", "instruct": ""},
        {"label": "激情带货 / 解说", "instruct": "用激情的语气说"},
        {"label": "温柔亲切", "instruct": "用温柔亲切的语气说"},
        {"label": "严肃新闻 / 播音腔", "instruct": "用沉稳正式的播音语气说"},
        {"label": "欢快活泼", "instruct": "用开心活泼的语气说"},
        {"label": "四川话方言", "instruct": "用四川话说"},
        {"label": "粤语方言", "instruct": "用粤语说"},
        {"label": "东北话方言", "instruct": "用东北话说"},
    ],
}

_SECRET_PATTERN = re.compile(r"KGAT_[A-Za-z0-9_-]{10,}")
_USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,63}$")


def ensure_dirs() -> None:
    for d in (
        DATA_DIR,
        VOICE_OUTPUTS_DIR,
        VOICE_LIBRARY_DIR,
        VIDEO_OUTPUTS_DIR,
        VCLONE_DIR,
        COOKIES_DIR,
        ASSETS_DIR / "workflows",
        ASSETS_DIR / "nodes",
        ASSETS_DIR / "refs",
    ):
        d.mkdir(parents=True, exist_ok=True)


def _restrict_windows_acl(path: Path) -> None:
    """Restrict Windows file ACL to the current user (best-effort)."""
    user = os.environ.get("USERNAME")
    if os.name == "nt" and user and path.exists():
        subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:(R,W)"],
            capture_output=True,
            check=False,
        )


def _write_secret_json_atomic(path: Path, data: Any) -> None:
    ensure_dirs()
    raw = json.dumps(data, indent=2, ensure_ascii=False)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(raw)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    _restrict_windows_acl(path)


def validate_kaggle_username(username: str) -> str:
    clean = (username or "").strip()
    if clean.startswith("https://") or clean.startswith("http://"):
        clean = clean.rstrip("/").split("/")[-1].strip()
    if not clean:
        raise ValueError("Kaggle 用户名不能为空。")
    if not _USERNAME_PATTERN.match(clean):
        raise ValueError("Kaggle 用户名格式不正确（请填写个人主页 URL kaggle.com/<用户名> 中的用户名，仅含字母/数字/下划线/短横线）。")
    return clean


def validate_kaggle_token(token: str) -> str:
    clean = (token or "").strip()
    if not clean.startswith("KGAT_"):
        raise ValueError("API Key (Token) 必须以 KGAT_ 开头。")
    return clean


def load_accounts_store() -> dict[str, Any]:
    if not ACCOUNTS_FILE.is_file():
        return {"active_username": None, "accounts": []}
    try:
        data = json.loads(ACCOUNTS_FILE.read_text(encoding="utf-8"))
        accounts = data.get("accounts") if isinstance(data, dict) else []
        if not isinstance(accounts, list):
            accounts = []
        active = data.get("active_username") if isinstance(data, dict) else None
        return {"active_username": active, "accounts": accounts}
    except Exception:
        return {"active_username": None, "accounts": []}


def save_accounts_store(store: dict[str, Any]) -> None:
    _write_secret_json_atomic(ACCOUNTS_FILE, store)


def get_kaggle_account(username: str | None = None) -> dict[str, Any] | None:
    store = load_accounts_store()
    accounts = store["accounts"]
    target = (username or store.get("active_username") or "").strip()
    if target:
        for acct in accounts:
            if acct.get("username") == target:
                return dict(acct)
    if username is None and accounts:
        return dict(accounts[0])
    # Fallback to env vars only when no multi-account config is present
    if username is None and not accounts:
        env_tok = os.environ.get("KAGGLE_API_TOKEN", "").strip()
        env_user = os.environ.get("KAGGLE_USERNAME", "").strip() or KAGGLE_OWNER
        if env_tok and env_user:
            return {
                "username": env_user,
                "token": env_tok,
                "label": "环境变量",
                "use_pinned_image": False,
            }
    return None


def list_kaggle_accounts_public() -> list[dict[str, Any]]:
    store = load_accounts_store()
    active = store.get("active_username")
    out: list[dict[str, Any]] = []
    for i, acct in enumerate(store["accounts"]):
        uname = acct.get("username", "")
        is_active = uname == active if active else i == 0
        out.append(
            {
                "username": uname,
                "label": acct.get("label") or "",
                "token_masked": mask_token(acct.get("token")),
                "use_pinned_image": bool(acct.get("use_pinned_image", False)),
                "deployed": bool(acct.get("deployed", False)),
                "deployed_at": acct.get("deployed_at") or "",
                "deploy_info": acct.get("deploy_info"),
                "created_at": acct.get("created_at") or "",
                "active": is_active,
            }
        )
    return out


def upsert_kaggle_account(
    *,
    username: str,
    token: str,
    label: str = "",
    use_pinned_image: bool = False,
    activate: bool = True,
) -> dict[str, Any]:
    uname = validate_kaggle_username(username)
    tok = validate_kaggle_token(token)
    clean_label = (label or "").strip()
    store = load_accounts_store()
    accounts: list[dict[str, Any]] = store["accounts"]

    entry = {
        "username": uname,
        "token": tok,
        "label": clean_label,
        "use_pinned_image": bool(use_pinned_image),
        "deployed": False,
        "deployed_at": "",
        "deploy_info": None,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    replaced = False
    for i, item in enumerate(accounts):
        if item.get("username") == uname:
            entry["created_at"] = item.get("created_at") or entry["created_at"]
            entry["deployed"] = bool(item.get("deployed", False))
            entry["deployed_at"] = item.get("deployed_at") or ""
            entry["deploy_info"] = item.get("deploy_info")
            accounts[i] = entry
            replaced = True
            break
    if not replaced:
        accounts.append(entry)

    if activate or not store.get("active_username"):
        store["active_username"] = uname
    save_accounts_store(store)
    return entry


def update_kaggle_account_deploy(
    username: str,
    *,
    deployed: bool,
    deploy_info: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    uname = (username or "").strip()
    store = load_accounts_store()
    updated = None
    for item in store["accounts"]:
        if item.get("username") == uname:
            item["deployed"] = bool(deployed)
            item["deployed_at"] = time.strftime("%Y-%m-%d %H:%M:%S") if deployed else ""
            if deploy_info is not None:
                item["deploy_info"] = deploy_info
            updated = dict(item)
            break
    if updated is not None:
        save_accounts_store(store)
    return updated


def switch_kaggle_account(username: str) -> dict[str, Any]:
    uname = validate_kaggle_username(username)
    store = load_accounts_store()
    found = next((a for a in store["accounts"] if a.get("username") == uname), None)
    if found is None:
        raise KeyError(f"未找到 Kaggle 账号 {uname}")
    store["active_username"] = uname
    save_accounts_store(store)
    return dict(found)


def delete_kaggle_account(username: str) -> str | None:
    uname = (username or "").strip()
    store = load_accounts_store()
    accounts = [a for a in store["accounts"] if a.get("username") != uname]
    store["accounts"] = accounts
    if store.get("active_username") == uname:
        store["active_username"] = accounts[0]["username"] if accounts else None
    elif store.get("active_username") and not any(a.get("username") == store["active_username"] for a in accounts):
        store["active_username"] = accounts[0]["username"] if accounts else None
    save_accounts_store(store)
    return store.get("active_username")


def load_kaggle_owner() -> str | None:
    acct = get_kaggle_account()
    if acct and acct.get("username"):
        return str(acct["username"])
    return None


def load_kaggle_token() -> str | None:
    acct = get_kaggle_account()
    if acct and acct.get("token"):
        return str(acct["token"])
    env_tok = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if env_tok:
        return env_tok
    if TOKEN_FILE.is_file():
        tok = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    return None


def save_kaggle_token(token: str) -> None:
    ensure_dirs()
    clean = validate_kaggle_token(token)
    TOKEN_FILE.write_text(clean, encoding="utf-8")
    _restrict_windows_acl(TOKEN_FILE)


def mask_token(token: str | None) -> str:
    if not token:
        return "未配置"
    if len(token) <= 10:
        return "KGAT_***"
    return f"{token[:8]}...{token[-4:]}"


def scrub_secrets(text: str) -> str:
    secrets: list[str] = []
    env_tok = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if env_tok:
        secrets.append(env_tok)
    if TOKEN_FILE.is_file():
        t = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if t:
            secrets.append(t)
    for acct in load_accounts_store()["accounts"]:
        t = (acct.get("token") or "").strip()
        if t:
            secrets.append(t)
    for tok in set(secrets):
        text = text.replace(tok, "<KGAT_TOKEN>")
    return _SECRET_PATTERN.sub("<KGAT_TOKEN>", text)
