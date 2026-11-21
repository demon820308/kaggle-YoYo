"""Multi-platform video URL resolver, cookies manager, and yt-dlp downloader."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from server.config import (
    COOKIES_DIR,
    VCLONE_MAX_DURATION_SEC,
    _restrict_windows_acl,
    ensure_dirs,
)
from server.storage import load_settings

_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)

PLATFORM_DOMAINS: dict[str, str] = {
    "douyin.com": "douyin",
    "iesdouyin.com": "douyin",
    "tiktok.com": "tiktok",
    "bilibili.com": "bilibili",
    "b23.tv": "bilibili",
    "youtube.com": "youtube",
    "youtu.be": "youtube",
    "kuaishou.com": "kuaishou",
    "xiaohongshu.com": "xiaohongshu",
    "xhslink.com": "xiaohongshu",
}

PLATFORM_LABELS: dict[str, str] = {
    "douyin": "抖音 (Douyin)",
    "tiktok": "TikTok",
    "bilibili": "哔哩哔哩 (Bilibili)",
    "youtube": "YouTube",
    "kuaishou": "快手 (Kuaishou)",
    "xiaohongshu": "小红书 (Xiaohongshu)",
    "generic": "通用视频链接",
    "local": "本地上传视频",
}

PLATFORM_DEFAULT_DOMAINS: dict[str, str] = {
    "douyin": ".douyin.com",
    "tiktok": ".tiktok.com",
    "bilibili": ".bilibili.com",
    "youtube": ".youtube.com",
    "generic": ".example.com",
}

OVERSEAS_PLATFORMS = {"youtube", "tiktok"}
_DOUYIN_SHORTLINK_CACHE: dict[str, tuple[str, str]] = {}


def extract_url_from_text(raw: str) -> str:
    """Extract the first http(s) URL from user input (supports share copy text from Douyin/Bilibili/TikTok)."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("请输入视频链接或分享文本。")
    m = _URL_RE.search(text)
    if not m:
        if re.match(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/", text):
            return "https://" + text.split()[0]
        raise ValueError("未在输入中识别到有效的 http/https 视频链接。")
    url = m.group(0).rstrip(")】》!！，,。.;；")
    return url


def detect_platform(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        host = ""
    for domain, plat in PLATFORM_DOMAINS.items():
        if host == domain or host.endswith("." + domain):
            return plat
    return "generic"


def get_cookie_file_path(platform: str) -> Path:
    ensure_dirs()
    safe = re.sub(r"[^a-z0-9_-]", "", (platform or "generic").lower()) or "generic"
    return COOKIES_DIR / f"{safe}.txt"


def convert_to_netscape_cookies(raw_text: str, platform: str) -> str:
    """Accept either Netscape cookies.txt content or a raw 'Cookie: a=b; c=d' string."""
    txt = (raw_text or "").strip()
    if not txt:
        return ""
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    # Check if already Netscape format (tab-separated lines with 7 columns)
    tab_lines = [ln for ln in lines if not ln.startswith("#") and ln.count("\t") >= 6]
    if tab_lines:
        header = "# Netscape HTTP Cookie File\n# Auto-saved by YoYo Kaggle Console\n"
        body = "\n".join(ln for ln in txt.splitlines() if not ln.startswith("# Netscape"))
        return header + body.strip() + "\n"

    # Try JSON array format (e.g. EditThisCookie / Cookie-Editor export)
    if txt.startswith("[") and txt.endswith("]"):
        try:
            arr = json.loads(txt)
            if isinstance(arr, list):
                out = ["# Netscape HTTP Cookie File"]
                for item in arr:
                    if not isinstance(item, dict) or not item.get("name"):
                        continue
                    dom = str(item.get("domain") or PLATFORM_DEFAULT_DOMAINS.get(platform, ".douyin.com"))
                    include_sub = "TRUE" if dom.startswith(".") else "FALSE"
                    path = str(item.get("path") or "/")
                    sec = "TRUE" if item.get("secure") else "FALSE"
                    exp = int(item.get("expirationDate") or (time.time() + 86400 * 30))
                    name = str(item["name"])
                    val = str(item.get("value") or "")
                    out.append(f"{dom}\t{include_sub}\t{path}\t{sec}\t{exp}\t{name}\t{val}")
                if len(out) > 1:
                    return "\n".join(out) + "\n"
        except Exception:
            pass

    # Treat as raw HTTP Cookie header: "Cookie: k1=v1; k2=v2"
    if txt.lower().startswith("cookie:"):
        txt = txt[7:].strip()
    default_domain = PLATFORM_DEFAULT_DOMAINS.get(platform, ".douyin.com")
    exp = int(time.time() + 86400 * 30)
    out = ["# Netscape HTTP Cookie File"]
    for pair in txt.replace("\n", ";").split(";"):
        pair = pair.strip()
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k:
            continue
        out.append(f"{default_domain}\tTRUE\t/\tTRUE\t{exp}\t{k}\t{v}")
        if platform == "douyin":
            out.append(f".iesdouyin.com\tTRUE\t/\tTRUE\t{exp}\t{k}\t{v}")
    if len(out) <= 1:
        raise ValueError("无法识别该 Cookie 格式。请粘贴 Netscape 格式 cookies.txt、Cookie-Editor JSON 或浏览器 F12 中的 Cookie 字符串（key=value; ...）。")
    return "\n".join(out) + "\n"


def save_platform_cookies(platform: str, raw_text: str) -> dict[str, Any]:
    path = get_cookie_file_path(platform)
    if not (raw_text or "").strip():
        path.unlink(missing_ok=True)
        return {"platform": platform, "configured": False, "cookie_count": 0}
    netscape = convert_to_netscape_cookies(raw_text, platform)
    path.write_text(netscape, encoding="utf-8")
    _restrict_windows_acl(path)
    count = sum(1 for ln in netscape.splitlines() if ln.strip() and not ln.startswith("#"))
    return {"platform": platform, "configured": True, "cookie_count": count}


def list_cookies_status() -> dict[str, dict[str, Any]]:
    ensure_dirs()
    status: dict[str, dict[str, Any]] = {}
    for plat in ("douyin", "bilibili", "tiktok", "youtube", "generic"):
        p = get_cookie_file_path(plat)
        if p.is_file():
            txt = p.read_text(encoding="utf-8", errors="replace")
            cnt = sum(1 for ln in txt.splitlines() if ln.strip() and not ln.startswith("#"))
            status[plat] = {
                "configured": cnt > 0,
                "cookie_count": cnt,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)),
            }
        else:
            status[plat] = {"configured": False, "cookie_count": 0, "updated_at": ""}
    return status


def _effective_proxy(platform: str, settings: dict[str, Any]) -> str | None:
    proxy_url = (settings.get("proxy_url") or "").strip()
    if not proxy_url:
        return None
    mode = settings.get("proxy_mode") or "overseas"
    if mode == "none":
        return None
    if mode == "all":
        return proxy_url
    if mode == "overseas" and platform in OVERSEAS_PLATFORMS:
        return proxy_url
    return None


def _build_ydl_opts(platform: str, settings: dict[str, Any], *, for_download: bool = False, out_tmpl: str | None = None) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 25,
        "retries": 3,
        "nocheckcertificate": True,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
        },
    }
    if platform == "bilibili":
        opts["http_headers"]["Referer"] = "https://www.bilibili.com/"
    elif platform == "douyin":
        opts["http_headers"]["Referer"] = "https://www.douyin.com/"
    elif platform == "tiktok":
        opts["http_headers"]["Referer"] = "https://www.tiktok.com/"

    proxy = _effective_proxy(platform, settings)
    if proxy:
        opts["proxy"] = proxy

    cookie_file = get_cookie_file_path(platform)
    if cookie_file.is_file() and cookie_file.stat().st_size > 30:
        opts["cookiefile"] = str(cookie_file)
    else:
        browser = (settings.get("cookies_browser") or "").strip().lower()
        if browser in ("chrome", "edge", "firefox"):
            opts["cookiesfrombrowser"] = (browser,)

    if for_download and out_tmpl:
        opts["format"] = "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b"
        opts["merge_output_format"] = "mp4"
        opts["outtmpl"] = out_tmpl

    return opts


def _resolve_douyin_redirect_and_normalize(url: str, proxy: str | None = None) -> tuple[str, str | None]:
    """Resolve v.douyin.com / iesdouyin short links to canonical douyin.com/video/<aweme_id>."""
    if url in _DOUYIN_SHORTLINK_CACHE:
        return _DOUYIN_SHORTLINK_CACHE[url]

    m_direct = re.search(r"/(?:video|note|slides)/(\d{15,22})", url) or re.search(r"modal_id=(\d{15,22})", url)
    if m_direct:
        aweme_id = m_direct.group(1)
        res = (f"https://www.douyin.com/video/{aweme_id}", aweme_id)
        _DOUYIN_SHORTLINK_CACHE[url] = res
        return res

    try:
        from curl_cffi import requests as cffi_requests
        sess = cffi_requests.Session(impersonate="chrome")
        proxies = {"http": proxy, "https": proxy} if proxy else None
        r = sess.get(url, allow_redirects=True, timeout=15, proxies=proxies)
        final_url = str(r.url)
    except Exception:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                final_url = resp.geturl()
        except Exception:
            final_url = url

    m = re.search(r"/(?:video|note|slides)/(\d{15,22})", final_url) or re.search(r"modal_id=(\d{15,22})", final_url)
    if not m:
        return final_url, None

    aweme_id = m.group(1)
    res = (f"https://www.douyin.com/video/{aweme_id}", aweme_id)
    _DOUYIN_SHORTLINK_CACHE[url] = res
    return res


def _sync_resolve_url(raw_input: str) -> dict[str, Any]:
    import yt_dlp

    url = extract_url_from_text(raw_input)
    platform = detect_platform(url)
    settings = load_settings()
    proxy = _effective_proxy(platform, settings)

    douyin_aweme_id = None
    douyin_fb = None
    if platform == "douyin":
        url, douyin_aweme_id = _resolve_douyin_redirect_and_normalize(url, proxy=proxy)
        if douyin_aweme_id:
            from server.vclone.cookie_harvester import extract_douyin_aweme_via_cdp
            douyin_fb = extract_douyin_aweme_via_cdp(douyin_aweme_id, proxy_url=proxy)

    info = douyin_fb
    ydl_err = None
    if info is None:
        opts = _build_ydl_opts(platform, settings, for_download=False)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as exc:
            ydl_err = str(exc)

    # Automatic silent cookie harvest & retry when yt-dlp fails on other platforms
    if info is None and platform in ("bilibili", "tiktok", "youtube"):
        try:
            from server.vclone.cookie_harvester import _sync_harvest_cookies
            _sync_harvest_cookies(platform, target_url=url, headless=True, proxy_url=proxy)
            opts_retry = _build_ydl_opts(platform, settings, for_download=False)
            with yt_dlp.YoutubeDL(opts_retry) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as retry_exc:
            ydl_err = ydl_err or str(retry_exc)

    if info is None:
        cookie_file = get_cookie_file_path(platform)
        has_ck = cookie_file.is_file() and cookie_file.stat().st_size > 30
        hint = ""
        if platform == "douyin" and not has_ck:
            hint = "（提示：抖音风控较严，建议点击右上角「Cookies 与代理配置」粘贴抖音 Cookie 后重试）"
        elif platform == "youtube" and not has_ck:
            hint = "（提示：若 YouTube 提示 Sign in / bot 校验或网络超时，请在「Cookies 与代理配置」中配置代理或 YouTube Cookies）"
        elif platform == "bilibili" and not has_ck:
            hint = "（提示：可在「Cookies 与代理配置」中粘贴 B 站 Cookie 以支持更多视频）"
        raise ValueError(f"解析视频链接失败：{ydl_err or '未知错误'} {hint}".strip())

    if "entries" in info and isinstance(info["entries"], list) and info["entries"]:
        info = info["entries"][0]

    dur = float(info.get("duration") or 0.0)
    cookie_status = list_cookies_status().get(platform, {"configured": False})
    return {
        "url": info.get("webpage_url") or url,
        "original_input": raw_input,
        "platform": platform,
        "platform_label": PLATFORM_LABELS.get(platform, platform),
        "video_id": str(info.get("id") or ""),
        "title": (info.get("title") or "未命名视频").strip(),
        "uploader": (info.get("uploader") or info.get("channel") or "").strip(),
        "duration": round(dur, 2),
        "thumbnail": info.get("thumbnail") or "",
        "exceeds_limit": dur > VCLONE_MAX_DURATION_SEC,
        "max_duration_sec": VCLONE_MAX_DURATION_SEC,
        "recommended_clip_start": 0.0,
        "recommended_clip_end": round(min(dur, VCLONE_MAX_DURATION_SEC), 1) if dur > 0 else VCLONE_MAX_DURATION_SEC,
        "cookies_configured": bool(cookie_status.get("configured")),
        "used_fallback_resolver": bool(douyin_fb and info is douyin_fb),
    }


def _sync_download_url(raw_input: str, out_dir: Path) -> tuple[Path, dict[str, Any]]:
    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)
    url = extract_url_from_text(raw_input)
    platform = detect_platform(url)
    settings = load_settings()
    proxy = _effective_proxy(platform, settings)

    douyin_aweme_id = None
    douyin_fb = None
    if platform == "douyin":
        url, douyin_aweme_id = _resolve_douyin_redirect_and_normalize(url, proxy=proxy)
        if douyin_aweme_id:
            from server.vclone.cookie_harvester import extract_douyin_aweme_via_cdp
            douyin_fb = extract_douyin_aweme_via_cdp(douyin_aweme_id, proxy_url=proxy)

    out_tmpl = str(out_dir / "raw_video.%(ext)s")
    opts = _build_ydl_opts(platform, settings, for_download=True, out_tmpl=out_tmpl)

    info = None
    ydl_err = None
    downloaded: list[Path] = []

    if douyin_fb and douyin_fb.get("direct_video_urls"):
        from curl_cffi import requests as cffi_requests
        dest = out_dir / "raw_video.mp4"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.douyin.com/",
        }
        for d_url in douyin_fb["direct_video_urls"]:
            try:
                r = cffi_requests.get(d_url, headers=headers, impersonate="chrome", timeout=120)
                if r.status_code in (200, 206) and len(r.content) > 10000:
                    dest.write_bytes(r.content)
                    downloaded = [dest]
                    info = douyin_fb
                    break
            except Exception:
                continue

    if not downloaded:
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except Exception as exc:
            ydl_err = str(exc)
        downloaded = [p for p in out_dir.glob("raw_video.*") if p.is_file() and p.suffix.lower() not in (".part", ".ytdl")]

    if not downloaded and platform in ("bilibili", "tiktok", "youtube"):
        try:
            from server.vclone.cookie_harvester import _sync_harvest_cookies
            _sync_harvest_cookies(platform, target_url=url, headless=True, proxy_url=proxy)
            opts_retry = _build_ydl_opts(platform, settings, for_download=True, out_tmpl=out_tmpl)
            with yt_dlp.YoutubeDL(opts_retry) as ydl:
                info = ydl.extract_info(url, download=True)
            downloaded = [p for p in out_dir.glob("raw_video.*") if p.is_file() and p.suffix.lower() not in (".part", ".ytdl")]
        except Exception as retry_exc:
            ydl_err = ydl_err or str(retry_exc)

    if not downloaded and douyin_fb and douyin_fb.get("direct_video_urls"):
        from curl_cffi import requests as cffi_requests
        dest = out_dir / "raw_video.mp4"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.douyin.com/",
        }
        for d_url in douyin_fb["direct_video_urls"]:
            try:
                r = cffi_requests.get(d_url, headers=headers, impersonate="chrome", timeout=120)
                if r.status_code in (200, 206) and len(r.content) > 10000:
                    dest.write_bytes(r.content)
                    downloaded = [dest]
                    info = douyin_fb
                    break
            except Exception:
                continue

    if not downloaded:
        raise ValueError(f"下载视频失败：{ydl_err or '未找到下载后的视频文件'}。请检查链接、代理或平台 Cookies 设置。")

    video_file = sorted(downloaded, key=lambda p: p.stat().st_size, reverse=True)[0]
    if info and "entries" in info and isinstance(info["entries"], list) and info["entries"]:
        info = info["entries"][0]

    meta = {
        "url": (info or {}).get("webpage_url") or url,
        "platform": platform,
        "platform_label": PLATFORM_LABELS.get(platform, platform),
        "title": ((info or {}).get("title") or video_file.stem).strip(),
        "uploader": ((info or {}).get("uploader") or "").strip(),
        "duration": round(float((info or {}).get("duration") or 0.0), 2),
        "thumbnail": (info or {}).get("thumbnail") or "",
        "raw_filename": video_file.name,
        "raw_bytes": video_file.stat().st_size,
    }
    return video_file, meta


async def resolve_video_url(raw_input: str) -> dict[str, Any]:
    return await asyncio.to_thread(_sync_resolve_url, raw_input)


async def download_video_url(raw_input: str, out_dir: Path) -> tuple[Path, dict[str, Any]]:
    return await asyncio.to_thread(_sync_download_url, raw_input, out_dir)
