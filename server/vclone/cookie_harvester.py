"""Automatic Cookie Harvester and Douyin Signed-API Extractor using local Chrome/Edge via CDP.

1. Bypasses Windows Chrome/Edge SQLite file locks and v20 App-Bound Encryption by reading
   already-decrypted cookies directly from a dedicated persistent profile via CDP Network.getAllCookies.
2. Bypasses Douyin's 2026 a_bogus + x-secsdk-web-signature check (which yt-dlp's DouyinIE lacks)
   by Spoofing Client Hints (Sec-CH-UA Google Chrome) over a persistent CDP session and intercepting
   the browser-signed /aweme/v1/web/aweme/detail/ response directly.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from server.config import COOKIES_DIR, _restrict_windows_acl, ensure_dirs

BROWSER_PROFILE_DIR = COOKIES_DIR / "chrome_profile"
_DOUYIN_DETAIL_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CDP_LOCK = threading.Lock()

CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

CHROME_UA_METADATA = {
    "brands": [
        {"brand": "Google Chrome", "version": "131"},
        {"brand": "Chromium", "version": "131"},
        {"brand": "Not_A Brand", "version": "24"},
    ],
    "fullVersionList": [
        {"brand": "Google Chrome", "version": "131.0.6778.86"},
        {"brand": "Chromium", "version": "131.0.6778.86"},
        {"brand": "Not_A Brand", "version": "24.0.0.0"},
    ],
    "platform": "Windows",
    "platformVersion": "15.0.0",
    "architecture": "x86",
    "model": "",
    "mobile": False,
}

PLATFORM_HOME_URLS: dict[str, str] = {
    "douyin": "https://www.douyin.com/",
    "bilibili": "https://www.bilibili.com/",
    "tiktok": "https://www.tiktok.com/",
    "youtube": "https://www.youtube.com/",
    "generic": "https://www.bilibili.com/",
}

PLATFORM_DOMAIN_MATCH: dict[str, tuple[str, ...]] = {
    "douyin": ("douyin.com", "iesdouyin.com"),
    "bilibili": ("bilibili.com", "bilivideo.com"),
    "tiktok": ("tiktok.com",),
    "youtube": ("youtube.com", "google.com"),
    "generic": (),
}

PLATFORM_AUTH_COOKIES: dict[str, tuple[str, ...]] = {
    "douyin": ("sessionid", "sessionid_ss", "uid_tt", "LOGIN_STATUS"),
    "bilibili": ("SESSDATA", "DedeUserID"),
    "tiktok": ("sessionid", "sessionid_ss", "sid_tt"),
    "youtube": ("SID", "SSID", "HSID", "__Secure-3PSID"),
}

PLATFORM_GUEST_COOKIES: dict[str, tuple[str, ...]] = {
    "douyin": ("ttwid", "s_v_web_id", "__ac_nonce", "uifid"),
    "bilibili": ("buvid3", "b_nut"),
    "tiktok": ("ttwid", "tt_csrf_token", "msToken"),
    "youtube": ("VISITOR_INFO1_LIVE", "YSC", "PREF"),
}


def find_browser_executable() -> str | None:
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    for name in ("chrome", "msedge", "google-chrome", "chromium"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _kill_proc_tree(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is None:
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    timeout=5,
                )
            else:
                proc.terminate()
                proc.wait(timeout=4)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


class CDPSession:
    """Persistent RFC-6455 WebSocket session to Chrome DevTools Protocol over 127.0.0.1."""

    def __init__(self, ws_url: str, timeout: float = 15.0) -> None:
        u = urlparse(ws_url)
        host = u.hostname or "127.0.0.1"
        port = u.port or 80
        path = u.path + (("?" + u.query) if u.query else "")
        self.sock = socket.create_connection((host, port), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        handshake = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(handshake.encode("ascii"))
        resp_buf = b""
        while b"\r\n\r\n" not in resp_buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise RuntimeError("CDP WebSocket handshake closed early")
            resp_buf += chunk
        self.leftover = resp_buf.split(b"\r\n\r\n", 1)[1]
        self._id = 0
        self.aweme_request_ids: list[str] = []

    def _read_exact(self, n: int) -> bytes:
        while len(self.leftover) < n:
            c = self.sock.recv(65536)
            if not c:
                raise RuntimeError("CDP WebSocket closed before response")
            self.leftover += c
        res = self.leftover[:n]
        self.leftover = self.leftover[n:]
        return res

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._id += 1
        cid = self._id
        payload = json.dumps({"id": cid, "method": method, "params": params or {}}).encode("utf-8")
        mask_key = os.urandom(4)
        header = bytearray([0x81])
        plen = len(payload)
        if plen < 126:
            header.append(0x80 | plen)
        elif plen < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", plen))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", plen))
        header.extend(mask_key)
        masked = bytes(payload[i] ^ mask_key[i & 3] for i in range(plen))
        self.sock.sendall(bytes(header) + masked)

        while True:
            b0, b1 = self._read_exact(2)
            opcode = b0 & 0x0F
            is_masked = bool(b1 & 0x80)
            length = b1 & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._read_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._read_exact(8))[0]
            mkey = self._read_exact(4) if is_masked else b""
            data = bytearray(self._read_exact(length))
            if is_masked:
                for i in range(length):
                    data[i] ^= mkey[i & 3]
            if opcode == 0x8:
                raise RuntimeError("CDP WebSocket closed by browser")
            if opcode == 0x1:
                msg = json.loads(data.decode("utf-8", errors="replace"))
                method_ev = msg.get("method")
                if method_ev == "Network.responseReceived":
                    resp_url = str(((msg.get("params") or {}).get("response") or {}).get("url") or "")
                    req_id = str((msg.get("params") or {}).get("requestId") or "")
                    if req_id and "/aweme/v1/web/aweme/detail/" in resp_url:
                        if req_id not in self.aweme_request_ids:
                            self.aweme_request_ids.append(req_id)
                if msg.get("id") == cid:
                    return msg.get("result") or {}

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


def _wait_for_page_ws(port: int, proc: subprocess.Popen[Any], timeout: float = 12.0) -> str | None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return None
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2) as r:
                tabs = json.loads(r.read().decode("utf-8", errors="replace"))
            for tab in tabs:
                if tab.get("type") == "page" and tab.get("webSocketDebuggerUrl"):
                    return str(tab["webSocketDebuggerUrl"])
            for tab in tabs:
                if tab.get("webSocketDebuggerUrl"):
                    return str(tab["webSocketDebuggerUrl"])
        except Exception:
            pass
        time.sleep(0.3)
    return None


def _filter_and_format_netscape(cdp_cookies: list[dict[str, Any]], platform: str) -> tuple[str, list[str]]:
    domains = PLATFORM_DOMAIN_MATCH.get(platform, ())
    lines = ["# Netscape HTTP Cookie File", f"# Auto-harvested for {platform} by YoYo Kaggle Console"]
    names: list[str] = []
    seen_keys: set[tuple[str, str]] = set()

    for c in cdp_cookies:
        dom = str(c.get("domain") or "").strip()
        name = str(c.get("name") or "").strip()
        val = str(c.get("value") or "").strip()
        if not dom or not name:
            continue
        if domains and not any(dom == d or dom.endswith("." + d) or d in dom for d in domains):
            continue
        if not dom.startswith("."):
            dom = "." + dom
        include_sub = "TRUE"
        path = str(c.get("path") or "/")
        sec = "TRUE" if c.get("secure") else "FALSE"
        exp_raw = float(c.get("expires") or 0)
        exp = int(exp_raw) if exp_raw > 0 else int(time.time() + 86400 * 30)

        if (dom, name) not in seen_keys:
            lines.append(f"{dom}\t{include_sub}\t{path}\t{sec}\t{exp}\t{name}\t{val}")
            seen_keys.add((dom, name))
            names.append(name)

        # Mirror .iesdouyin.com cookies (such as ttwid) onto .douyin.com as well
        if platform == "douyin" and "iesdouyin.com" in dom and (".douyin.com", name) not in seen_keys:
            lines.append(f".douyin.com\tTRUE\t{path}\t{sec}\t{exp}\t{name}\t{val}")
            seen_keys.add((".douyin.com", name))

    return "\n".join(lines) + "\n", names


def _extract_douyin_once(aweme_id: str, profile_dir: Path, proxy_url: str | None = None) -> dict[str, Any] | None:
    browser_exe = find_browser_executable()
    if not browser_exe:
        return None

    port = _get_free_port()
    args = [
        browser_exe,
        f"--remote-debugging-port={port}",
        "--remote-allow-origins=*",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-popup-blocking",
        "--disable-blink-features=AutomationControlled",
        f"--user-agent={CHROME_UA}",
        "--headless=new",
        "--mute-audio",
        "--window-size=1280,800",
        "about:blank",
    ]
    if proxy_url:
        args.insert(-1, f"--proxy-server={proxy_url}")

    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cdp: CDPSession | None = None
    try:
        ws_url = _wait_for_page_ws(port, proc, timeout=8.0)
        if not ws_url:
            return None
        cdp = CDPSession(ws_url, timeout=10.0)
        cdp.call("Page.enable")
        # Deliberately do NOT call Network.enable (prevents 1500+ Network.* events from filling the 64KB socket buffer)
        cdp.call(
            "Emulation.setUserAgentOverride",
            {
                "userAgent": CHROME_UA,
                "platform": "Win32",
                "userAgentMetadata": CHROME_UA_METADATA,
            },
        )
        # Hook JSON.parse and Response.prototype.json WITHOUT touching XMLHttpRequest or window.fetch
        # so Douyin's bdms.js a_bogus + x-secsdk-web-signature signer stays 100% undisturbed
        cdp.call(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": """
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                window.__AWEME_DETAIL__ = null;
                const origParse = JSON.parse;
                JSON.parse = function(text, reviver) {
                    const res = origParse.call(this, text, reviver);
                    try {
                        if (res && typeof res === 'object' && res.aweme_detail && res.aweme_detail.video) {
                            window.__AWEME_DETAIL__ = res.aweme_detail;
                        }
                    } catch(e) {}
                    return res;
                };
                const origRespJson = Response.prototype.json;
                Response.prototype.json = async function() {
                    const res = await origRespJson.apply(this, arguments);
                    try {
                        if (res && typeof res === 'object' && res.aweme_detail && res.aweme_detail.video) {
                            window.__AWEME_DETAIL__ = res.aweme_detail;
                        }
                    } catch(e) {}
                    return res;
                };
                """
            },
        )
        canonical = f"https://www.douyin.com/video/{aweme_id}"
        cdp.call("Page.navigate", {"url": canonical})

        detail: dict[str, Any] | None = None
        for _ in range(16):
            time.sleep(0.4)
            r = cdp.call(
                "Runtime.evaluate",
                {"expression": "JSON.stringify(window.__AWEME_DETAIL__)", "returnByValue": True},
            )
            val = (r.get("result") or {}).get("value")
            if val and val != "null":
                detail = json.loads(val)
                break

        # Also save any fresh cookies gathered during this visit
        try:
            ck_res = cdp.call("Network.getAllCookies")
            netscape_txt, names = _filter_and_format_netscape(ck_res.get("cookies") or [], "douyin")
            if names:
                cookie_path = COOKIES_DIR / "douyin.txt"
                cookie_path.write_text(netscape_txt, encoding="utf-8")
                _restrict_windows_acl(cookie_path)
        except Exception:
            pass

        if not detail:
            return None

        vid = detail.get("video") or {}
        play_urls: list[str] = []
        # Put standard H.264 play_addr first, then bit_rate URLs
        for u in ((vid.get("play_addr") or {}).get("url_list") or []):
            if u and u not in play_urls:
                play_urls.append(u)
        for u in ((vid.get("play_addr_h264") or {}).get("url_list") or []):
            if u and u not in play_urls:
                play_urls.append(u)
        for br in (vid.get("bit_rate") or []):
            u_list = ((br.get("play_addr") or {}).get("url_list")) or []
            for u in u_list:
                if u and u not in play_urls:
                    play_urls.append(u)

        cover_list = ((vid.get("cover") or {}).get("url_list")) or ((vid.get("origin_cover") or {}).get("url_list")) or []
        dur_ms = float(vid.get("duration") or detail.get("duration") or 0.0)
        dur_sec = round(dur_ms / 1000.0, 2) if dur_ms > 500 else round(dur_ms, 2)

        return {
            "id": aweme_id,
            "title": (detail.get("desc") or f"抖音视频_{aweme_id}").strip(),
            "uploader": ((detail.get("author") or {}).get("nickname") or "抖音用户").strip(),
            "duration": dur_sec,
            "thumbnail": cover_list[0] if cover_list else "",
            "direct_video_urls": play_urls,
            "webpage_url": canonical,
        }
    except Exception:
        return None
    finally:
        if cdp:
            cdp.close()
        _kill_proc_tree(proc)


def extract_douyin_aweme_via_cdp(aweme_id: str, proxy_url: str | None = None) -> dict[str, Any] | None:
    """Extract Douyin video metadata & watermark-free MP4 stream URLs via Chrome CDP,
    with 10-minute in-memory cache and thread-safe profile locking."""
    ensure_dirs()
    cached = _DOUYIN_DETAIL_CACHE.get(aweme_id)
    if cached and (time.time() - cached[0]) < 600:
        return dict(cached[1])

    with _CDP_LOCK:
        cached = _DOUYIN_DETAIL_CACHE.get(aweme_id)
        if cached and (time.time() - cached[0]) < 600:
            return dict(cached[1])

        BROWSER_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        res = _extract_douyin_once(aweme_id, BROWSER_PROFILE_DIR, proxy_url=proxy_url)
        if res is None:
            with tempfile.TemporaryDirectory(prefix="yoyo_dy_cdp_") as td:
                res = _extract_douyin_once(aweme_id, Path(td), proxy_url=proxy_url)

        if res is not None:
            _DOUYIN_DETAIL_CACHE[aweme_id] = (time.time(), dict(res))
        return res


def _sync_harvest_cookies(
    platform: str,
    *,
    target_url: str | None = None,
    headless: bool = True,
    wait_login_sec: int = 90,
    proxy_url: str | None = None,
) -> dict[str, Any]:
    with _CDP_LOCK:
        return _sync_harvest_cookies_locked(
            platform,
            target_url=target_url,
            headless=headless,
            wait_login_sec=wait_login_sec,
            proxy_url=proxy_url,
        )


def _sync_harvest_cookies_locked(
    platform: str,
    *,
    target_url: str | None = None,
    headless: bool = True,
    wait_login_sec: int = 90,
    proxy_url: str | None = None,
) -> dict[str, Any]:
    ensure_dirs()
    BROWSER_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    cookie_path = COOKIES_DIR / f"{platform}.txt"

    browser_exe = find_browser_executable()
    if not browser_exe:
        raise ValueError("未在本机检测到 Chrome 或 Edge 浏览器，无法自动提取 Cookie。")

    port = _get_free_port()
    clean_target = None
    if target_url:
        from server.vclone.downloader import detect_platform, extract_url_from_text
        try:
            extracted = extract_url_from_text(target_url)
            if detect_platform(extracted) == platform:
                clean_target = extracted
        except Exception:
            clean_target = None
    url = (clean_target or PLATFORM_HOME_URLS.get(platform) or "https://www.douyin.com/").strip()

    args = [
        browser_exe,
        f"--remote-debugging-port={port}",
        "--remote-allow-origins=*",
        f"--user-data-dir={BROWSER_PROFILE_DIR}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-popup-blocking",
        "--new-window",
        "--disable-blink-features=AutomationControlled",
        f"--user-agent={CHROME_UA}",
    ]
    if proxy_url:
        args.append(f"--proxy-server={proxy_url}")
    if headless:
        args.extend(["--headless=new", "--mute-audio", "--window-size=1280,800", "about:blank"])
    else:
        args.extend(["--window-size=1100,780", url])

    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cdp: CDPSession | None = None
    best_netscape = ""
    best_names: list[str] = []
    has_auth = False

    try:
        ws_url = _wait_for_page_ws(port, proc, timeout=10.0)
        if not ws_url:
            raise ValueError("启动浏览器调试端口超时（若已有独立抓取窗口未关闭，请先关闭后重试）。")

        cdp = CDPSession(ws_url, timeout=10.0)
        cdp.call("Page.enable")
        cdp.call(
            "Emulation.setUserAgentOverride",
            {
                "userAgent": CHROME_UA,
                "platform": "Win32",
                "userAgentMetadata": CHROME_UA_METADATA,
            },
        )
        cdp.call(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"},
        )
        if headless:
            cdp.call("Page.navigate", {"url": url})
            time.sleep(4.5)

        auth_keys = PLATFORM_AUTH_COOKIES.get(platform, ())
        guest_keys = PLATFORM_GUEST_COOKIES.get(platform, ())
        max_wait = 7.0 if headless else float(max(15, wait_login_sec))
        t_loop = time.time()

        while time.time() - t_loop < max_wait:
            if proc.poll() is not None:
                break
            try:
                res = cdp.call("Network.getAllCookies")
                cookies = res.get("cookies") or []
                netscape_txt, names = _filter_and_format_netscape(cookies, platform)
                if len(names) >= len(best_names):
                    best_netscape = netscape_txt
                    best_names = names
                has_auth = any(k in names for k in auth_keys)
                has_guest = any(k in names for k in guest_keys)
                if headless and (has_auth or (has_guest and len(names) >= 3)):
                    break
                if not headless and has_auth:
                    time.sleep(1.5)
                    res2 = cdp.call("Network.getAllCookies")
                    best_netscape, best_names = _filter_and_format_netscape(res2.get("cookies") or [], platform)
                    break
            except Exception:
                if proc.poll() is not None:
                    break
            time.sleep(1.0)
    finally:
        if cdp:
            cdp.close()
        _kill_proc_tree(proc)

    if not best_names:
        raise ValueError(f"未能从浏览器提取到 {platform} 的有效 Cookie。")

    cookie_path.write_text(best_netscape, encoding="utf-8")
    _restrict_windows_acl(cookie_path)
    return {
        "ok": True,
        "platform": platform,
        "method": "chrome_cdp_headless" if headless else "chrome_cdp_interactive",
        "cookie_count": len(best_names),
        "has_auth": has_auth,
        "cookie_names": best_names[:20],
        "netscape_preview": best_netscape,
    }


async def auto_harvest_cookies(
    platform: str,
    *,
    target_url: str | None = None,
    headless: bool = True,
    wait_login_sec: int = 90,
    proxy_url: str | None = None,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _sync_harvest_cookies,
        platform,
        target_url=target_url,
        headless=headless,
        wait_login_sec=wait_login_sec,
        proxy_url=proxy_url,
    )
