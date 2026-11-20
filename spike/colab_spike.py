"""Phase 0 Colab Connectivity Spike (Items 1-6).

Tests:
1. google-colab-cli keyring credential load & token info (email check)
2. MCP endpoint (https://colaboratory.googleapis.com/mcp) vs colab_cli client
3. Runtime creation (G4 high-mem; prints specs/capacity; checks df -h /content)
4. execute_code (nvidia-smi + nohup background polling)
5. connectionInfo / ContentsClient 1MB upload + download MD5 verification
6. cloudflared tunnel reachability vs local runner fallback
7. CCU balance (ccu-info)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

import httpx
from colab_cli.auth import AuthProvider, _get_token_from_keyring, get_credentials
from colab_cli.client import (
    Accelerator,
    Client,
    ColabRequestError,
    Prod,
    Shape,
    Variant,
    resolve_assign_shape,
)
from colab_cli.contents import ContentsClient
from colab_cli.runtime import ColabRuntime
from colab_cli.state import SessionState

EXPECTED_EMAIL = "demon820308@gmail.com"
MCP_URL = "https://colaboratory.googleapis.com/mcp"


def check_step_1_auth() -> tuple[Client, str]:
    print("\n=== [Spike 1] Checking google-colab-cli credentials ===")
    token_json = _get_token_from_keyring()
    if not token_json:
        raise RuntimeError(
            "No token found in Windows Keyring for 'google-colab-cli'. "
            "Please run `.\\.venv\\Scripts\\colab.exe usage` once in PowerShell to sign in."
        )
    session = get_credentials(provider=AuthProvider.OAUTH2)
    token = session.credentials.token
    resp = httpx.get(
        "https://oauth2.googleapis.com/tokeninfo",
        params={"access_token": token},
        timeout=15,
    )
    resp.raise_for_status()
    info = resp.json()
    email = info.get("email", "")
    print(f"Authenticated email: {email}")
    if email.lower() != EXPECTED_EMAIL:
        raise RuntimeError(f"Unexpected account {email!r}; expected {EXPECTED_EMAIL!r}")
    client = Client(Prod(), session)
    return client, token


def check_step_6_ccu(client: Client) -> None:
    print("\n=== [Spike 6] Checking CCU balance (ccu-info) ===")
    ccu = client.get_consumption_user_info()
    print(
        f"CCU Balance: {ccu.paid_compute_units_balance:.2f} | "
        f"Rate/hr: {ccu.consumption_rate_hourly:.2f} | "
        f"Active assignments: {ccu.assignments_count}"
    )
    raw = client.session.get(
        "https://colab.research.google.com/tun/m/ccu-info",
        params={"authuser": "0"},
        headers={"Accept": "application/json", "X-Colab-Client-Agent": "colab-cli"},
        timeout=15,
    )
    print(f"Raw ccu-info ({raw.status_code}): {raw.text.strip()}")


def probe_mcp_endpoint(token: str) -> None:
    print("\n=== [Spike 2a] Probing MCP endpoint ===")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "colab-h3-spike", "version": "0.1.0"},
        },
    }
    try:
        r = httpx.post(MCP_URL, headers=headers, json=payload, timeout=20)
        print(f"MCP initialize HTTP {r.status_code}: {r.text[:400]}")
        if r.status_code == 200:
            session_id = r.headers.get("mcp-session-id")
            list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
            h2 = dict(headers)
            if session_id:
                h2["mcp-session-id"] = session_id
            r2 = httpx.post(MCP_URL, headers=h2, json=list_req, timeout=20)
            print(f"MCP tools/list HTTP {r2.status_code}: {r2.text[:600]}")
    except Exception as exc:
        print(f"MCP probe failed: {exc}")


def check_runtime_flow(
    client: Client,
    accelerator_name: str = "G4",
    high_mem: bool = True,
    keep: bool = False,
) -> None:
    print(f"\n=== [Spike 2b] Assigning {accelerator_name} (high_mem={high_mem}) runtime ===")
    existing = client.list_assignments()
    print(f"Existing assignments: {len(existing)}")

    accel = Accelerator(accelerator_name)
    shape = resolve_assign_shape(accel, high_mem=high_mem)
    nbh = uuid.uuid4()
    assigned = client.assign(nbh, Variant.GPU, accel, shape)
    endpoint = assigned.endpoint
    proxy = assigned.runtime_proxy_info
    print(f"Assigned endpoint={endpoint} | proxy_url={proxy.url} | expires_in={proxy.token_expires_in_seconds}s")

    state = SessionState(
        name="spike",
        token=proxy.token,
        url=proxy.url,
        endpoint=endpoint,
        token_expires_at=proxy.expires_at(),
        variant="GPU",
        accelerator=accelerator_name,
        machine_shape="HIGH_RAM" if high_mem else "STANDARD",
    )
    rt = ColabRuntime(url=proxy.url, token=proxy.token, session_name="spike")

    try:
        print("\n=== [Spike 3] Executing nvidia-smi & disk check ===")
        outs = rt.execute_code(
            "import subprocess; print(subprocess.check_output(['nvidia-smi'], text=True)); "
            "print(subprocess.check_output(['df', '-h', '/content'], text=True))"
        )
        print(json.dumps(outs, indent=2, ensure_ascii=False)[:1500])

        print("\n=== [Spike 3b] Testing nohup background job + tail ===")
        rt.execute_code(
            "import subprocess, os; os.makedirs('/content/logs', exist_ok=True); "
            "subprocess.Popen('nohup bash -c \"for i in 1 2 3; do echo tick_$i; sleep 1; done; touch /content/logs/spike.done\" > /content/logs/spike.log 2>&1 &', shell=True)"
        )
        time.sleep(4)
        outs2 = rt.execute_code(
            "from pathlib import Path; "
            "print('DONE_EXISTS=', Path('/content/logs/spike.done').exists()); "
            "print(Path('/content/logs/spike.log').read_text())"
        )
        print(json.dumps(outs2, indent=2, ensure_ascii=False)[:800])

        print("\n=== [Spike 4] Testing ContentsClient 1MB upload/download MD5 ===")
        contents = ContentsClient(state)
        payload = os.urandom(1024 * 1024)
        expected_md5 = hashlib.md5(payload).hexdigest()
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "spike_1mb.bin"
            dst = Path(td) / "spike_1mb_down.bin"
            src.write_bytes(payload)
            contents.upload(str(src), "content/spike_1mb.bin")
            contents.download("content/spike_1mb.bin", str(dst))
            actual_md5 = hashlib.md5(dst.read_bytes()).hexdigest()
            print(f"1MB MD5 expected={expected_md5} actual={actual_md5} match={expected_md5 == actual_md5}")
            contents.rm("content/spike_1mb.bin")

        print("\n=== [Spike 5] Testing cloudflared quick tunnel vs local HTTP ===")
        tunnel_outs = rt.execute_code(
            "import subprocess, time, re, urllib.request, os\n"
            "srv = subprocess.Popen(['python3', '-m', 'http.server', '8188', '--bind', '127.0.0.1'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "time.sleep(0.5)\n"
            "print('local http:', urllib.request.urlopen('http://127.0.0.1:8188').status)\n"
            "cf = '/usr/local/bin/cloudflared'\n"
            "if not os.path.exists(cf):\n"
            "    subprocess.run('curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /tmp/cloudflared && chmod +x /tmp/cloudflared', shell=True, check=True)\n"
            "    cf = '/tmp/cloudflared'\n"
            "p = subprocess.Popen([cf, 'tunnel', '--url', 'http://127.0.0.1:8188', '--no-autoupdate'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n"
            "url = None\n"
            "deadline = time.time() + 25\n"
            "while time.time() < deadline:\n"
            "    line = p.stdout.readline()\n"
            "    if not line: break\n"
            "    m = re.search(r'https://[a-z0-9-]+\\.trycloudflare\\.com', line)\n"
            "    if m:\n"
            "        url = m.group(0)\n"
            "        break\n"
            "print('TUNNEL_URL=' + str(url))\n"
            "time.sleep(2)\n"
            "if url:\n"
            "    try:\n"
            "        print('tunnel_from_colab=', urllib.request.urlopen(url, timeout=10).status)\n"
            "    except Exception as exc:\n"
            "        print('tunnel_from_colab_err=', exc)\n"
            "p.terminate(); srv.terminate()\n"
        )
        print(json.dumps(tunnel_outs, indent=2, ensure_ascii=False)[:1200])
    finally:
        if not keep:
            print(f"\n=== Cleaning up runtime {endpoint} ===")
            client.unassign(endpoint)
            print("Unassigned successfully.")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    accel = args[0] if args else "G4"
    keep = "--keep" in sys.argv
    high_mem = "--no-high-mem" not in sys.argv
    client, token = check_step_1_auth()
    check_step_6_ccu(client)
    probe_mcp_endpoint(token)
    try:
        check_runtime_flow(client, accelerator_name=accel, high_mem=high_mem, keep=keep)
    except ColabRequestError as exc:
        print(f"ColabRequestError: {exc} | body={exc.response_body}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
