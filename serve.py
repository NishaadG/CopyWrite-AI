"""Start CopyWrite AI and give it a public link that works.

    Colab / Kaggle notebook:   %run serve.py            (or: !python serve.py)
    Any machine:               python serve.py [--backend font|emuru] [--port 7860]
    See what's going on:       python serve.py --status
    Stop it:                   python serve.py --stop

What it does, in order:
  1. stops a previous run (so re-running is safe)
  2. starts app.py in the background, logging to outputs/app.log
  3. waits until the app really answers on localhost - if it crashes, prints the error
  4. opens a free Cloudflare quick tunnel (https://....trycloudflare.com: no account, no
     sign-up) and waits until that link really works before printing it
  5. on Colab, also prints a Colab-proxied link as a backup

Why not Gradio's own share link (gradio.live)? Its relay often answers "504 Gateway Time-out"
even when the app is fine. Use --gradio-share to try it as well.
"""

import argparse
import os
import platform
import re
import signal
import stat
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
APP_LOG = os.path.join(OUT, "app.log")
TUNNEL_LOG = os.path.join(OUT, "tunnel.log")
PIDS = os.path.join(OUT, "serve.pids")
CF_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/"


def _say(msg=""):
    print(msg, flush=True)


def _http_ok(url: str, timeout: float = 5) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "copywrite-serve"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status < 400
    except Exception:
        return False


def _tail(path: str, n: int = 60) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except FileNotFoundError:
        return "(no log yet)"


def _stop_previous():
    if not os.path.exists(PIDS):
        return
    for pid in open(PIDS).read().split():
        try:
            os.kill(int(pid), signal.SIGTERM)
        except (OSError, ValueError):
            pass
    os.remove(PIDS)
    time.sleep(2)


def _spawn(cmd, log_path):
    kw = {"start_new_session": True} if os.name == "posix" else {}  # keep running after this script
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, **kw)
    with open(PIDS, "a") as f:
        f.write(f"{proc.pid}\n")
    return proc


def _cloudflared() -> str:
    """Path to the cloudflared binary, downloaded once into outputs/bin."""
    system, machine = platform.system().lower(), platform.machine().lower()
    arch = "arm64" if machine in ("aarch64", "arm64") else "amd64"
    if system == "linux":
        name = f"cloudflared-linux-{arch}"
    elif system == "windows":
        name = "cloudflared-windows-amd64.exe"
    else:
        raise RuntimeError("Install cloudflared yourself on this OS (e.g. brew install cloudflared).")
    path = os.path.join(OUT, "bin", name)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _say("Downloading cloudflared (one time, ~40 MB)...")
        urllib.request.urlretrieve(CF_URL + name, path + ".part")
        os.replace(path + ".part", path)
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def start_app(port: int, backend, gradio_share: bool, wait_s: int = 300):
    cmd = [sys.executable, "-u", "app.py", "--port", str(port)]
    if backend:
        cmd += ["--backend", backend]
    if gradio_share:
        cmd += ["--share"]
    proc = _spawn(cmd, APP_LOG)
    local = f"http://127.0.0.1:{port}/"
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if proc.poll() is not None:
            return proc, False
        if _http_ok(local):
            return proc, True
        time.sleep(2)
    return proc, False


def start_tunnel(port: int, wait_s: int = 90):
    proc = _spawn([_cloudflared(), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"], TUNNEL_LOG)
    t0, url = time.time(), None
    while time.time() - t0 < wait_s and proc.poll() is None:
        if url is None:
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", _tail(TUNNEL_LOG, 200))
            url = m.group(0) if m else None
        elif _http_ok(url, timeout=10):  # the name takes a few seconds to go live
            return url
        time.sleep(2)
    return None


def status():
    _say("=== app log (outputs/app.log) ===")
    _say(_tail(APP_LOG))
    _say("=== tunnel log (last lines) ===")
    _say(_tail(TUNNEL_LOG, 15))
    try:
        _say(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv"],
                            capture_output=True, text=True).stdout)
    except FileNotFoundError:
        _say("No GPU (nvidia-smi not found).")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--backend", choices=["emuru", "font"], default=None,
                    help="default: emuru if a GPU is available, otherwise font")
    ap.add_argument("--no-tunnel", action="store_true", help="local only (no public link)")
    ap.add_argument("--gradio-share", action="store_true", help="also try Gradio's gradio.live link")
    ap.add_argument("--status", action="store_true", help="show the logs of the running app and exit")
    ap.add_argument("--stop", action="store_true", help="stop the app and the tunnel")
    args = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    if args.status:
        status()
        return
    if args.stop:
        _stop_previous()
        _say("Stopped.")
        return

    _stop_previous()
    _say("Starting the app (about 30-60 s; the AI model keeps loading in the background after that)...")
    proc, up = start_app(args.port, args.backend, args.gradio_share)
    if not up:
        _say("\nTHE APP DID NOT START. This is the error - copy everything below if you need help:\n")
        _say(_tail(APP_LOG, 80))
        return
    _say(f"App is running locally on port {args.port}.")

    links = []
    if not args.no_tunnel:
        _say("Opening a public link (Cloudflare)...")
        try:
            url = start_tunnel(args.port)
        except Exception as e:
            url = None
            _say(f"Cloudflare tunnel failed: {e}")
        if url:
            links.append(("Public link (any device)", url))
        else:
            _say("Couldn't open the Cloudflare link. Tunnel log:\n" + _tail(TUNNEL_LOG, 15))
    if _in_colab():
        try:
            from google.colab.output import eval_js

            links.append(("Colab link (this browser only)", eval_js(f"google.colab.kernel.proxyPort({args.port})")))
        except Exception:
            pass
    if args.gradio_share:
        m = re.search(r"https://[\w.-]+\.gradio\.live", _tail(APP_LOG, 200))
        if m:
            links.append(("gradio.live link (may 504)", m.group(0)))

    _say("\n" + "=" * 70)
    if links:
        _say("CopyWrite AI is ready. Open:")
        for name, url in links:
            _say(f"  {name:32s} {url}")
    else:
        _say(f"CopyWrite AI is running locally: http://127.0.0.1:{args.port}/")
    _say("=" * 70)
    _say("The app keeps running in the background. To see its log or errors: python serve.py --status")


if __name__ == "__main__":
    main()
