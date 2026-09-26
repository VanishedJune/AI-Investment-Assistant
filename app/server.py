# -*- coding: utf-8 -*-
"""Optional static-file server for the frozen desktop report.

This module never imports market data, an analysis engine or a report builder.
It exposes no API and cannot calculate or modify report content.

Serving boundary: only HTML files directly under ``scripts/`` and assets under
``app/assets/`` are reachable.  The project root, ``数据/``, ``portfolio_state.json``
and source directories are explicitly denied and directory listings are disabled.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlsplit

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
REPORT_DIR = PROJECT_ROOT / "scripts"
ASSET_DIR = APP_DIR / "assets"


class StaticOnlyHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def log_message(self, format: str, *args) -> None:
        print("[%s] %s" % (self.log_date_time_string(), format % args))

    @staticmethod
    def _is_allowed_file(path: str) -> bool:
        try:
            candidate = Path(path).resolve()
        except (OSError, TypeError, ValueError):
            return False
        root = PROJECT_ROOT.resolve()
        report_root = REPORT_DIR.resolve()
        asset_root = ASSET_DIR.resolve()
        if not candidate.is_file():
            return False
        if candidate == root or root not in candidate.parents:
            return False
        if candidate.parent == report_root and candidate.suffix.lower() == ".html":
            return True
        return candidate.is_relative_to(asset_root)

    def send_head(self):
        if (PROJECT_ROOT / '.publication-journal.json').exists():
            self.send_error(503, 'Publication recovery required')
            return None
        path = self.translate_path(self.path)
        if not self._is_allowed_file(path):
            self.send_error(403, "Forbidden: only frozen report HTML and assets are served")
            return None
        return super().send_head()

    def do_GET(self) -> None:
        request_path = urlsplit(self.path).path
        if request_path in ("/", ""):
            try:
                report = latest_frozen_report()
            except (OSError, ValueError, KeyError) as exc:
                self.send_error(503, 'Published report unavailable: ' + str(exc))
                return
            relative = report.relative_to(PROJECT_ROOT).as_posix()
            self.send_response(302)
            self.send_header("Location", "/" + quote(relative, safe="/"))
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            return
        super().do_GET()


def find_free_port(preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return sock.getsockname()[1] if port == 0 else port
            except OSError:
                continue
    raise OSError("无法分配本地静态文件端口")


def latest_frozen_report() -> Path:
    if (PROJECT_ROOT / '.publication-journal.json').exists():
        raise ValueError('Publication recovery required')
    latest = PROJECT_ROOT / 'app/reports/monthly/latest.json'
    payload = latest.read_bytes()
    report = json.loads(payload)
    from datetime import date
    day = date.fromisoformat(report['generated_date']).isoformat()
    dated = latest.with_name(day + '.json')
    if dated.read_bytes() != payload:
        raise ValueError('Latest and dated report disagree')
    expected = REPORT_DIR / ('投资决策月报' + day + '.html')
    candidate = (PROJECT_ROOT / report['html_path']).resolve()
    if candidate != expected.resolve() or not StaticOnlyHandler._is_allowed_file(str(candidate)):
        raise ValueError('Published report HTML missing or path invalid')
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Investment Assistant 静态报告托管")
    parser.add_argument("--port", type=int, default=8973, help="本地端口（默认8973）")
    parser.add_argument("--no-open", action="store_true", help="启动后不自动打开浏览器")
    args = parser.parse_args()
    report = latest_frozen_report()
    port = find_free_port(args.port)
    server = ThreadingHTTPServer(("127.0.0.1", port), StaticOnlyHandler)
    relative_report = report.relative_to(PROJECT_ROOT).as_posix()
    url = f"http://127.0.0.1:{port}/{quote(relative_report, safe='/')}"
    print(f"静态报告：{report.name}")
    print(f"静态文件服务已启动：{url}")
    print("仅托管 scripts/*.html 与 app/assets/*，不提供 API，不执行任何分析计算。按 Ctrl+C 停止。")
    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
