# -*- coding: utf-8 -*-
"""Legacy阻断入口：旧周报布局重写器已经永久停用。"""

from __future__ import annotations

import json


def main() -> int:
    print(json.dumps({"status": "LEGACY_BLOCKED", "reason": "网页排版永久锁定，旧布局脚本不得写入正式产物。"}, ensure_ascii=False, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
