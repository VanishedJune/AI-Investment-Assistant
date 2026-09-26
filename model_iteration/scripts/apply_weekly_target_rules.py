# -*- coding: utf-8 -*-
"""Legacy阻断入口：旧周报目标同步器已经永久停用。"""

from __future__ import annotations

import json


def main() -> int:
    print(
        json.dumps(
            {
                "status": "LEGACY_BLOCKED",
                "reason": "旧周报目标同步器已停用，不得修改正式产物。",
                "replacement": "scripts.prepare_monthly_decision + scripts.publish_monthly_report",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
