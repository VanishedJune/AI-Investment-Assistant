# -*- coding: utf-8 -*-
"""已停用的独立行情副本同步入口。

自2026-08-30起，8只正式ETF的模型与辅助分析统一读取项目根目录 ``数据/``。
保留本文件只为让旧命令明确失败，避免再次创建 ``model_iteration/etf_*/data``
并造成行情截止日分叉。
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "LEGACY_BLOCKED: ETF模型已统一读取根目录 数据/；不再维护512690或518600的独立数据副本。",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
