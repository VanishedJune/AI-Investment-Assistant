# AI Investment Assistant

ETF 与基金研究、冻结冠军信号监控及月报生成的 Python 项目。

## 发布范围

- 当前研究脚本、模型配置、8 个 ETF 槽位及 021528 基金所需的冻结模型状态；
- 可公开的 ETF 行情、基金净值、宏观辅助数据和客观指标快照；
- 与个人账户无关的应用源码、更新脚本和模型回归测试。
- 冻结冠军读取所必需的公司行动登记和轻量迭代记录。



## 本地准备

建议 Windows、Python 3.11 或更新版本。在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-update.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

行情和基金净值文件随源码提供，截止日期以 `data_manifest.json` 与具体文件为准。冻结冠军监控入口：

```powershell
python -B model_iteration/scripts/monitoring_baseline_report.py --help
```

正式月报、真实持仓核对及成交登记需要私人的 `portfolio_state.json` 和相应历史记录；缺失时应明确失败，而不是把演示数据当成真实账户。完整本地治理规则和历史审查材料仍保留在私人工作区，公开版摘要见 [PUBLIC_STRATEGY.md](PUBLIC_STRATEGY.md)。

## 注意

研究结果不构成投资建议。使用前请自行核实行情截止日、数据来源、模型适用性与账户状态。
