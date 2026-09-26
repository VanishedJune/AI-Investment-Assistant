# AI Investment Assistant

ETF 与基金研究、冻结冠军信号监控及月报生成的 Python 项目。本仓库是从本地工作项目整理的**脱敏源码版**：不连接券商、不自动交易，也不包含任何用户的真实持仓或成交记录。

## 发布范围

- 当前研究脚本、模型配置、8 个 ETF 槽位及 021528 基金所需的冻结模型状态；
- 可公开的 ETF 行情、基金净值、宏观辅助数据和客观指标快照；
- 与个人账户无关的应用源码、更新脚本和模型回归测试。
- 冻结冠军读取所必需的公司行动登记和轻量迭代记录。

本仓库**不包含** `portfolio_state.json`、实际持仓/决策历史、个人月报、备份、历史迭代事件与大量回放中间产物。它们不能从公开源码还原。若要在另一台电脑生成个人月报，须自行从可信的私人备份恢复账户状态和历史记录；不要把这些文件提交到公开仓库。

模型状态位于 `model_iteration/etf_<代码>/weekly_rolling/state.json`。更新行情或运行日常监控不会自动训练或晋升冠军。历史回放/再训练必须单独明确启动，且本脱敏仓库不附带完整历史台账，因此不能据此宣称可复现全部旧迭代结果。

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
