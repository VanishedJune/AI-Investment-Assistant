# AI Investment Assistant

ETF 与基金研究工具，提供行情更新、技术指标计算、冻结模型信号监控和周期报告相关代码。项目不连接券商，也不自动交易。

## 主要功能

- 8 个 ETF 排序槽位与 `021528` 基金的独立数据分析；
- ETF 行情、基金净值和黄金宏观辅助数据整理；
- 日线、周线技术指标与冻结冠军信号监控；
- B0207 周期基线、V70 风险保护及报告相关研究代码。

冻结模型状态位于 `model_iteration/etf_<代码>/weekly_rolling/state.json`。行情更新和日常监控不会自动训练或晋升模型。

## 本地准备

建议 Windows、Python 3.11 或更新版本。在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-update.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

数据截止日期以 `data_manifest.json` 与具体文件为准。查看冻结冠军监控入口：

```powershell
python -B model_iteration/scripts/monitoring_baseline_report.py --help
```

## 注意

研究结果不构成投资建议。使用前请核实行情截止日、数据来源及模型适用性。
