# model_iteration（8只ETF冠军模型迭代线）

> **正式保护口径（2026-09-13确认）**：正式方案为B0207每28天完整调仓＋Agent V70看空保护＋单标的连续持有期高点回撤10%止损＋保护资金回投其他原有持仓。V70新看空区间参考减持70%；持有期高点回撤严格超过10%时清仓100%。保护卖出款按其他原有持仓市值比例回投，没有合格接收标的时才留现金。该调整属于组合保护层，不修改本目录冠军模型、训练或迭代规则。

隔离实验线：Agent 诊断 → Champion/Challenger → PIT/回放/Forward OOS/账户模拟/门禁 → ITERATION 记录。

> 运行边界：本实验线及其Champion只提供冠军信号证据，并只在名义决策日作为B0207周期基线输入；非决策周的最新推理只作监控，不替换周期基线。本实验线不直接产生正式买卖、Agent保护建议或用户目标。正式流程必须由Agent以最近一次名义决策日的B0207周期基线为主要机械参考，以周K为主、日K为辅比较全部8只ETF，并通过`agent-multi-parameter-v3`日周导数、均线、RSI、量能和相对强弱联合门禁；正式配置按B0207每28天完整调仓，非决策周Agent只提出已有持仓的V70看空保护，并并列检查本次连续持有期高点回撤10%止损，保护资金按其他原有持仓市值比例回投，没有合格接收标的时才留现金；实盘由用户确认。本实验线原有周频回放规则保持独立。详见项目根目录《投资策略.md》。

> **旧ETF池入口边界：**仍硬编码`518600`或`512010`的脚本和配置只用于冻结历史结果解释，不得作为当前8只池入口。`scripts/optimize_allocation.py`、`scripts/strategy_backtest_pk.py`和`scripts/verify_b0207_champion.py`已设置`LEGACY_BLOCKED`；当前生产顺序只读取`configs/investment_priority.json`。其他文件中的旧代码引用必须结合文件日期和历史用途判断，不能据此覆盖517520、515220或当前B0207状态。

## 515220 / 517520 全周期迭代口径（2026-09-07）

- 按用户明确要求，这两只ETF复用项目早期且仍由账户、挑战和晋级模块支持的“周频模型目标仓位直接执行”路径，配置以`execution_policy: null`显式选择；不使用B0207四周决策、非决策周只加仓或看空保护执行规则。
- 仅执行口径采用原冠军模型路径；诊断、sanitized_only提案、Forward OOS、同现金流基准门禁和promotion-v2.0.0晋级复核均保持项目既有实现，不新增规则。
- 截至2026-09-04，515220完成295个锚点、517520完成108个锚点；两者均无候选晋级，最终冠军保持CH_S064_1。517520当前冠军的事实源是自身`weekly_rolling/state.json`，518600只保留在初始参数来源说明中，不再参与当前推理。

## ITERATION_001（全历史基线）
- `scripts/run_experiment.py --etf 159915 --iteration ITERATION_001_159915`
- 产物：`outputs/`、`iterations/ITERATION_001_159915.json/.md`
- 只读使用 `数据\` 与 `app\features\latest.json`；不修改网页线任何文件。

## 周滚动子系统（ITERATION_002，promotion-v2.0.0）

`rolling/` 实现严格 PIT 周滚动：712 锚点（2012-09-14 ~ 2026-08-07，末锚 pending）、每周冻结 1/2/4/8W 预测、成熟评价、每 4/8 个成熟 1W 触发 Prediction/Strategy Due。

### B0207式研究账户执行策略（v3.0.0交易节奏）

以下规则仅用于模型迭代、回放和研究账户，不直接控制正式B0207周期基线或真实持仓。研究账户交易不再逐周改变持仓，改为与B0207相似的操作逻辑：
- **每 4 周一次决策锚点**（`decision_cycle_weeks=4`，以 0 号可用锚点为相位起点）：按模型目标权重完整调仓，可买可卖、可新开仓/清仓；
- **非决策周资金只用于加仓**：当前持有份额时最多加至目标权重，不新开仓、不卖出；
- **非决策周卖出唯一例外 = 验证看空保护**：五条件同时成立（`d_dif1<0`、`d_dif2<0`、`w_dif1<0`、`w_dif2<0`、`mom20<-0.05`）时按 `sell_frac=70%` 减持当前份额；单次触发（条件持续期间不重复卖），条件恢复后重置保护期；
- **同现金流门禁口径同步**：每周入金 2000 的规则账户在非决策周只按现有持仓比例加仓，决策周按目标调仓；
- 配置：`configs/etf_<code>.json` 的 `execution_policy` 可覆盖，默认值见 `rolling/config.py` 与 `rolling/execution.py`。

### 核心规则
- **创建与晋级分离**：Due 只做盲化诊断 → Agent 动态设计 → precheck → 冻结 FINAL → 登记 EVALUATING；晋级只使用创建后的真实 Forward OOS，且只与 paired baseline 比较。
- **门禁瘦身（v2.0.0）**：Prediction 核心 = 1W/2W 方向准确率、MAE、`p_up` Brier（EXPANDING 需 1W/2W 至少一项相对改进 ≥5%，RECENT 仅 non-inferiority）；Strategy 核心 = 累计收益、Sharpe、MDD。REGIME、强涨/强跌 Brier、4W/8W、胜率/中位/参与率/换手、FULL_AVAILABLE_PIT 全部只作诊断。
- **CA 仅计量**：成交一律 `raw_open`；账户 = raw price + 显式 CA 事件账本；分析层 = analysis_price；CA 事件不得作为特征或门禁。当前 `weekly_rolling/corporate_actions/159915_ca_events.json` 已 verified（窗口内零事件）。
- **PIT 防火墙**：Agent 输入只含 opaque_due_id / TARGET_A / 归一化统计；禁止日期、代码、绝对价格、anchor_index；Proposal 校验器拒绝越界字段。
- **状态机**：EVALUATING → PROMOTION_CANDIDATE / SHADOW_EVALUATION / REJECTED；同检查点至多一个 PROMOTED；Winner 生效后旧 parent Challenger 转 STALE_SHADOW；晋级只继承 spec，不继承 shadow NAV。

### 运行
```powershell
# 回放（首次从 anchor 0；暂停点用 --resume 继续）
python -m rolling.cli replay --etf 159915 [--resume] [--mode generate|verify]

# 冻结前筛查（schema/PIT/divergence，不写台账）
python -m rolling.cli precheck --file scripts\proposal_drafts\ROUND_XXX.draft.json

# 冻结 FINAL Proposal
python -m rolling.cli proposal --file scripts\proposal_drafts\ROUND_XXX.draft.json

# 读取盲化诊断包
python -m rolling.cli diagnose --round ROUND_XXX

# 每周实时推进
python -m rolling.cli weekly --etf 159915
```

### 台账
- `weekly_rolling/events.jsonl`（只追加）、`state.json`（派生快照，原子覆盖）、`state_history/`、`diagnostics/`、`proposals/`、`corporate_actions/`
- `iterations/ITERATION_002_159915.json/.md`：首个 Prediction Due + 首个 Strategy Due 的完整记录

### 当前进度（159915 全历史迭代已完成）
- 独立工作区：`etf_159915/`（台账/草稿/隔离设计/迭代记录/输出全部按 ETF 隔离）
- **ITERATION_003（角色池扩展版）**：265 份 Proposal、617 个 Challenger、4 次晋级
- 最终 Champion：CH_P132_3（anchor 593 生效）；链见
  `etf_159915/iterations/ITERATION_003_159915.json`
- 台账哈希 `421bd5a3…`（生成/验证三段逐段一致）；60 项测试通过
- 归档：`etf_159915/champions/CH_S064_1.json`、`etf_159915/iterations/legacy_v2.1/`
- 多 ETF：复制模板（`template/README.md` + `configs/etf_TEMPLATE.json`），
  CLI 一律 `--etf <code>` 自动切换工作区

### 159611独立接入与回放（2026-09-19）

- 初始参数来自159941当前冠军`CH_S064_1`，只复制结构、参数和规则；不复制159941的收益、持仓、交易或评估结果。
- 159611自身行情覆盖2022-01-07至2026-09-18，形成201个可用锚点、75份冻结提案；既有门禁下0次晋级，最终冠军保持`CH_S064_1`。
- 选模回放累计收益3.2381%、最大回撤-16.0041%、Sharpe 0.1296；同期买入持有累计收益17.2896%、最大回撤-21.6539%、Sharpe 0.3027。末26个完成执行周的冻结参数时间审计为-11.8810%、最大回撤-16.0041%、Sharpe -1.1267；同期买入持有为-7.0734%、-18.8995%、-0.4187。该末段只作冻结参数审计，保留决定也查看了全回放，因此不宣称为独立选模样本外成绩。
- 当前第5槽位和冠军加载入口指向159611；原159941模型、配置与行情在`archive/instrument_replacements/159941_to_159611/`保留，可恢复。
- 日常行情更新和策略分析只调用冻结冠军推理，不自动训练、晋级或回放；显式迭代入口为`python -m scripts.run_etf_iteration --etf 159611 --refresh-syntax`。

### 测试
```powershell
python -m unittest discover -s tests
```
43+ 项，覆盖 CA/日历/Due/Proposal/divergence/门禁 v2.0.0/回放确定性/网页线哈希/桶成熟约束/隔离机制。

### 知识隔离（sanitized_only）
- `sanitized_workspace/`：只含盲化诊断（opaque_due_id/TARGET_A/week_offset，无日期/代码/绝对价格）与
  固定策略语法 `spec_syntax.json`；
- `scripts/sanitized_factory.py`：只读该工作区生成 Proposal（越界读取被拒绝、禁止字段自检），
  输出 `context_scope=sanitized_only`（可晋级）；
- 本项目历史轮次均标 `full_context` → 登记为 `non_decisional`，永不晋级；
  只有经隔离流程重新生成并验证的 sanitized_only Proposal 才恢复决策性。
