# 通用 ETF 周滚动迭代模板

本模板可在**其他对话/工作区**复制使用：把 `model_iteration/` 拷到目标 ETF 的独立工作区，
按下面 6 步即可开启该 ETF 的“周滚动预测—成熟评价—挑战竞争—AI 复核晋级”迭代。
代码已 ETF 参数化：除数据文件与 CA 表外，硬编码只保留在 `configs/etf_<code>.json` 中。

## 目录

```
model_iteration/
  rolling/                 # 引擎（通用，不绑定 ETF）
    config.py              # 读取 configs/etf_<code>.json
    data.py                # 对齐装配 + fail-closed 断言（期望锚点数来自配置）
    calendar.py            # NAMES: {code: 名称}（新 ETF 在此登记）
    forecast/evaluate/ledger/replay/promotion/challenge/review/statindex
    learning.py            # 学轨：Ridge 滚动学习模板（超参冻结、embargo、训练内标准化）
    benchmark.py           # 基准轨：买入持有 + 智能定投（NON_DECISIONAL）
  configs/etf_TEMPLATE.json
  sanitized_workspace/     # 隔离设计工作区（盲化诊断 + 固定语法）
  scripts/                 # diagnose-all / proposal_factory / sanitized_factory /
                           # fix_proposals / run_experiment / smart_dca
  template/README.md       # 本文件
  weekly_rolling/          # 台账（events/state/diagnostics/proposals/corporate_actions）
```

## 新 ETF 起步（6 步）

1. **数据文件**：在 `数据\` 放两份 CSV：
   `{code}_{名称}_日线.csv` 与 `{code}_{名称}_周线.csv`，
   列需同时含 `date / raw_open / raw_high / raw_low / raw_close` 与
   `adj_open / adj_high / adj_low / adj_close`（raw 用于成交层，adj 只用于指标计算）。
2. **登记名称**：在 `rolling/calendar.py` 的 `NAMES` 增加 `"{code}": "名称"`。
3. **显式 CA 表**：`weekly_rolling/corporate_actions/{code}_ca_events.json`
   （字段 type/ex_date/entitlement_date/pay_date/ratio/dps/source/verified；
   缺失或未 verified 一律 fail-closed；禁止从跳空/adj_factor 反推）。
4. **配置**：复制 `configs/etf_TEMPLATE.json` 为 `configs/etf_{code}.json`，
   填写覆盖区间；`expected_anchors` 先用 `null`，跑一次
   `python -m rolling.cli diagnose-all --etf <code>` 看实际可用锚点数后回填。
5. **生成 Proposal**：
   - `python -m rolling.cli diagnose-all --etf <code>`（265 份 Due 的盲化诊断，
     数量按该 ETF 成熟窗口动态推导）；
   - 把诊断拷入 `sanitized_workspace/diagnostics/`，运行
     `python -m scripts.sanitized_factory`（只读隔离工作区，输出
     `context_scope=sanitized_only` 的 draft）；
   - 运行 `python -m scripts.fix_proposals`（顺序 divergence 修正）；
   - 批量冻结（`freeze_final_proposal(..., etf=<code>)`）。
6. **回放与验收**：
   - 重置台账后 `python -m rolling.cli replay --etf <code>`；
   - 有候选时暂停走 `rolling approve`（或 `auto_review` 驱动）；
   - 验收：全量测试 + REPLAY/VERIFY 两次台账哈希一致 + 网页线零改动。

## 双轨同场

| 轨 | 冻结什么 | 参赛身份 | 门禁 |
|---|---|---|---|
| 验轨（规则） | 全部参数 | 单个 spec | 共用 Forward OOS + paired baseline + AI 复核 |
| 学轨（learning_ridge） | **超参**（alpha/embargo/min_train/特征白名单/仓位映射） | 学习算法整体 | 输出仓位→策略类门禁；输出预测→预测类门禁 |

学轨泄漏口已封死（`rolling/learning.py`）：
- 训练样本只取 `s + embargo <= t`（`embargo >= horizon`）；
- mean/std 只在训练集内计算；
- 超参随 Proposal 冻结，**禁止 OOS 调参**；改进=新一代 Challenger 重新注册；
- 预测 t 冻结、t+h 成熟评价。

学轨**已接入引擎**（不是仅模板）：
- `forecast.build_forecast` / `challenge.position_path` 识别 `spec["kind"] == "learning_ridge"`
  后调用 `learning.walkforward_position(features, fwd_arrays, i, spec)` 返回目标仓位；
- `spec.validate_spec` 支持 learning_ridge（走 `learning.validate_learning_spec`，embargo>=horizon 等）；
- `create_screening` / `promotion.checkpoint` / `fix_proposals` 已贯通 fwd_arrays；
- 固定语法中新增 `learning_ridge` 策略角色（超参随 Proposal 冻结，生成器对其不做 jitter），
  sanitized_factory 生成该角色即产出可参赛的学习 Challenger；
- 学轨输出仓位 → 按策略类门禁（累计/Sharpe/MDD + 基准硬对照 + AI 复核）。

## 基准轨（常驻，NON_DECISIONAL）

`rolling/benchmark.py`：
- 买入持有：永远满仓（含 10bp 成本）；
- 智能定投：2000/周 × 扣款率（±2.5% 内 100%，涨降最低 50%，跌升最高 200%，T-1 参考净值）。

每次 ITERATION 输出“规则 vs 基准”对比；若没有模型能在风险调整后跑赢买入持有，
系统应诚实给出“建议直接持有”。

## 决策性标记

- `context_scope=sanitized_only`：隔离流程生成，可晋级；
- `context_scope=full_context`：历史上下文生成，登记 `non_decisional`，永不晋级；
- `auto_approve=false`：机器硬门禁只产出候选，`ai_review`（引用 `promotion_checkpoint` 哈希）后才晋级。

## 研究账户执行策略（B0207式，v3.0.0）

本节仅定义模型迭代、回放和研究账户的模拟执行，不生成或覆盖正式B0207周期基线，不改变真实持仓。正式项目中，B0207周期基线只在名义决策日生成；非决策周Agent只能以该周期基线为主要参考提出建议。

- 每 `decision_cycle_weeks`（默认 4）周一个决策锚点，按目标权重完整调仓；
- 非决策周资金只用于加仓（当前持有份额时最多加至目标权重），不新开仓、不卖出；
- 非决策周卖出唯一例外 = 验证看空保护：五条件（`d_dif1<0`、`d_dif2<0`、`w_dif1<0`、`w_dif2<0`、`mom20<-0.05`）
  同时成立时按 `sell_frac`（默认 70%）减持当前份额，单次触发、恢复重置；
- 同现金流规则账户：每周入金在非决策周只按现有持仓比例加仓，决策周按目标调仓；
- 配置在 `configs/etf_<code>.json` 的 `execution_policy`，默认见 `rolling/config.py`。

## 锦标赛优化（v2 策略）

- **长周期稳健性进材料**：每个 Challenger 登记时计算全历史 `long_horizon`（累计/Sharpe/MDD/胜率/平均仓位），
  晋级检查点连同基准对照一起写入 `promotion_checkpoint` 与 `ai_review` 材料；
- **同现金流基准硬对照（晋级硬门槛）**：挑战者从创建锚点到当前 Due（截止点），每周 2000 元
  同现金流累积，规则自主决定交易时间与份额；TWR 必须**显著跑赢 max(现任 Champion, 智能定投, 固定定投)**
  （优势 ≥ max(基准 TWR×5%, +2pp)），且 **IRR ≥ 基准最高、MDD ≥ 基准最差−5pp**（均为硬条件），
  否则 `CASHFLOW_GATE` 不晋级；短期窗口只作分析；
  `promotion_policy.benchmark_hard_gate=true` 默认开启；
- **最低 OOS 26 周**（`min_oos_weeks`）；**52 周内最多 2 次晋级**（`max_promotions_per_52w`）；
  **全历史 MDD ≥ −45%** 硬约束（`long_horizon_mdd_hard`，防只经历牛市窗口的规则蒙混）；
  **rebase 最多 2 代**（`max_rebase_generations`），克隆必须重新积累 OOS；
- **预测类死锁已修**：规则型 spec 的 `p_up/expected_return/direction` 由 `score→概率` 映射自生成，
  不再与 Champion 共享桶统计，MAE/Brier/准确率可被 spec 实质改善，预测类门禁可真正通过。
- **打破 STALE 锁死（rebase）**：Champion 更替后，`STALE_SHADOW` 中全历史 Sharpe ≥
  `promotion_policy.rebase_min_sharpe`（默认 0.7）的候选自动以新 Champion 为 parent 重新登记
  （新 id=`旧id@R<anchor>`），重新积累 Forward OOS；事件 `challenger_rebased` 留痕。

## 验收清单

- 数据装配：行数/末锚 pending/逐行 close 对齐/无 NaN，任一失败拒绝启动；
- 诊断：failure_cases 非空（matured>=4 为空 → DIAGNOSTIC_INCOMPLETE）；
- 门禁：Prediction=1W/2W 准确率+MAE+p_up Brier；Strategy=累计/Sharpe/MDD；REGIME 等只诊断；
- 状态机：EVALUATING→PROMOTION_CANDIDATE→(APPROVE)→PROMOTED；STALE/SHADOW/REJECTED；
- 复现：回放两次 events/state 哈希一致；网页线哈希不变。
