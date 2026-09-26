# -*- coding: utf-8 -*-
"""四周决策月报的客观证据与B0207机械基线快照工具。

本模块校验本地数据，调用冻结冠军的最新行情推理入口，并按 B0207 生成
可审计的机械基线。机械基线不是最终
建议或用户目标；本模块不生成自然语言买卖结论，也不修改真实持仓或HTML。
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable, Mapping
from zoneinfo import ZoneInfo

from rolling.inference import InferenceError, build_reference as build_frozen_reference
from scripts.agent_v70_protection import (
    POLICY as PROTECTION_POLICY,
    empty_holding_peak_stop_review,
    empty_review,
    resolve_versions,
    ProtectionError,
)
from scripts.agent_multi_parameter_gate import (
    GATE_VERSION, AgentGateError, build_quantitative_evidence, empty_evidence_review,
    empty_forward_analysis,
)
from scripts.agent_forward_evidence import ForwardEvidenceError, load_forward_evidence
from scripts.investment_priority import CONFIG

MODEL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODEL_ROOT.parent
MANIFEST_PATH = PROJECT_ROOT / "data_manifest.json"
PORTFOLIO_PATH = PROJECT_ROOT / "portfolio_state.json"
FEATURE_PATH = PROJECT_ROOT / "app" / "features" / "latest.json"
SNAPSHOT_DIR = PROJECT_ROOT / "app" / "policy_snapshots" / "monthly"
CANDIDATE_DIR = PROJECT_ROOT / "app" / "reports" / "monthly" / "candidates"
ITERATION_MODEL_DIR = Path(__file__).resolve().parents[3] / "workspace-materials" / "ETF做题记录"
ITERATION_MODEL_FILES = ("训练规则.md", "训练状态.json", "优化记忆.json")
# 兜底锚点。运行时优先从 portfolio_state.json 的正式周期记录读取，避免发布后
# 代码仍停留在旧日期。
BASE_NOMINAL_DATE = date(2026, 7, 25)
SHANGHAI = ZoneInfo("Asia/Shanghai")
FEATURE_FORBIDDEN_KEYS = {
    "decision",
    "buy",
    "sell",
    "hold",
    "ranking",
    "eligibility",
    "recommended_weight",
    "suggested_weight",
    "target_position",
    "entry_action",
    "model_gate_passed",
}


class MonthlyPolicyError(RuntimeError):
    """Fail-closed policy preparation error."""


STALE_LOCK_SECONDS = 12 * 60 * 60


class ProjectLock:
    """Single-instance process lock with stale-lock recovery.

    A crash can leave an O_EXCL lock file behind.  A lock is considered stale
    after ``stale_seconds`` and is removed automatically; shorter-lived locks
    still fail closed.
    """

    def __init__(self, path: Path, *, stale_seconds: int = STALE_LOCK_SECONDS) -> None:
        self.path = path
        self.stale_seconds = stale_seconds
        self.fd: int | None = None

    def __enter__(self) -> "ProjectLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                age = time.time() - self.path.stat().st_mtime
            except OSError as exc:
                raise MonthlyPolicyError(f"无法读取锁文件: {self.path}") from exc
            if age < 0 or age < self.stale_seconds:
                raise MonthlyPolicyError(f"已有同类操作正在运行或锁文件未过期: {self.path}")
            self.path.unlink(missing_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            payload = json.dumps(
                {"pid": os.getpid(), "created_at": datetime.now(SHANGHAI).isoformat(timespec="seconds")},
                ensure_ascii=False,
            )
            os.write(self.fd, payload.encode("utf-8"))
        except FileExistsError as exc:
            raise MonthlyPolicyError(f"已有同类操作正在运行: {self.path}") from exc
        return self

    def __exit__(self, *_args: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.path.unlink(missing_ok=True)


def iteration_model_reference() -> dict:
    """读取 ../workspace-materials/ETF做题记录 迭代模型三份权威文件的版本、大小与修改时间。

    该参考只登记客观版本与文件元数据，不包含Agent判断；Agent在候选分析中
    填写 read=true 与 how_applied，发布器会与当前文件核对。
    """
    missing = [name for name in ITERATION_MODEL_FILES if not (ITERATION_MODEL_DIR / name).is_file()]
    if missing:
        raise MonthlyPolicyError(f"ETF做题记录迭代模型缺失: {', '.join(missing)}")
    rules_text = (ITERATION_MODEL_DIR / "训练规则.md").read_text(encoding="utf-8")
    state = json.loads((ITERATION_MODEL_DIR / "训练状态.json").read_text(encoding="utf-8"))
    memory = json.loads((ITERATION_MODEL_DIR / "优化记忆.json").read_text(encoding="utf-8"))
    try:
        versions = resolve_versions(rules_text, state, memory)
    except ProtectionError as exc:
        raise MonthlyPolicyError(str(exc)) from exc
    rule_version = versions['rule_version']
    state_version = versions['state_version']
    memory_version = versions['memory_version']
    rules_meta = file_meta(ITERATION_MODEL_DIR / "训练规则.md")
    state_meta = file_meta(ITERATION_MODEL_DIR / "训练状态.json")
    memory_meta = file_meta(ITERATION_MODEL_DIR / "优化记忆.json")
    return {
        "source_project": str(ITERATION_MODEL_DIR.resolve()),
        "rule_file": "训练规则.md",
        "rule_version": rule_version,
        "policy_version": versions['policy_version'],
        "rule_size_bytes": rules_meta["size_bytes"],
        "rule_modified_utc": rules_meta["modified_utc"],
        "state_file": "训练状态.json",
        "state_version": state_version,
        "state_size_bytes": state_meta["size_bytes"],
        "state_modified_utc": state_meta["modified_utc"],
        "memory_file": "优化记忆.json",
        "memory_version": memory_version,
        "memory_size_bytes": memory_meta["size_bytes"],
        "memory_modified_utc": memory_meta["modified_utc"],
        "read": False,
        "how_applied": "",
    }


def pretty_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def canonical_json_bytes(value: object) -> bytes:
    """稳定JSON序列化；仅用于结构化内容比较，不生成文件指纹。"""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def file_meta(path: Path) -> dict:
    """返回文件的非哈希元数据：字节大小与UTC最后修改时间。"""
    stat = path.stat()
    return {
        "size_bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise MonthlyPolicyError(f"非法日期: {value}") from exc


def decision_cycle_anchor() -> date:
    """Return the current 28-day cycle anchor from persisted state.

    Publishing a monthly report advances ``portfolio_state.json``.  Reading the
    anchor from state keeps future runs aligned with the actual cycle even if a
    long period elapses without a code change.
    """

    if PORTFOLIO_PATH.is_file():
        try:
            state = json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
            for key in ("active_policy_target", "active_mechanical_baseline"):
                record = state.get(key) or {}
                raw_date = record.get("decision_date")
                if raw_date:
                    return parse_iso_date(str(raw_date))
        except (OSError, ValueError, json.JSONDecodeError, MonthlyPolicyError):
            pass
    return BASE_NOMINAL_DATE


def is_nominal_decision_date(value: date) -> bool:
    delta = (value - decision_cycle_anchor()).days
    return delta >= 0 and delta % int(CONFIG["decision_cycle_days"]) == 0


def next_nominal_date(value: date) -> date:
    return value + timedelta(days=int(CONFIG["decision_cycle_days"]))


def previous_weekday(value: date) -> date:
    cursor = value - timedelta(days=1)
    while cursor.weekday() >= 5:
        cursor -= timedelta(days=1)
    return cursor


def price_protection_for(raw_close: Decimal | float | str) -> dict:
    """按0.001最小价格单位生成严格的±2%执行保护边界。"""

    try:
        close = raw_close if isinstance(raw_close, Decimal) else Decimal(str(raw_close))
    except InvalidOperation as exc:
        raise MonthlyPolicyError("参考收盘价必须为正有限数") from exc
    if not close.is_finite() or close <= 0:
        raise MonthlyPolicyError("参考收盘价必须为正有限数")
    tick = Decimal("0.001")
    return {
        "reference_raw_close": float(close),
        "lower_98_pct": float((close * Decimal("0.98")).quantize(tick, rounding=ROUND_HALF_UP)),
        "upper_102_pct": float((close * Decimal("1.02")).quantize(tick, rounding=ROUND_HALF_UP)),
        "tick_size": 0.001,
        "rounding": "ROUND_HALF_UP",
        "check_time": "09:45 Asia/Shanghai",
    }


def protected_execution_plan(
    current_weights_pct: Mapping[str, float | int],
    current_cash_pct: float | int,
    target_weights_pct: Mapping[str, float | int],
    target_cash_pct: float | int,
    observed_prices: Mapping[str, float | str | Decimal],
    price_protection: Mapping[str, Mapping[str, float | int]],
) -> dict:
    """生成价格保护后的确定性差额执行参数，不修改用户目标或真实持仓。

    买入未通过±2%保护时，取消该买入中由卖出配对融资的等额部分；政策本身
    要求增加现金的净卖出仍保留，使用既有现金的被取消部分则继续留在现金。
    """

    codes = list(CONFIG["priority"])

    def pct(value: float | int, label: str) -> Decimal:
        try:
            result = Decimal(str(value))
        except InvalidOperation as exc:
            raise MonthlyPolicyError(f"{label}不是有效数字") from exc
        if not result.is_finite() or result < 0 or result > 100:
            raise MonthlyPolicyError(f"{label}百分比越界")
        return result

    current = {code: pct(current_weights_pct.get(code, 0), f"{code}当前仓位") for code in codes}
    target = {code: pct(target_weights_pct.get(code, 0), f"{code}目标仓位") for code in codes}
    current_cash = pct(current_cash_pct, "当前现金")
    target_cash = pct(target_cash_pct, "目标现金")
    if sum(current.values(), current_cash) != Decimal("100"):
        raise MonthlyPolicyError("当前ETF仓位与现金必须合计100%")
    if sum(target.values(), target_cash) != Decimal("100"):
        raise MonthlyPolicyError("目标ETF仓位与现金必须合计100%")

    desired_buys = {code: max(target[code] - current[code], Decimal("0")) for code in codes}
    desired_sells = {code: max(current[code] - target[code], Decimal("0")) for code in codes}
    executable_buys = dict(desired_buys)
    price_status: dict[str, str] = {}
    for code, buy in desired_buys.items():
        if buy == 0:
            price_status[code] = "not_a_buy"
            continue
        band = price_protection.get(code) or {}
        observed = observed_prices.get(code)
        if observed is None:
            executable_buys[code] = Decimal("0")
            price_status[code] = "blocked_missing_price"
            continue
        try:
            price = Decimal(str(observed))
            lower = Decimal(str(band.get("lower_98_pct")))
            upper = Decimal(str(band.get("upper_102_pct")))
        except InvalidOperation as exc:
            raise MonthlyPolicyError(f"{code}价格保护参数非法") from exc
        if not price.is_finite() or not lower.is_finite() or not upper.is_finite() or lower > upper:
            raise MonthlyPolicyError(f"{code}价格保护参数非法")
        if lower <= price <= upper:
            price_status[code] = "passed"
        else:
            executable_buys[code] = Decimal("0")
            price_status[code] = "blocked_outside_band"

    buy_total = sum(desired_buys.values(), Decimal("0"))
    sell_total = sum(desired_sells.values(), Decimal("0"))
    executable_buy_total = sum(executable_buys.values(), Decimal("0"))
    paired_sell_total = min(buy_total, sell_total)
    net_cash_sell_total = sell_total - paired_sell_total
    if buy_total > 0:
        executable_paired_sell = paired_sell_total * executable_buy_total / buy_total
    else:
        executable_paired_sell = Decimal("0")
    executable_sell_total = net_cash_sell_total + executable_paired_sell
    if sell_total > 0:
        executable_sells = {
            code: desired_sells[code] * executable_sell_total / sell_total for code in codes
        }
    else:
        executable_sells = {code: Decimal("0") for code in codes}

    resulting_weights = {
        code: current[code] + executable_buys[code] - executable_sells[code] for code in codes
    }
    resulting_cash = current_cash + executable_sell_total - executable_buy_total
    if abs(sum(resulting_weights.values(), resulting_cash) - Decimal("100")) > Decimal("0.00000001"):
        raise MonthlyPolicyError("价格保护后的执行参数未守恒")

    def public_map(values: Mapping[str, Decimal]) -> dict[str, float]:
        return {code: round(float(values[code]), 10) for code in codes}

    return {
        "schema_version": "execution-protection-v1",
        "desired_buy_pct": public_map(desired_buys),
        "desired_sell_pct": public_map(desired_sells),
        "executable_buy_pct": public_map(executable_buys),
        "executable_sell_pct": public_map(executable_sells),
        "canceled_buy_pct": round(float(buy_total - executable_buy_total), 10),
        "canceled_paired_sell_pct": round(float(paired_sell_total - executable_paired_sell), 10),
        "resulting_weights_pct": public_map(resulting_weights),
        "resulting_cash_pct": round(float(resulting_cash), 10),
        "price_status": price_status,
    }


def resolve_trading_cycle(nominal: date, trading_dates: Iterable[date] | None = None) -> tuple[date, date]:
    """返回(实际执行日, 所需前一完整交易日)。

    有交易日集合时严格按集合处理休市顺延；准备未来周期而集合尚未覆盖时，
    仅提供工作日预估，正式准备仍会因共同截止日不匹配而阻断。
    """

    dates = sorted(set(trading_dates or []))
    future = [item for item in dates if item >= nominal]
    if future:
        execution = future[0]
        earlier = [item for item in dates if item < execution]
        if not earlier:
            raise MonthlyPolicyError("交易日历缺少执行日前一交易日")
        return execution, earlier[-1]
    execution = nominal
    while execution.weekday() >= 5:
        execution += timedelta(days=1)
    return execution, previous_weekday(execution)


def cycle_context(nominal: date, portfolio_state: dict, trading_dates: Iterable[date] | None = None) -> dict:
    if not is_nominal_decision_date(nominal):
        raise MonthlyPolicyError(f"{nominal.isoformat()}不是B0207的28天名义决策日")
    execution, required = resolve_trading_cycle(nominal, trading_dates)
    configured = portfolio_state.get("policy_cycle") or {}
    if configured.get("next_decision_date") == nominal.isoformat():
        configured_required = configured.get("required_data_as_of")
        if configured_required:
            required = parse_iso_date(configured_required)
        configured_execution = configured.get("effective_execution_date")
        if configured_execution:
            execution = parse_iso_date(configured_execution)
    return {
        "nominal_decision_date": nominal.isoformat(),
        "decision_date": nominal.isoformat(),
        "execution_date": execution.isoformat(),
        "required_data_as_of": required.isoformat(),
        "next_nominal_decision_date": next_nominal_date(nominal).isoformat(),
        "execution_time": configured.get("execution_time", "09:45 Asia/Shanghai"),
    }


def _manifest_records(manifest: dict) -> list[dict]:
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise MonthlyPolicyError("data_manifest.json缺少files清单")
    return records


def validate_manifest(required_as_of: str, *, verify_all_files: bool = True) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    run_id = str(manifest.get("run_id") or "")
    if not run_id:
        raise MonthlyPolicyError("data_manifest.json缺少run_id")
    if manifest.get("market_status") != "closed":
        raise MonthlyPolicyError("正式月报只接受已收盘数据")
    if manifest.get("quality_status") not in {"passed", "passed_with_warnings"}:
        raise MonthlyPolicyError(f"核心数据质量未通过: {manifest.get('quality_status')}")
    if manifest.get("as_of") != required_as_of:
        raise MonthlyPolicyError(
            f"共同数据截止日必须恰好为{required_as_of}，当前为{manifest.get('as_of')}；"
            "为避免未来数据泄漏不允许自动切片"
        )
    latest_dates = manifest.get("latest_dates") or {}
    if list(sorted(latest_dates)) != list(sorted(CONFIG["priority"])):
        raise MonthlyPolicyError("data_manifest.json的8只ETF集合与B0207配置不一致")
    if any(value != required_as_of for value in latest_dates.values()):
        raise MonthlyPolicyError("8只ETF没有共同有效截止日")

    if verify_all_files:
        metadata_filenames = {"fund_snapshot.csv"}
        for record in _manifest_records(manifest):
            rel = Path(str(record.get("path") or ""))
            path = (PROJECT_ROOT / rel).resolve()
            if PROJECT_ROOT.resolve() not in path.parents:
                raise MonthlyPolicyError(f"清单路径越界: {rel}")
            if not path.is_file():
                raise MonthlyPolicyError(f"清单文件缺失: {rel}")
            recorded_size = record.get("size_bytes")
            if recorded_size is not None and int(recorded_size) != path.stat().st_size:
                raise MonthlyPolicyError(f"清单文件大小不一致: {rel}")
            recorded_modified = record.get("modified_utc")
            if recorded_modified and file_meta(path)["modified_utc"] != str(recorded_modified):
                raise MonthlyPolicyError(f"清单文件修改时间不一致: {rel}")
            if record.get("run_id") != run_id:
                raise MonthlyPolicyError(f"清单记录run_id不一致: {rel}")
            metadata_mode = path.name in metadata_filenames
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = 0
                seen_dates: list[date] = []
                for row in reader:
                    rows += 1
                    if row.get("run_id") != run_id:
                        raise MonthlyPolicyError(f"文件run_id不一致: {rel}")
                    raw_date = row.get("date") or row.get("metadata_date")
                    if raw_date:
                        try:
                            seen_dates.append(date.fromisoformat(raw_date))
                        except ValueError as exc:
                            raise MonthlyPolicyError(f"文件日期非法: {rel} {raw_date}") from exc
            if rows != int(record.get("rows", -1)):
                raise MonthlyPolicyError(f"文件行数不一致: {rel}")
            if metadata_mode:
                # fund_snapshot.csv 的一行代表一只ETF元数据快照，8行使用同一个
                # metadata_date，不适用“逐行严格递增且不重复”的K线文件规则。
                if not seen_dates:
                    raise MonthlyPolicyError(f"元数据快照缺少日期: {rel}")
                if any(value != parse_iso_date(required_as_of) for value in seen_dates):
                    raise MonthlyPolicyError(f"元数据快照日期必须全部等于{required_as_of}: {rel}")
                if record.get("start") != required_as_of or record.get("end") != required_as_of:
                    raise MonthlyPolicyError(f"元数据快照日期范围与清单不一致: {rel}")
            elif seen_dates:
                if seen_dates != sorted(seen_dates) or len(seen_dates) != len(set(seen_dates)):
                    raise MonthlyPolicyError(f"文件日期未严格递增或存在重复: {rel}")
                if record.get("start") != seen_dates[0].isoformat() or record.get("end") != seen_dates[-1].isoformat():
                    raise MonthlyPolicyError(f"文件日期范围与清单不一致: {rel}")
    return manifest


def validate_feature_snapshot(expected_run_id: str, expected_as_of: str) -> dict:
    """验证Agent将读取的日K、周K客观参数快照并返回绑定信息。"""

    if not FEATURE_PATH.is_file():
        raise MonthlyPolicyError("缺少app/features/latest.json，不能让Agent脱离客观参数判断")
    payload = json.loads(FEATURE_PATH.read_text(encoding="utf-8"))
    schema = str(payload.get("schema_version") or "")
    if not schema.startswith("features-v"):
        raise MonthlyPolicyError("客观参数快照schema_version非法")
    if payload.get("data_run_id") != expected_run_id:
        raise MonthlyPolicyError("客观参数快照run_id与行情清单不一致")
    if payload.get("as_of") != expected_as_of:
        raise MonthlyPolicyError("客观参数快照截止日与决策数据截止日不一致")
    instruments = payload.get("instruments")
    if not isinstance(instruments, dict) or set(instruments) != set(CONFIG["priority"]):
        raise MonthlyPolicyError("客观参数快照没有完整覆盖8只ETF")

    found_forbidden: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in FEATURE_FORBIDDEN_KEYS:
                    found_forbidden.add(str(key))
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, float) and not math.isfinite(value):
            raise MonthlyPolicyError("客观参数快照含NaN或无穷值")

    walk(payload)
    if found_forbidden:
        raise MonthlyPolicyError(f"客观参数脚本越权输出决策字段: {sorted(found_forbidden)}")
    for code in CONFIG["priority"]:
        item = instruments[code]
        for timeframe in ("weekly", "daily"):
            values = item.get(timeframe)
            if not isinstance(values, dict):
                raise MonthlyPolicyError(f"{code}缺少{timeframe}客观参数")
            if values.get("as_of") != expected_as_of or values.get("is_partial") is not False:
                raise MonthlyPolicyError(f"{code}的{timeframe}不是截止日已完成参数")
        try:
            build_quantitative_evidence(item)
        except AgentGateError as exc:
            raise MonthlyPolicyError(f"{code}缺少Agent多参数门禁所需客观数据: {exc}") from exc
    feature_meta = file_meta(FEATURE_PATH)
    return {
        "path": FEATURE_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": feature_meta["size_bytes"],
        "modified_utc": feature_meta["modified_utc"],
        "schema_version": schema,
    }


def newest_workspace_state(code: str) -> tuple[Path, dict]:
    raise MonthlyPolicyError(
        f"{code}旧训练工作区重建入口已停用；当前分析必须使用冻结冠军最新行情推理"
    )


def _initial_champion(code: str) -> tuple[str, dict]:
    config_path = MODEL_ROOT / "configs" / f"etf_{code}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    champion = config.get("champion") or {}
    if champion.get("kind"):
        return str(champion["kind"]), {"kind": champion["kind"]}
    champion_id = champion.get("id")
    champion_spec = champion.get("spec")
    if not champion_id or not isinstance(champion_spec, dict):
        raise MonthlyPolicyError(f"{code}初始冠军配置不完整")
    return str(champion_id), champion_spec


def load_champion_chain(code: str, state: dict) -> list[dict]:
    _ = state
    raise MonthlyPolicyError(
        f"{code}旧冠军链重建入口已停用；当前分析不回放历史冠军链"
    )


def champion_position_snapshot(code: str, required_as_of: str) -> tuple[dict, float, dict]:
    """兼容入口：冻结冠军读取截止日行情推理，不要求训练锚点数随行情增长。"""
    try:
        reference = build_frozen_reference(required_as_of, project=PROJECT_ROOT)
    except InferenceError as exc:
        raise MonthlyPolicyError(f"冻结冠军最新行情推理失败: {exc}") from exc
    row = next((item for item in reference.get("instruments", []) if item.get("code") == code), None)
    if row is None:
        raise MonthlyPolicyError(f"冻结冠军推理缺少{code}")
    try:
        protection = price_protection_for(Decimal(str(row["reference_raw_close"])))
    except (KeyError, InvalidOperation, MonthlyPolicyError) as exc:
        raise MonthlyPolicyError(f"{code}冻结推理参考收盘价非法") from exc
    current_position = float(row["position"])
    public = {
        "workspace": Path(str(row["champion_source"])).parents[1].name,
        "champion_id": row["champion_id"],
        "trained_anchor_index": row["trained_anchor_index"],
        "inference_anchor_index": row["inference_anchor_index"],
        "latest_anchor_index": row["inference_anchor_index"],
        "position": round(current_position, 10),
        "bullish": bool(row["bullish"]),
        "signal_as_of": row["signal_as_of"],
        "model_features": row["model_features"],
        "inference_mode": "current_frozen_champion_no_training",
    }
    return public, current_position, protection


def _champion_position_snapshot_in_workspace(code: str, required_as_of: str) -> tuple[dict, float, dict]:
    """旧内部入口保留兼容，但统一转入冻结冠军最新行情推理。"""
    return champion_position_snapshot(code, required_as_of)


def build_policy_snapshot(nominal: date, *, verify_all_files: bool = True) -> tuple[dict, dict]:
    portfolio_state = json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
    context = cycle_context(nominal, portfolio_state)
    manifest = validate_manifest(context["required_data_as_of"], verify_all_files=verify_all_files)
    feature_snapshot = validate_feature_snapshot(manifest["run_id"], context["required_data_as_of"])
    # 名义决策与非决策监控必须共用冻结冠军的最新行情推理入口。
    # 训练台账中的 latest_anchor_index / expected_anchors 只描述模型冻结时的
    # 训练快照；行情新增后推理锚点自然增加，不能据此要求重训或推进台账。
    try:
        frozen_reference = build_frozen_reference(
            context["required_data_as_of"], project=PROJECT_ROOT
        )
    except InferenceError as exc:
        raise MonthlyPolicyError(f"冻结冠军最新行情推理失败: {exc}") from exc
    if frozen_reference.get("data_run_id") != manifest["run_id"]:
        raise MonthlyPolicyError("冻结冠军推理与正式行情批次不一致")
    if frozen_reference.get("data_as_of") != context["required_data_as_of"]:
        raise MonthlyPolicyError("冻结冠军推理截止日与名义决策截止日不一致")

    inference_rows = {
        str(item.get("code")): item for item in frozen_reference.get("instruments", [])
    }
    if set(inference_rows) != set(CONFIG["priority"]):
        raise MonthlyPolicyError("冻结冠军推理没有完整覆盖固定8只ETF")
    champions: dict[str, dict] = {}
    protections: dict[str, dict] = {}
    for code in CONFIG["priority"]:
        row = inference_rows[code]
        try:
            protection = price_protection_for(Decimal(str(row["reference_raw_close"])))
        except (KeyError, InvalidOperation, MonthlyPolicyError) as exc:
            raise MonthlyPolicyError(f"{code}冻结推理参考收盘价非法") from exc
        champions[code] = {
            "workspace": Path(str(row["champion_source"])).parents[1].name,
            "champion_id": row["champion_id"],
            "trained_anchor_index": row["trained_anchor_index"],
            "inference_anchor_index": row["inference_anchor_index"],
            # 兼容既有展示字段；其含义现在明确为本次推理末锚，而非训练末锚。
            "latest_anchor_index": row["inference_anchor_index"],
            "position": round(float(row["position"]), 10),
            "bullish": bool(row["bullish"]),
            "signal_as_of": row["signal_as_of"],
            "model_features": row["model_features"],
            "inference_mode": "current_frozen_champion_no_training",
        }
        protections[code] = protection
    allocation = {
        "weights": dict(frozen_reference["b0207_weights_pct"]),
        "cash_pct": frozen_reference["b0207_cash_pct"],
    }
    growth_slot = dict((frozen_reference.get("priority_slot_analysis") or {}).get("159915") or {})
    current_portfolio = portfolio_state.get("last_confirmed_portfolio") or {}
    portfolio_meta = file_meta(PORTFOLIO_PATH)
    snapshot = {
        "schema_version": "monthly-policy-baseline-v2",
        "policy_id": "B0207",
        "policy_role": "mechanical_baseline_only",
        "allocation_engine": "investment_priority_v3",
        "weekly_deploy": bool(CONFIG.get("weekly_deploy", True)),
        "data_run_id": manifest["run_id"],
        "data_as_of": context["required_data_as_of"],
        "nominal_decision_date": context["nominal_decision_date"],
        "decision_date": context["decision_date"],
        "execution_date": context["execution_date"],
        "next_decision_date": context["next_nominal_decision_date"],
        "execution_time": context["execution_time"],
        "is_decision_cycle": True,
        "champions": champions,
        "mechanical_baseline_weights_pct": allocation["weights"],
        "mechanical_baseline_cash_pct": allocation["cash_pct"],
        "priority_slot_analysis": {"159915": growth_slot},
        "feature_snapshot_path": feature_snapshot["path"],
        "feature_snapshot_size_bytes": feature_snapshot["size_bytes"],
        "feature_snapshot_modified_utc": feature_snapshot["modified_utc"],
        "feature_schema_version": feature_snapshot["schema_version"],
        "price_protection": protections,
        "last_confirmed_portfolio_revision": str(portfolio_state.get("decision_revision") or ""),
        "last_confirmed_portfolio_size_bytes": portfolio_meta["size_bytes"],
        "last_confirmed_portfolio_modified_utc": portfolio_meta["modified_utc"],
        "quality_status": manifest["quality_status"],
    }
    return snapshot, portfolio_state


def latest_report_template() -> Path:
    monthly = sorted((PROJECT_ROOT / "scripts").glob("投资决策月报*.html"), reverse=True)
    if monthly:
        return monthly[0]
    weekly = sorted((PROJECT_ROOT / "scripts").glob("投资决策周报*.html"), reverse=True)
    if weekly:
        return weekly[0]
    raise MonthlyPolicyError("没有可用的锁定HTML模板")


def candidate_stub(snapshot: dict, snapshot_path: Path, portfolio_state: dict) -> dict:
    template = latest_report_template()
    snapshot_meta = file_meta(snapshot_path)
    template_meta = file_meta(template)
    confirmed = portfolio_state.get("last_confirmed_portfolio") or {}
    current_weights = confirmed.get("weights_pct") or {}
    feature_payload = json.loads((PROJECT_ROOT / snapshot["feature_snapshot_path"]).read_text(encoding="utf-8"))
    feature_instruments = feature_payload.get("instruments") or {}
    growth_slot = (snapshot.get("priority_slot_analysis") or {}).get("159915") or {}
    try:
        forward_evidence = load_forward_evidence(
            PROJECT_ROOT, data_as_of=snapshot["data_as_of"], data_run_id=snapshot["data_run_id"],
            instruments={code: feature_instruments[code] for code in CONFIG["priority"]},
        )
    except ForwardEvidenceError as exc:
        raise MonthlyPolicyError(str(exc)) from exc
    strategies = []
    comparison_rows = []
    for priority, code in enumerate(CONFIG["priority"], 1):
        protection = snapshot["price_protection"][code]
        name = str((feature_instruments.get(code) or {}).get("name") or code)
        if code == "159915" and growth_slot:
            name = str(growth_slot.get("display_name") or name)
        strategies.append(
            {
                "code": code,
                "name": name,
                "baseline_priority": priority,
                "champion_id": snapshot["champions"][code]["champion_id"],
                "mechanical_champion_position": snapshot["champions"][code]["position"],
                "mechanical_bullish": snapshot["champions"][code]["bullish"],
                "mechanical_baseline_pct": snapshot["mechanical_baseline_weights_pct"][code],
                "current_weight_pct": current_weights.get(code, 0),
                "raw_close": protection["reference_raw_close"],
                "quantitative_evidence": build_quantitative_evidence(feature_instruments[code]),
                "forward_evidence": forward_evidence[code],
                "forward_analysis": empty_forward_analysis(
                    forward_evidence[code], snapshot["execution_date"], snapshot["next_decision_date"],
                ),
                "evidence_review": empty_evidence_review(),
                "v70_protection_review": empty_review(),
                "holding_peak_stop_review": empty_holding_peak_stop_review(),
                "entry_setup": "",
                "countertrend_thesis": "",
                "position_size_reason": "",
                "weekly_evidence": "",
                "daily_evidence": "",
                "timeframe_relationship": "",
                "volume_confirmation": "",
                "relative_advantage": "",
                "evidence_for": "",
                "evidence_against": "",
                "fixed_priority": priority,
                "agent_direction": "",
                "structural_trend": "",
                "momentum_state": "",
                "trade_action": "",
                "new_position_eligible": None,
                "staged_entry": None,
                "holding_inertia_note": "",
                "multi_parameter_conclusion": "",
                "agent_recommendation": "",
                "agent_recommended_weight_pct": None,
                "position_target_pct": None,
                "entry_trigger": "",
                "execution_time": "",
                "cancel_or_exit": "",
                "risk_note": "",
                **(
                    {
                        "priority_slot": {
                            key: growth_slot.get(key)
                            for key in (
                                "slot_id", "members", "selected_member", "slot_position",
                                "member_positions", "slot_weight_pct", "member_weights_pct",
                                "selection_rule", "fund_analysis_path",
                            )
                        },
                        "alternative_fund_analysis": growth_slot.get("fund_analysis"),
                    }
                    if code == "159915" and growth_slot else {}
                ),
            }
        )
        comparison_rows.append(
            {
                "code": code,
                "name": name,
                "mechanical_baseline_pct": snapshot["mechanical_baseline_weights_pct"][code],
                "agent_recommended_pct": None,
                "difference_pct": None,
                "agent_reason": "",
            }
        )
    comparison_rows.append(
        {
            "code": "CASH",
            "name": "现金",
            "mechanical_baseline_pct": snapshot["mechanical_baseline_cash_pct"],
            "agent_recommended_pct": None,
            "difference_pct": None,
            "agent_reason": "",
        }
    )
    rel_snapshot = snapshot_path.relative_to(PROJECT_ROOT).as_posix()
    return {
        "schema_version": "monthly-decision-candidate-v2",
        "status": "awaiting_agent_analysis",
        "policy_snapshot_path": rel_snapshot,
        "policy_snapshot_size_bytes": snapshot_meta["size_bytes"],
        "policy_snapshot_modified_utc": snapshot_meta["modified_utc"],
        "template": {
            "path": template.relative_to(PROJECT_ROOT).as_posix(),
            "size_bytes": template_meta["size_bytes"],
            "modified_utc": template_meta["modified_utc"],
        },
        "report": {
            "schema_version": "monthly-report-v2",
            "report_status": "pending_agent_analysis",
            "generated_at": "",
            "generated_date": "",
            "data_run_id": snapshot["data_run_id"],
            "data_as_of": snapshot["data_as_of"],
            "decision_date": snapshot["decision_date"],
            "execution_date": snapshot["execution_date"],
            "next_decision_date": snapshot["next_decision_date"],
            "decision_authority": "user_explicit_confirmation",
            "policy_id": "B0207",
            "policy_role": "mechanical_baseline",
            "signal_source": "current_champion_models",
            "allocation_engine": "investment_priority_v3",
            "recommendation_maker": "Agent",
            "agent_role": "v70_bearish_plus_holding_peak_stop_only",
            "timeframe_policy": "weekly_primary_daily_confirmation",
            "policy_snapshot_size_bytes": snapshot_meta["size_bytes"],
            "policy_snapshot_modified_utc": snapshot_meta["modified_utc"],
            "feature_snapshot_path": snapshot["feature_snapshot_path"],
            "feature_snapshot_size_bytes": snapshot["feature_snapshot_size_bytes"],
            "feature_snapshot_modified_utc": snapshot["feature_snapshot_modified_utc"],
            "last_confirmed_portfolio": confirmed,
            "execution_protection": snapshot["price_protection"],
            "agent_analysis": {
                "status": "pending",
                "protection_policy": PROTECTION_POLICY,
                "protection_base": "b0207_cycle_baseline",
                "multi_parameter_gate_version": GATE_VERSION,
                "weekly_primary_confirmed": False,
                "weekly_first_second_joint_reviewed": False,
                "daily_confirmation_reviewed": False,
                "all_quantitative_fields_reviewed": False,
                "trend_action_separated": False,
                "future_return_and_entry_reviewed": False,
                "holding_inertia_reviewed": False,
                "all_8_compared": False,
                "first_priority_slot_reviewed": False,
                "current_holding_and_no_action_compared": False,
                "iteration_model_reference": iteration_model_reference(),
                "fixed_priority_order": list(CONFIG["priority"]),
                "priority_slot_analysis": snapshot.get("priority_slot_analysis") or {},
                "recommended_weights_pct": {},
                "recommended_cash_pct": None,
                "baseline_deviations": [],
                "portfolio_thesis": "",
                "no_action_comparison": "",
                "scenario_analysis": {"down": "", "sideways": "", "up": ""},
                "risk_notes": [],
            },
            "user_decision": {
                "status": "pending",
                "confirmed_at": "",
                "confirmation_source": "",
                "final_target_weights_pct": {},
                "final_cash_pct": None,
                "confirmation_note": "",
            },
            "decision_comparison": {
                "status": "pending_agent_analysis",
                "display_order": [*CONFIG["priority"], "CASH"],
                "rows": comparison_rows,
                "user_feedback_required": True,
            },
            "decision_snapshot": {
                "strategies": strategies,
                "portfolio_plan": {
                    "mechanical_baseline_weights_pct": snapshot["mechanical_baseline_weights_pct"],
                    "mechanical_baseline_cash_pct": snapshot["mechanical_baseline_cash_pct"],
                    "agent_recommended_weights_pct": {},
                    "agent_recommended_cash_pct": None,
                    "target_weights_pct": {},
                    "cash_pct": None,
                },
            },
        },
        "html_replacements": [],
        "html_assertions": [],
    }


def prepare(nominal: date, *, verify_all_files: bool = True) -> tuple[Path, Path, dict]:
    with ProjectLock(PROJECT_ROOT / ".monthly-pipeline.lock"):
        snapshot, portfolio_state = build_policy_snapshot(nominal, verify_all_files=verify_all_files)
        snapshot_path = SNAPSHOT_DIR / f"{snapshot['decision_date']}.json"
        snapshot_payload = pretty_json_bytes(snapshot)
        atomic_write(snapshot_path, snapshot_payload)
        atomic_write(SNAPSHOT_DIR / "latest.json", snapshot_payload)
        candidate = candidate_stub(snapshot, snapshot_path, portfolio_state)
        candidate_path = CANDIDATE_DIR / f"{snapshot['decision_date']}.candidate.json"
        atomic_write(candidate_path, pretty_json_bytes(candidate))
        return snapshot_path, candidate_path, file_meta(snapshot_path)


def status_for(nominal: date) -> dict:
    portfolio_state = json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
    context = cycle_context(nominal, portfolio_state)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    metadata_ready = (
        manifest.get("as_of") == context["required_data_as_of"]
        and manifest.get("market_status") == "closed"
        and manifest.get("quality_status") in {"passed", "passed_with_warnings"}
    )
    reason = None
    ready = False
    if metadata_ready:
        try:
            with ProjectLock(PROJECT_ROOT / ".monthly-pipeline.lock"):
                build_policy_snapshot(nominal, verify_all_files=True)
            ready = True
        except (MonthlyPolicyError, OSError, ValueError, json.JSONDecodeError) as exc:
            reason = str(exc)
    else:
        reason = (
            f"本地共同截止日尚未达到要求：当前{manifest.get('as_of')}，"
            f"需要{context['required_data_as_of']}；不联网、不发布月报"
        )
    has_monthly_html = any((PROJECT_ROOT / "scripts").glob("投资决策月报*.html"))
    has_monthly_contract = (PROJECT_ROOT / "app" / "reports" / "monthly" / "latest.json").is_file()
    if has_monthly_html and has_monthly_contract:
        current_formal = "monthly"
    elif has_monthly_html:
        current_formal = "monthly_transition_waiting_policy_snapshot"
    else:
        current_formal = "weekly_legacy_until_first_monthly"
    return {
        "status": "READY" if ready else "BLOCKED",
        **context,
        "current_data_as_of": manifest.get("as_of"),
        "data_run_id": manifest.get("run_id"),
        "current_formal_report": current_formal,
        "reason": reason,
        "checked_at": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
    }
