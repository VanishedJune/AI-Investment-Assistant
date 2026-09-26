"""Compatibility helpers; incomplete migration publication is permanently retired.

Current reports must use the regular publisher, never replay migration holdings.
"""
from __future__ import annotations

def replace_text(node, text):
    """Replace visible content while retaining every descendant element."""
    for child in node.iter():
        child.text = None
        if child is not node:
            child.tail = None
    node.text = text


def color(value):
    magnitude = abs(value)
    if magnitude < .05:
        return '#d7dde5'
    if value > 0:
        return '#f4a582' if magnitude < .25 else '#d6604d' if magnitude < .75 else '#b2182b'
    return '#4393c3' if magnitude < .75 else '#2166ac'


def resolve_directions(code, features, reference, agent_rows):
    """Resolve actual predictions, never blank a direction because a code changed."""
    if (reference.get('data_as_of') != features['as_of']
            or reference.get('data_run_id') != features['data_run_id']):
        raise ValueError('模型预测与行情批次不一致')
    position = reference['champion_positions'].get(code)
    if position is None or not 0 <= position <= 1:
        raise ValueError('缺少有效的当前模型预测')
    agent = agent_rows.get(code) or {}
    if not agent.get('agent_direction'):
        raise ValueError(f'{code}缺少独立Agent分析；模型预测已可用，不得发布占位月报')
    if (agent.get('data_as_of') != features['as_of']
            or agent.get('data_run_id') != features['data_run_id']):
        raise ValueError('Agent分析与行情批次不一致')
    return ('看多' if position > 0 else '看空'), agent['agent_direction']


def render_snapshot(root, portfolio, migration, features, *, reference=None, agent_rows=None):
    # This legacy migration renderer must never overwrite a completed report
    # with forced null directions. Callers must supply current predictions.
    if reference is None or agent_rows is None:
        raise ValueError('禁止发布不完整换仓快照：必须提供冻结模型当前预测和同批次Agent分析')
    for code in features['instrument_order']:
        resolve_directions(code, features, reference, agent_rows)
    raise ValueError('旧换仓快照发布入口已停用；请使用正式月报发布器保留最新实际持仓')


if __name__ == '__main__':
    raise SystemExit('旧占位快照入口已停用，未修改报告。模型预测使用 model_iteration/scripts/monitoring_baseline_report.py --check；正式报告使用正常发布流程。')
