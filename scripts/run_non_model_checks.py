"""Run only data/publication/holdings tests; no replay, training or baseline evaluation."""
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
MODULES=['test_non_model_safety','test_monthly_details','test_confirmed_monitoring_target',
         'test_next_week_holdings_display','test_gold_macro_data','test_gold_fomc_data',
         'test_turnover_supplement']
STATIC_METHODS=['test_manifest_hashes_rows_and_run_id','test_feature_snapshot_contract_and_freshness',
    'test_feature_snapshot_contains_no_prescriptive_fields','test_persistent_state_uses_current_instruments',
    'test_current_monthly_page_uses_one_as_generated_data_cutoff','test_latest_monthly_json_matches_holdings_and_target',
    'test_actual_portfolio_history_is_queryable','test_static_server_exposes_no_dynamic_api',
    'test_monthly_launcher_has_portable_python_fallbacks']


def main():
    environment={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONIOENCODING':'utf-8'}
    check=subprocess.run([sys.executable,'-B','scripts/check_project.py','--non-model'],cwd=ROOT,env=environment)
    if check.returncode:return check.returncode
    # Separate processes avoid collisions between the two historical scripts packages.
    for name in MODULES:
        result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p',name+'.py'],cwd=ROOT,env=environment)
        if result.returncode:return result.returncode
    names=['test_static_contract.StaticContractTests.'+name for name in STATIC_METHODS]
    result=subprocess.run([sys.executable,'-B','-m','unittest',*names],cwd=ROOT/'tests',env=environment)
    return result.returncode


if __name__=='__main__':raise SystemExit(main())
