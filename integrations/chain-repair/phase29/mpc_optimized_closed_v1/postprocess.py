"""Preserve v2 safety analysis and add matched-prefix performance evidence."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand
Path = _HugsimPath

import json,sys
import numpy as np
pass  # Path is provided by the portable boundary above
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from global_repair_v2.postprocess import main as original

def compare_runs(baseline,current,times,baseline_path):
    a=baseline.get('timing',{}).get('steps',[]);b=current.get('timing',{}).get('steps',[])
    n=min(len(a),len(b));base_seconds=sum(r['wall_seconds'] for r in a[:n]);new_seconds=sum(r['wall_seconds'] for r in b[:n])
    comparison=dict(baseline=str(baseline_path.parent),matched_step_prefix=n,baseline_prefix_step_wall_seconds=base_seconds,optimized_prefix_step_wall_seconds=new_seconds,prefix_step_wall_reduction_percent=100*(1-new_seconds/base_seconds) if base_seconds else None,baseline_total_wall_seconds=baseline.get('wall_seconds'),optimized_total_wall_seconds=current.get('wall_seconds'),optimized_pure_mpc_seconds=sum(r['wall_seconds'] for r in times.get('calls',[])),optimized_mpc_call_count=len(times.get('calls',[])),scope='Fresh GPU closed-loop comparison; timing alone does not certify trajectory equivalence or successful avoidance',formal_evaluation_admitted=False)
    old_trace=baseline.get('trace',[]);new_trace=current.get('trace',[]);trace_n=min(len(old_trace),len(new_trace))
    errors=[]
    for old,new in zip(old_trace[:trace_n],new_trace[:trace_n]):
        errors.append(float(np.max(np.abs(np.asarray(old['state_after'])-np.asarray(new['state_after'])))))
    comparison.update(matched_control_prefix=trace_n,max_state_after_difference=max(errors) if errors else None,actions_exact_for_common_prefix=all(a['action']==b['action'] for a,b in zip(old_trace[:trace_n],new_trace[:trace_n])) if trace_n else None,comparison_valid_only_for_matched_prefix=True)
    return comparison

def main():
    original()
    run=Path(sys.argv[1]);path=run/'analysis/summary.json';summary=_hugsim_expand(json.loads(path.read_text()))
    current=_hugsim_expand(json.loads((run/'run_report.json').read_text()))
    baseline_path=HERE.parent/'runs/global_evidence_closed552_v2/run_report.json'
    baseline=_hugsim_expand(json.loads(baseline_path.read_text()))
    times=_hugsim_expand(json.loads((run/'mpc_timing.json').read_text())) if (run/'mpc_timing.json').exists() else {}
    comparison=compare_runs(baseline,current,times,baseline_path)
    summary.update(mpc_backend='mpc_reuse_v1',performance_comparison=comparison)
    path.write_text(json.dumps(summary,indent=2)+'\n')
    (run/'analysis/performance_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    print(json.dumps(comparison,indent=2),flush=True)
if __name__=='__main__':main()
