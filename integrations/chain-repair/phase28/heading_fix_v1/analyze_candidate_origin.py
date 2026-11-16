"""Summarize measured state ablations; never claim an online fix or success."""

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

import json
pass  # Path is provided by the portable boundary above

def analyze(output):
    output=Path(output)
    report=_hugsim_expand(json.loads((output/'report.json').read_text()))
    if not report['complete']:raise ValueError('Incomplete inference experiment')
    cases=report['cases'];required=('baseline','drop_bev','drop_tracks','drop_both','baseline_repeat')
    if set(cases)!=set(required):raise ValueError('Missing ablation or restored baseline')
    for name in ('baseline','baseline_repeat'):
        if any(v is None or v>1e-4 for v in cases[name]['maximum_array_errors_vs_original'].values()):
            raise ValueError('Baseline mismatch; attribution withheld')
    if any(cases[name]['special_ego_in_decode'] for name in cases):
        finding='Special ego query reached candidate decode: inspect query exclusion bug before interpreting other ablations.'
    else:
        scores={name:max([c['score'] for c in cases[name]['near_target_candidates']],default=0.) for name in required}
        survives={name:scores[name]>=.4 for name in required}
        if not survives['baseline']:
            finding='No threshold-retained candidate near the target in baseline; spatial correspondence requires review.'
        elif not survives['drop_bev'] and survives['drop_tracks']:
            finding='Removing BEV history suppresses this near-target retained detection; removing tracking history alone does not. BEV history contributes under this intervention.'
        elif survives['drop_bev'] and not survives['drop_tracks']:
            finding='Removing tracking state/query composition suppresses this near-target retained detection; removing BEV history alone does not. Tracking state contributes under this intervention.'
        elif not survives['drop_bev'] and not survives['drop_tracks']:
            finding='Either state removal suppresses this retained detection. Both history paths participate or the result is sensitive to their joint representation; do not assign a unique cause.'
        elif not survives['drop_both']:
            finding='Individual removals retain a near-target detection; joint removal suppresses it. A combined history/state effect is supported.'
        else:
            finding='Near-target retained detection survives both state removals. The removed BEV/tracking state is not required in this test; inspect current rendered input/model domain response and remaining metadata.'
    summary=dict(finding=finding,diagnostic_uncertified=True,formal_evaluation_admitted=False,
        caveats=['Spatially nearby boxes after ablation are not guaranteed to be the same semantic object.',
            'Tracking reset changes active query count and attention composition; it does not isolate a single memory tensor.',
            'Ablation effects do not certify physical absence, a deployable suppression rule or successful avoidance.'],
        maximum_near_target_scores={name:max([c['score'] for c in cases[name]['near_target_candidates']],default=0.) for name in required})
    (output/'attribution.json').write_text(json.dumps(summary,indent=2))
    lines=['# 近身车辆检测来源：GPU推理对照','',finding,'',
        '| 分支 | 原候选附近最高车辆分数 |','|---|---:|']
    for name,score in summary['maximum_near_target_scores'].items():lines.append(f'| {name} | {score:.6f} |')
    lines+=['','同一pre39快照/RNG、同一第39步输入；末尾基线回放通过。未执行控制或闭环，未部署修复。','',
        '原候选的query来源见baseline/neural_query_origin.json。完整输出及对照条件见report.json。']
    (output/'归因结果.md').write_text('\n'.join(lines)+'\n')
    return summary

if __name__=='__main__':
    import sys
    print(json.dumps(analyze(sys.argv[1]),indent=2))
