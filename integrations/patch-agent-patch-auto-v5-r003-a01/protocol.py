"""Preregistered paired engineering validation; incomplete coverage never passes."""
import math
CASES={'original551':dict(scene=551,edited=False,role='nonregression',ready=True),
       'cutin551':dict(scene=551,edited=True,role='improvement',ready=True),
       'original172':dict(scene=172,edited=False,role='nonregression',ready=False),
       'original552':dict(scene=552,edited=False,role='nonregression',ready=False),
       'original621':dict(scene=621,edited=False,role='nonregression',ready=False)}
LIMITS=dict(max_hd_drop=.01,max_pdms_drop=.01,min_progress_ratio=.95,max_extra_stopped_s=1.,minimum_target_hd_gain=.01)
SHARED=('scene','edited','seed','controls','initial_context','scene_manifest','map','scenario','target_track','vehicle','worker','native_collision','tracker','evaluator','termination','source_assets')
def shared_digest_data(contract):return {key:contract[key] for key in SHARED}
def validate_contracts(left,right):
    if shared_digest_data(left)!=shared_digest_data(right):raise ValueError('Paired exogenous inputs or execution/evaluation contract changed')
    if left['patch']!='on' or right['patch']!='on' or left.get('comparison_version')!='v13' or right.get('comparison_version')!='patch_auto_v5_r003':raise ValueError('Need frozen full-v13 and independent full-v17 episodes')
    if left.get('fresh_history') is not True or right.get('fresh_history') is not True:raise ValueError('History replay is not admitted')
    if left.get('score_kind')!='executed_path' or right.get('score_kind')!='executed_path':raise ValueError('Planning score cannot replace execution')
def evaluate_pair(case,left,right):
    if left is None or right is None:return dict(status='pending',passed=False,reason='both independent arms required')
    validate_contracts(left['contract'],right['contract'])
    errors=[]
    for name,x in [('off',left),('on',right)]:
        if x['status'] in ('failed','resource_limited','initializing') or x.get('score') is None:return dict(status='invalid',passed=False,reason=name+' has incomplete execution/scoring')
        for value in [x['score'].get(k) for k in ('hdscore','pdms')]+[x.get(k) for k in ('progress_m','stopped_s')]:
            if not isinstance(value,(int,float)) or not math.isfinite(value):return dict(status='invalid',passed=False,reason='nonfinite/missing execution metric')
    if CASES[case]['role']=='nonregression' and (left['status']!='completed' or left['collision'] or left['map_departure']):
        return dict(status='invalid',passed=False,reason='original-scene baseline is unhealthy; diagnose before nonregression certification')
    if right['collision']:errors.append('patched_collision')
    if right['map_departure']:errors.append('patched_map_departure')
    if right['unresolved_stop']:errors.append('patched_stop_unresolved')
    if right['status']!='completed':errors.append('patched_episode_not_complete')
    hd=right['score']['hdscore']-left['score']['hdscore'];pdms=right['score']['pdms']-left['score']['pdms']
    if hd < -LIMITS['max_hd_drop']:errors.append('hd_regression')
    if pdms < -LIMITS['max_pdms_drop']:errors.append('pdms_regression')
    if right['progress_m'] < left['progress_m']*LIMITS['min_progress_ratio']:errors.append('progress_regression')
    if right['stopped_s']>left['stopped_s']+LIMITS['max_extra_stopped_s']:errors.append('extra_stop')
    if CASES[case]['role']=='improvement' and hd<LIMITS['minimum_target_hd_gain']:errors.append('no_target_improvement')
    return dict(status='passed' if not errors else 'failed',passed=not errors,reasons=errors,hd_gain=hd,pdms_gain=pdms,formal_score=left.get('formal_score',False) and right.get('formal_score',False))
def overall(results):
    missing=[name for name in CASES if name not in results or results[name]['status'] in ('pending','unavailable')]
    failed=[name for name,r in results.items() if r['status'] in ('failed','invalid')]
    return dict(status='failed' if failed else 'pending' if missing else 'engineering_matrix_passed',
        global_patch_validated=not missing and not failed,missing_cases=missing,failed_cases=failed,
        scope='only this frozen matrix; no universal no-regression guarantee',formal_certification=False)
