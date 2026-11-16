"""Accept actual control recovery without fabricating a completed failed baseline."""
import copy
from common_flow import AGENT,healthy,read,write,asset
import sys
sys.path.insert(0,str(AGENT))
from engine import Engine

class RecoveryEngine(Engine):
 def initialize(self,campaign):
  state=super().initialize(campaign)
  if not state.get('registry'):state.pop('registry',None)
  state['champion']['profile']='v15_joint';self.save(state);return state
 def finalize_evaluation(self,state,r):
  recovery=bool(state['task_convergence'][state['active_task_id']].get('recovery_baseline'))
  accepted=r['verdict']['accepted']
  super().finalize_evaluation(state,r)
  if recovery and accepted:
   # Establish a real complete baseline only after a complete executed recovery.
   state['reference_metrics']=copy.deepcopy(state['champion']['metrics']);state['champion']['reference_metrics']=copy.deepcopy(state['reference_metrics']);state['champion']['profile']='v15_joint'
   c=copy.deepcopy(state['champion']);key=c['version'];c.update(pool_id=key,profile='v15_joint',eligible=True,evaluated_repairs=0)
   state['champion']['pool_id']=key;state['registry']={key:c}
   write(self.root/'recovery_baseline.json',dict(champion=c,source_failure_retained=True,failed_episode_never_marked_completed=True))
   write(self.root/'candidate_registry.json',state['registry']);self.save(state)

def judge_recovery(candidate,baseline,policy):
 reasons=[]
 for case,n in candidate.items():
  if not healthy(n):reasons.append(case+':candidate_not_complete_healthy')
  b=baseline[case]
  if n['score']['pdms']<b['score']['pdms']-policy['numeric_tolerance']:reasons.append(case+':pdms_regression')
  if n['progress_m']<b['progress_m']*policy['min_progress_ratio'] or n['stopped_s']>b['stopped_s']+policy['max_extra_stopped_seconds']:reasons.append(case+':progress_or_stop_regression')
 for key in ('final_abs_lateral_m','peak_abs_lateral_m'):
  if candidate['original551'][key]>baseline['original551'][key]+policy['lateral_tolerance_m']:reasons.append('original551:'+key+'_regression')
 if candidate['original551']['score']['hdscore']<baseline['original551']['score']['hdscore']-policy['original_max_hd_drop']-policy['numeric_tolerance']:reasons.append('original551:original_hd_regression')
 if not candidate['cutin551']['progress_m']>baseline['cutin551']['progress_m']:reasons.append('cutin551:no_progress_beyond_failure')
 return dict(accepted=not reasons,reasons=reasons,target_hd_gain=None,target_hd_gain_vs_v13=None,comparison_kind='control_failure_recovery',numeric_full_horizon_gain_claim=False,failed_baseline_controls=baseline['cutin551']['controls'],candidate_controls=candidate['cutin551']['controls'],global_patch_validated=False,formal_certification=False)
