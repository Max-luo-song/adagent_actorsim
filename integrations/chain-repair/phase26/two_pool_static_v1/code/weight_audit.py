from common import *
def main():
 source=ROOT/'code/UniAD_SIM/ckpts/uniad_base_e2e.pth'
 config=ROOT/'waymo23-native-import/configs/uniad_runtime_cache.json'
 cfg=read(config);entry=asset(source);entry['bytes']=source.stat().st_size
 if cfg['enabled'] and entry['sha256']!=cfg['checkpoint']['sha256']:raise ValueError('Frozen weight hash mismatches runtime configuration')
 retired=[asset(p) for p in sorted((P26.parent/'phase14/runs').glob('*/training/planning_head_candidate.pth'))]
 if any(r['sha256']==entry['sha256'] for r in retired):raise ValueError('Retired artifact selected')
 paths=[ROOT/'integrations/light-tracking-curvature-v15-r2/worker.py',ROOT/'integrations/light-cutin-diagnostic/worker_trace.py',P26.parent/'phase46/framework_driver_v1/worker.py',P26.parent/'phase46/framework_driver_v1/protocol_branch_worker.py',ROOT/'waymo23-native-import/five_uniad.py',ROOT/'waymo23-native-import/runtime_bundle.py']
 sources=[asset(p) for p in paths]
 for p in paths:
  text=p.read_text()
  if 'planning_head_candidate.pth' in text or 'UNIAD_PLANNING_CHECKPOINT' in text:raise ValueError('Unexpected phase14 override in selected boot chain')
 write(OUT/'weight_lineage_audit.json',{'schema':'frozen_uniad_static_lineage_v1','frozen_base_checkpoint':entry,'runtime_configuration':asset(config),'matches_pinned_runtime_hash':entry['sha256']==cfg['checkpoint']['sha256'],'retired_phase14_artifacts_opaque_hash_only':retired,'selected_worker_boot_sources':sources,'phase14_load_path_in_selected_boot_sources':False,'evidence_level':'static configured loader path and exact whole-file hash; no running worker or model inference inspected','phase14_not_inherited_basis':['user explicitly confirms discarded artifact never adopted','current hash equals configured frozen base weight hash','configured loader copies only base weight, verifies its exact hash, five_uniad selects that base/copy','no phase14 overlay path in inspected boot sources'],'not_a_tensor_ancestry_proof_against_unavailable_upstream_original':True,'scene_exposure':{'3':'zero current-patch exposure per user-confirmed scope; retired training not adopted','23':'same as scene3','31':'same as scene3','233':'same as scene3','552':'historical_test_exposure_repair_not_merged_into_global_patch; Pool_1_only','172':'historical_development_exposure; excluded from zero-exposure Pool_2','621':'source preparation known; zero-exposure audit not independently certified'},'closed_loop_score_read':False,'model_executed':False})
 freeze(OUT/'protected_code_before.json',{str(p):sha(p) for p in (ROOT/'integrations/autonomous-patch-agent-v8').rglob('*.py')})
 print('Frozen UniAD',entry['sha256'],'retired artifacts',len(retired),flush=True)
if __name__=='__main__':main()
