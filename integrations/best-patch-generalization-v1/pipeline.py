"""Cross-scene admission and asset preparation; never substitutes for executed rollouts."""
import argparse,fcntl,importlib.util,subprocess,sys,time,math
from common_eval import *
LOCAL=Path(__file__).resolve().parent
REGISTRY=OUT.parent/'autonomous_patch_agent_v7/campaigns/patch_auto_v7/candidate_registry.json'

def registered_sources():
    path=ROOT/'integrations/patch-generalization-assets-v1/pipeline.py'
    spec=importlib.util.spec_from_file_location('_frozen_asset_contract',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.verify()

def select_best(registry):
    candidates={}
    for key,row in registry.items():
        metrics=row.get('metrics',{})
        if not all(metrics.get(c,{}).get('status')=='completed' for c in ['original551','cutin551']):continue
        score=metrics['cutin551']['score']['hdscore']
        if not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid recorded score')
        candidates[key]=score
    if not candidates:raise ValueError('No completed paired candidate')
    return max(candidates,key=candidates.get),candidates

def verify_bundle(scene,bundle,checkpoint_hash=True):
    m=read(bundle/'source_manifest.json');source=read(REPORTS[scene])
    if m['scene']!=scene or m['frames']!=source['frames'] or m['raw_data']!=str(RAW_BASE/str(scene)) or m['source_checkpoint']!=source['checkpoint']:raise ValueError('Scene/source identity mismatch')
    if checkpoint_hash and sha(m['source_checkpoint'])!=m['source_checkpoint_sha256']:raise ValueError('Checkpoint changed')
    if len(m['frames_data'])!=m['frames'] or [r['frame'] for r in m['frames_data']]!=list(range(m['frames'])):raise ValueError('Incomplete frames')
    for name,h in m['artifacts'].items():
        if sha(bundle/name)!=h:raise ValueError('Changed prepared asset: '+name)
    return m

def verify_raw(bundle,m):
    p=bundle/'raw_metadata/raw_metadata.json'
    if not p.exists():return False
    r=read(p)
    if r['scene_index']!=m['scene'] or [f['frame'] for f in r['frames']]!=list(range(m['frames'])):raise ValueError('Wrong raw scene or coverage')
    if not all(r.get(k) is True for k in ['raw_processed_pose_identity_all_frames','raw_processed_calibration_identity','raw_processed_image_identity_all_images']):raise ValueError('Unverified raw identity')
    if sha(r['raw_record'])!=r['raw_record_sha256']:raise ValueError('Raw record changed')
    return True

def road_diagnostic(bundle,scene):
    import numpy as np
    from scipy.spatial import cKDTree
    road=np.load(bundle/'road_sr_world.npy',mmap_mode='r')
    poses=np.load(bundle/'poses_and_clock.npz')['relative_ego_to_sr_world']
    frame=10;pose=poses[frame]
    x,y=np.meshgrid(np.linspace(-2,2,17),np.linspace(-1.3,1.3,13));local=np.c_[x.ravel(),y.ravel()]
    query=local@pose[:2,:2].T+pose[:2,3];distance=cKDTree(road[:,:2]).query(query)[0]
    result={'source_frame':frame,'road_points':len(road),'footprint_sample_count':len(query),'footprint_size_m':[4,2.6],
        'nearest_road_point_distance_m':{'minimum':float(distance.min()),'median':float(np.median(distance)),'maximum':float(distance.max())},
        'samples_with_road_point_within_0_5m':int((distance<=.5).sum()),'diagnostic_only':True,
        'road_points_do_not_certify_drivable_boundary':True,'source_route_is_not_online_control_reference':True}
    return result,road,poses

def inspect(render=False):
    registered_sources();registry=read(REGISTRY);best,scores=select_best(registry)
    old=OUT/'evaluation_plan.json'
    if old.exists() and read(old)['best_measured_branch']!=best:raise ValueError('Best branch changed; freeze a new named evaluation instead')
    contract={'schema':'held_out_admission_plan_v1','best_measured_branch':best,'completed_target_scores':scores,
        'comparison_arms':['v13','v15','v17'],'baseline_scope':'incremental improvement over v13 reference, not total improvement over raw UniAD',
        'registered_scenes':[172,552,621],'no_scene_specific_policy_changes':True,'fresh_perception_history_per_episode':True,
        'paired_scene_inputs_and_native_evaluator_required':True,'incomplete_or_missing_results_are_not_zero_scores':True,
        'registry_snapshot':asset(REGISTRY),'native_score_deltas':None,'GPU_executed':False,'global_patch_validated':False}
    if not old.exists():write(old,contract)
    report={'schema':'held_out_asset_admission_v1','checked_epoch':time.time(),'best_measured_branch':best,
        'GPU_available_here':Path('/dev/nvidia0').exists(),'new_closed_loop_executed':False,'global_patch_validated':False,
        'scope':'asset preparation and coverage diagnostics only','cases':{}}
    plots=[]
    for scene in REPORTS:
        bundle=OUT/'assets'/str(scene);m=verify_bundle(scene,bundle);raw=verify_raw(bundle,m)
        sem=bundle/'semantic19/semantic_manifest.json';points=bundle/'point_semantics/point_semantics_manifest.json'
        missing=[]
        if not raw:missing.append('verified original record identity')
        if not sem.exists():missing.append('full19 semantic maps')
        if not points.exists():missing.append('background point semantics with unknown retained')
        missing+=['reviewed scene-specific drivable boundary','parameterized runtime and native scoring adapter','new independent paired closed loops']
        diagnostic,road,poses=road_diagnostic(bundle,scene);plots.append((scene,road,poses))
        report['cases'][str(scene)]={'frames':m['frames'],'source_geometry_verified':True,'raw_identity_verified':raw,
            'navigation_command':COMMANDS[scene],'source_manifest':asset(bundle/'source_manifest.json'),
            'road_coverage_diagnostic':diagnostic,'native_evaluation_admitted':False,'missing':missing,'native_score_delta':None}
    if render:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,len(plots),figsize=(15,6))
        for ax,(scene,road,poses) in zip(axes,plots):
            step=max(1,len(road)//40000);ax.scatter(road[::step,0],road[::step,1],s=.15,c='gray',alpha=.35,label='Reconstructed road points')
            ax.plot(poses[:,0,3],poses[:,1,3],c='tab:blue',label='Recorded ego reference')
            ax.scatter(*poses[10,:2,3],c='red',s=25,label='Registered initial frame 10');ax.set_aspect('equal');ax.set_title('Scene '+str(scene));ax.set_xlabel('SR world x (m)');ax.set_ylabel('SR world y (m)');ax.legend(fontsize=7)
        fig.suptitle('Source coverage audit: recorded reference, NOT closed-loop predictions');fig.tight_layout()
        path=OUT/'source_coverage.png';fig.savefig(path,dpi=150);plt.close(fig);report['coverage_figure']=str(path)
    write(OUT/'admission_report.json',report)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['check','prepare','assets','status']);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    if a.stage=='status':
        report=read(OUT/'admission_report.json') if (OUT/'admission_report.json').exists() else {'status':'not_checked'}
        print(json.dumps(report,ensure_ascii=False,indent=2));return
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'preparation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);registered_sources()
        if a.stage in ['prepare','assets']:
            stages=['data','raw'] if a.stage=='prepare' else ['data','raw','semantics','points']
            for scene in REPORTS:
                for stage in stages:
                    write(OUT/'progress.json',{'stage':stage,'scene':scene,'pid':os.getpid(),'GPU_closed_loop_executed':False})
                    subprocess.run([sys.executable,'-u','-B',str(LOCAL/'asset_stage.py'),str(scene),stage,'--device',a.device],check=True)
        report=inspect(render=True);write(OUT/'progress.json',{'stage':'awaiting_scene_admission','report':str(OUT/'admission_report.json'),'GPU_closed_loop_executed':False})
        print(json.dumps({'status':'blocked_scene_admission','best_measured_branch':report['best_measured_branch'],
            'scenes':list(report['cases']),'new_closed_loop_executed':False,'report':str(OUT/'admission_report.json')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
