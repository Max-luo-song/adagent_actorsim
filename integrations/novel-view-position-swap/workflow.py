"""Explicit scene edit + new-view preview; no default model or training mutation."""

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

import argparse,importlib.util,sys,json
pass  # Path is provided by the portable boundary above
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];BASE=ROOT/'integrations/novel-view-distillation'
sys.path[:0]=[str(HERE),str(BASE)]
from contracts import P26,read,write,sha
from geometry import design
OLD=P26/'novel_distillation_large_motion_v1/scene551_lanechange4m'
EFFECTIVE=P26/'light_map551_v4/A/actor_tracks.npz'
SOURCE=P26.parents[0]/'phase33/scene_data551_v1/runs/scene551_20261106T050218Z'
OUTPUT=P26/'novel_view_position_swap_v1/scene551_ego_native3'

def asset(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def verify_asset(a):
    if sha(a['path'])!=a['sha256']:raise ValueError('Changed swap input: '+a['path'])
def pins():return {str(p):sha(p) for p in sorted(HERE.glob('*.py'))+sorted(HERE.glob('*.sh'))}

def prepare(out):
    out=Path(out).resolve()
    if out.exists():check(out);return dict(prepared=True,reused=True,output=str(out))
    oldjob=read(OLD/'job.json');compiled=read(OLD/'compiled_trajectory.json')
    if sha(OLD/'compiled_trajectory.json')!=oldjob['trajectory_sha256']:raise ValueError('Source ego pose pack changed')
    source=read(SOURCE/'source_manifest.json');identity=read(source['instances_info.json']['path'])['5']
    if identity['id']!='o24FGKGBO7aPlRVYkDyU7A' or identity['class_name']!='Vehicle':raise ValueError('Target actor identity changed')
    with np.load(EFFECTIVE) as z:actors={k:z[k].copy() for k in z.files}
    with np.load(SOURCE/'actors.npz') as z:raw={k:z[k].copy() for k in z.files}
    if not actors['RigidNodes__instances_fv'][:,3].all():raise ValueError('Target does not have full source visibility')
    ego,car,audit=design(compiled['samples'],actors,compiled['gravity_up_source'])
    frames=np.array([s['source_frame'] for s in compiled['samples']]);planned=raw['RigidNodes__instances_trans'][:,3].copy();planned[frames]=car
    # Additional actor checks use current source state only, for offline diagnosis.
    from scipy.spatial.transform import Rotation
    from geometry import footprint
    conflicts=[];minimum=float('inf')
    for i,frame in enumerate(frames):
        ego_poly=footprint(ego[i],3.,1.6);car_pose=np.eye(4);car_pose[:3,3]=car[i]
        car_pose[:3,:3]=Rotation.from_quat(actors['RigidNodes__instances_quats'][frame,3][[1,2,3,0]]).as_matrix()
        car_poly=footprint(car_pose,*actors['RigidNodes__instances_size'][3,:2])
        for group in ['RigidNodes','DeformableNodes','SMPLNodes']:
            for other in np.flatnonzero(actors[group+'__instances_fv'][frame]):
                if group=='RigidNodes' and other==3:continue
                t=actors[group+'__instances_trans'][frame,other];q=actors[group+'__instances_quats'][frame,other].reshape(-1,4)[0]
                size=actors[group+'__instances_size'][other];pose=np.eye(4);pose[:3,3]=t;pose[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();poly=footprint(pose,*size[:2])
                gap=min(ego_poly.distance(poly),car_poly.distance(poly));minimum=min(minimum,float(gap))
                if gap<.1:conflicts.append(dict(source_frame=int(frame),group=group,index=int(other),gap_m=float(gap)))
    audit.update(other_actor_conflicts=conflicts,minimum_other_actor_gap_m=minimum)
    if conflicts:raise ValueError('Swap intersects another source actor: '+str(conflicts[:5]))
    out.mkdir(parents=True)
    np.savez_compressed(out/'swap_program.npz',frames=frames,ego_poses=ego,car_trans=planned,
        original_car_trans=raw['RigidNodes__instances_trans'][:,3],original_car_quats=raw['RigidNodes__instances_quats'][:,3],
        original_ego=np.array([s['recorded_ego_to_world'] for s in compiled['samples']]))
    write(out/'ego_trajectory.json',dict(schema='ego_trajectory_v1',coordinate_frame='source_world',attitude_policy='gravity_level',start_frame=0,
        waypoints=[dict(time_s=s['time_s'],ego_to_world=p.tolist()) for s,p in zip(compiled['samples'],ego)]))
    write(out/'geometry_report.json',audit)
    assets={name:asset(path) for name,path in dict(source_pose_pack=OLD/'compiled_trajectory.json',source_job=OLD/'job.json',
        effective_actor_tracks=EFFECTIVE,source_actors=SOURCE/'actors.npz',source_identity=source['instances_info.json']['path'],
        swap_program=out/'swap_program.npz',trajectory=out/'ego_trajectory.json').items()}
    # Freeze the already accepted appearance policy and its source implementation.
    spec=importlib.util.spec_from_file_location('_swap_conditioned_provider',BASE/'regeneration_v5/workflow.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assets['conditioning_policy']=asset(module.POLICY)
    write(out/'plan.json',dict(schema='position_swap_preview_v1',assets=assets,code=pins(),provider_code=module.family_pins(),
        source_task=oldjob['task'],source_input_assets=oldjob['inputs'],conditioning=oldjob['conditioning'],
        output=str(out),object_id='native:RigidNodes:3',target_index=3,frames=frames.tolist(),
        training_performed=False,control_executed=False,default_model_changed=False))
    return dict(prepared=True,output=str(out),minimum_ego_target_gap_m=audit['minimum_ego_target_footprint_gap_m'],minimum_other_actor_gap_m=minimum)

def check(out):
    out=Path(out).resolve();plan=read(out/'plan.json')
    if plan['code']!=pins():raise ValueError('Swap adapter code changed; use a new output')
    for a in list(plan['assets'].values())+list(plan['source_input_assets'].values()):verify_asset(a)
    return plan

def preview(out):
    out=Path(out).resolve();plan=check(out);stage=out/'preview'
    if stage.exists():raise FileExistsError('Existing preview retained: '+str(stage))
    import preview as rendering
    return rendering.run(out,plan)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','check','preview']);p.add_argument('--output',default=str(OUTPUT));a=p.parse_args()
    if a.stage=='prepare':r=prepare(a.output)
    elif a.stage=='check':check(a.output);r=dict(verified=True,output=a.output)
    else:r=preview(a.output)
    print(json.dumps(r,indent=2))

if __name__=='__main__':main()
