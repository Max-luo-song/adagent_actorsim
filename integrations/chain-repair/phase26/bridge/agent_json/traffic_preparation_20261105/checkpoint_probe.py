
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

pass  # Path is provided by the portable boundary above
import json,torch,numpy as np
from PIL import Image,ImageDraw
p=Path('outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26');out=p/'bridge/agent_json/traffic_preparation_20261105';out.mkdir(exist_ok=True)
ckpt=Path(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/adscenegraph/Omniscenegraph/tmp/outputs/waymo/waymo_552/20260418_lidar+cam0_1_2adstudiov_pbr_3view/checkpoint_final.pth'))
m=torch.load(ckpt,map_location='cpu',mmap=True,weights_only=False)['models']['TrafficLightNodes'];info=_hugsim_expand(json.loads(Path(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/adscenegraph/data/waymo/processed/training/552/instances/instances_info.json')).read_text()))
origin=np.loadtxt('outputs/waymo552_three_setup_20261018_043131/inputs/ego_pose/000.txt');inv=np.linalg.inv(origin)
models=[]
for i in range(12):
 frames=m['instances_fv'][:,i].nonzero().flatten();ids=m['points_ids'].flatten()==i
 if not len(frames):continue
 t=int(frames[len(frames)//2]);center=m['instances_trans'][t,i].numpy()
 matches=[]
 for k,r in info.items():
  if r.get('class_name')!='TrafficLight' or t not in r['frame_annotations']['frame_idx']:continue
  j=r['frame_annotations']['frame_idx'].index(t);pose=inv@np.array(r['frame_annotations']['obj_to_world'][j]);matches.append((float(np.linalg.norm(pose[:3,3]-center)),k,r['id']))
 models.append(dict(model_index=i,point_count=int(ids.sum()),first_frame=int(frames[0]),last_frame=int(frames[-1]),match=min(matches) if matches else None))
report=dict(schema='traffic_checkpoint_audit_v1',checkpoint=str(ckpt),total_points=len(m['_means']),instances=models)
(out/'checkpoint_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
run=p/'bridge/unified/agent_json_runs/20261105T020241Z/session/trajectories/legacy_attitude/packets.json';packet=_hugsim_expand(json.loads(run.read_text()))['frames'][0]['new']
cam=packet['cameras']['CAM_FRONT'];im=Image.open(cam['rgb']).convert('RGB');draw=ImageDraw.Draw(im);k=np.asarray(cam['K']);w2c=np.linalg.inv(np.asarray(cam['camera_to_world']))
for i in range(12):
 if not m['instances_fv'][80,i]:continue
 center=m['instances_trans'][80,i].numpy();point=w2c@np.r_[center,1];pixel=k@point[:3]
 if point[2]>0:
  u,v=pixel[:2]/pixel[2];draw.rectangle([u-15,v-22,u+15,v+22],outline='red',width=2);draw.text((u+17,v-12),f'idx{i} n={int((m["points_ids"].flatten()==i).sum())}',fill='red');print('project',i,u,v)
im.save(out/'frame80_front_labels.png')
