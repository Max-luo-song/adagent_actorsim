
# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

import json,pickle,sys,time,copy
import numpy as np
from paths import *
sys.path.insert(0,str(DRIVER))
from bootstrap import *
from runtime_configuration import load_vehicle_config
from white_edge import detect,WhiteEdge,restrict_road,geometry,project,unproject
from drivable_area import DrivableArea
SOURCE=CR/'phase43/runs/cutin551_20261109T045728Z_11636'
OUT=HERE.parent/'reviews';OUT.mkdir(exist_ok=True)
def draw(idx,obs,info,p,contract,states):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 rgb=obs['rgb']['CAM_FRONT'];h,w=rgb.shape[:2];K,M,z=geometry(info,'CAM_FRONT',rgb.shape);road=DrivableArea(contract,vehicle);yy,xx=np.mgrid[:h,:w];g=unproject(np.c_[xx.ravel(),yy.ravel()],K,M,z);uv,d=project(g,K,M)
 bounds=road.bounds;col=np.floor((g[:,0]-bounds[0])/road.dx).astype(int);row=np.floor((g[:,1]-bounds[1])/road.dy).astype(int)
 valid=(d>0)&(g[:,1]>1)&(g[:,1]<40)&(abs(g[:,0])<15)&(col>=0)&(col<road.mask.shape[1])&(row>=0)&(row<road.mask.shape[0])
 depth=np.asarray(obs['depth']['CAM_FRONT']).reshape(-1);alpha=np.asarray(obs['depth_alpha']['CAM_FRONT']).reshape(-1)
 valid&=(alpha>.6)&(abs(depth-d)<np.maximum(1.,.15*d))
 allowed=np.zeros(len(g),bool);allowed[valid]=road.mask[row[valid],col[valid]]
 render=rgb.copy().reshape(-1,3).astype(float);render[allowed]=.6*render[allowed]+.4*np.array([25.,190.,70.]);render=render.reshape(rgb.shape).clip(0,255).astype(np.uint8)
 fig,(ax,bx)=plt.subplots(1,2,figsize=(16,7));ax.imshow(render);ax.set_axis_off();ax.set_title('Observed white edge limits the derived road mask')
 bx.imshow(road.mask,origin='lower',extent=bounds[[0,2,1,3]],interpolation='nearest',cmap='Greens',vmin=0,vmax=1);bx.set_xlim(-9,7);bx.set_ylim(-5,38);bx.set_aspect('equal');bx.set_xlabel('Model x: right (m)');bx.set_ylabel('Model y: forward (m)')
 for e in p['edges']:
  y=np.linspace(*e['valid_y_range'],200);x=e['a']*y+e['b'];xyz=np.c_[x,y,np.full(len(y),z)];uv,d=project(xyz,K,M);ok=(d>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
  ax.plot(uv[ok,0],uv[ok,1],color='yellow',lw=2,label='Observed solid road edge');bx.plot(x,y,color='orange',lw=2,label='Observed edge')
 if states is not None:
  xyz=np.c_[states[:,:2],np.full(len(states),z)];uv,d=project(xyz,K,M);ok=(d>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
  ax.plot(uv[ok,0],uv[ok,1],color='deepskyblue',lw=3,label='OLD plan (violates new edge)');bx.plot(states[:,0],states[:,1],color='deepskyblue',lw=2,label='OLD plan')
 bx.scatter([0],[-np.array(info['model_lidar_frame']['virtual_to_ego'])[1,3]],color='black',label='Current ego center');bx.legend(loc='upper left');ax.legend(loc='upper left',fontsize=8)
 fig.suptitle(f'Step {idx:03d}: WHITE line is a separate hard boundary; model mask is only soft road evidence')
 fig.text(.5,.02,'Green = derived road mask inside observed edge, not an obstacle-free claim. Camera projection assumes local flat ground. Blue is archived, not a new rollout.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.04,1,.95));path=OUT/f'white_edge_step{idx:03d}.png';fig.savefig(path,dpi=140);plt.close(fig)
vehicle=load_vehicle_config();rows=[];began=time.monotonic()
for idx in range(100):
 folder=SOURCE/f'uniad_worker/step_{idx:03d}'
 with (folder/'observation.pkl').open('rb') as f:obs,info=pickle.load(f)
 t=time.monotonic();p=detect(obs,info);seconds=time.monotonic()-t;edge=WhiteEdge(p,vehicle);c=np.array(info['model_lidar_frame']['virtual_to_ego']);initial=np.array([[-c[0,3],-c[1,3],0.,info['ego_velo'],info['ego_steer']]])
 result=_hugsim_expand(json.loads((folder/'optimization.json').read_text()));selected=result.get('selected');selected=selected or result.get('braking_fallback');states=np.array(selected['states']) if selected and selected.get('states') else None
 initialgap=float(edge.clearance(initial)[0]);gap=float(edge.swept_clearance(states).min()) if states is not None else float('inf')
 rows.append(dict(step=idx,status=p['status'],edges=p['edges'],detector_seconds=seconds,old_initial_gap_m=initialgap if np.isfinite(initialgap) else None,old_plan_gap_m=gap if np.isfinite(gap) else None,old_plan_rejected=bool(gap< -1e-5)))
 if idx in (0,49):
  contract=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()))['predicted_drivable_area'];contract=restrict_road(contract,p,vehicle);draw(idx,obs,info,p,contract,states)
 if idx%20==0:print('Reviewed',idx,flush=True)
report=dict(source=str(SOURCE),frames=len(rows),fresh_frames=sum(r['status']=='fresh' for r in rows),old_plans_rejected=sum(r['old_plan_rejected'] for r in rows),first_old_plan_rejected=next((r['step'] for r in rows if r['old_plan_rejected']),None),first_archived_body_outside=next((r['step'] for r in rows if r['old_initial_gap_m'] is not None and r['old_initial_gap_m']< -1e-5),None),median_detector_seconds=float(np.median([r['detector_seconds'] for r in rows])),rows=rows,gpu_executed=False,new_closed_loop_executed=False,ground_truth_used=False,limitation='Straight visible white road edge plus shoulder colour transition only; local ground-plane approximation. Does not certify new closed-loop avoidance.')
(OUT/'white_edge_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print({k:v for k,v in report.items() if k!='rows'},flush=True)
