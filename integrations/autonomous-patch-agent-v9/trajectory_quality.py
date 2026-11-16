"""Host-only kinematic checks on executed references. No model-supplied waivers."""

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

import math
pass  # Path is provided by the portable boundary above
import numpy as np
from core import AgentError,finite,read,verify,asset

def _alternates(values,epsilon):
 signs=np.sign(np.asarray(values)[np.abs(values)>epsilon]);return len(signs)>2 and np.count_nonzero(signs[1:]!=signs[:-1])>=2

def _valid_pose(p):
 return p.shape==(4,4) and np.isfinite(p).all() and np.allclose(p[3],[0,0,0,1],atol=1e-6) and np.allclose(p[:3,:3].T@p[:3,:3],np.eye(3),atol=1e-5) and np.linalg.det(p[:3,:3])>0

def assess(plans,vehicle,policy,verified_risks=None):
 """plans: timestamp, ego_to_world, dense vehicle-frame [forward,right,yaw,v,steer]."""
 v=vehicle;p=policy;risks=verified_risks or {};reasons=[];audits=[];previous=None;executed_acc=[];executed_steer=[];low_speed_samples=0
 if not plans:return {'passed':False,'reasons':['missing_dense_reference']}
 dt=float(v['dt']);acc_min=v['min_acc_mps2'];acc_max=v['max_acc_mps2'];max_steer=math.radians(v['max_steer_degrees']);max_rate=math.radians(v['max_steer_rate_degrees'])
 for i,item in enumerate(plans):
  try:
   t=float(item['timestamp']);s=np.asarray(item['states'],float);pose=np.asarray(item['ego_to_world'],float)
   if not finite(t) or s.ndim!=2 or s.shape[1]!=5 or len(s)<4 or not np.isfinite(s).all() or not _valid_pose(pose):raise ValueError('malformed plan or pose')
   if previous and abs(t-previous['time']-dt)>1e-6:raise ValueError('missing or repeated execution clock')
   speed=s[:,3];low=speed<p['low_speed_mps'];low_speed_samples+=int(low.sum());a=np.diff(speed)/dt;jerk=np.diff(a)/dt;steer_rate=np.diff(s[:,4])/dt
   physical=bool(np.all(speed>=-1e-8) and np.all(speed<=v['max_speed_mps']+1e-7) and np.all(a>=acc_min-1e-6) and np.all(a<=acc_max+1e-6) and np.all(np.abs(s[:,4])<=max_steer+1e-6) and np.all(np.abs(steer_rate)<=max_rate+1e-6))
   if not physical:reasons.append(str(i)+':physical_bounds')
   # Mask inferred differential geometry only. Declared steering/rate bounds always hold.
   yaw=np.unwrap(s[:,2]);yawrate=np.diff(yaw)/dt;moving=(speed[:-1]>=p['low_speed_mps'])&(speed[1:]>=p['low_speed_mps']);curvature=np.divide(yawrate,np.maximum(speed[1:],p['low_speed_mps']))
   inferred=np.arctan(v['wheelbase_m']*curvature);valid_rates=moving[:-1]&moving[1:]
   if np.any(np.abs(inferred[moving])>max_steer+1e-5) or np.any(np.abs(np.diff(inferred)[valid_rates]/dt)>max_rate+1e-4):reasons.append(str(i)+':inferred_curvature_or_rate')
   # An abrupt position overwrite is not excused by low speed or emergency status.
   displacement=np.linalg.norm(np.diff(s[:,:2],axis=0),axis=1);travel=np.maximum(speed[:-1],speed[1:])*dt+.5*max(abs(acc_min),acc_max)*dt*dt+.02
   if np.any(displacement>travel):reasons.append(str(i)+':position_kinematic_jump')
   if np.linalg.norm(s[0,:2])>.05:reasons.append(str(i)+':reference_not_at_current_vehicle')
   actual=item.get('action',{});aa=actual.get('acc_mps2');sr=actual.get('steer_rate_radps')
   if not finite(aa) or not finite(sr) or not acc_min-1e-7<=aa<=acc_max+1e-7 or abs(sr)>max_rate+1e-7:reasons.append(str(i)+':executed_control_bounds')
   executed_acc.append(aa if finite(aa) else 0.);executed_steer.append(sr if finite(sr) else 0.)
   oscillation=_alternates(executed_acc[-5:],p['acceleration_sign_epsilon_mps2']) or _alternates(executed_steer[-5:],p['steer_sign_epsilon_rad'])
   if oscillation:reasons.append(str(i)+':alternating_execution')
   risk=risks.get(i,{});urgent=risk.get('host_verified') is True and finite(risk.get('ttc_s')) and 0<=risk['ttc_s']<=p['low_ttc_s'] and abs(risk.get('timestamp',float('inf'))-t)<1e-7
   braking=bool(np.all(a<=p['acceleration_sign_epsilon_mps2']) and np.any(a<-.05) and aa is not None and finite(aa) and aa<=0 and not _alternates(a,.05) and not oscillation)
   # Waive only excessive negative onset jerk, never rebound/positive shock or bounds.
   stop_clip=np.array([speed[k+1]<=1e-8 and speed[k+2]<=1e-8 and a[k]<0 and abs(a[k+1])<1e-8 and np.all(speed[k+1:]<=1e-8) for k in range(len(jerk))])
   shocks=(np.abs(jerk)>p['max_jerk_mps3']+1e-5)&~stop_clip;waiver=bool(urgent and braking and np.all(jerk[shocks]<0) and physical)
   actual_jerk=(executed_acc[-1]-(executed_acc[-2] if i else 0.))/dt
   actual_stop_clip=bool(i and speed[0]<=1e-8 and abs(executed_acc[-1])<1e-8 and executed_acc[-2]<0 and np.all(speed<=1e-8))
   if (np.any(shocks) and not waiver) or (abs(actual_jerk)>p['max_jerk_mps3']+1e-5 and not actual_stop_clip and not (waiver and actual_jerk<0)):reasons.append(str(i)+':longitudinal_jerk')
   # Vehicle right is negative source-world lateral: motion_pose uses [forward,-right].
   xyz=np.c_[s[:,0],-s[:,1],np.zeros(len(s))];world=xyz@pose[:3,:3].T+pose[:3,3];times=t+np.arange(len(s))*dt
   if previous:
    end=min(times[-1],previous['times'][-1],t+p['alignment_horizon_s']);aligned_times=times[times<=end+1e-7]
    if not len(aligned_times):raise ValueError('no common time support')
    old=np.column_stack([np.interp(aligned_times,previous['times'],previous['world'][:,j]) for j in range(3)]);new=world[:len(aligned_times)]
    if np.max(np.linalg.norm(new-old,axis=1))>p['max_position_jump_m']:reasons.append(str(i)+':aligned_position_jump')
    active=(speed[:len(aligned_times)]>=p['low_speed_mps'])&(np.interp(aligned_times,previous['times'],previous['speeds'])>=p['low_speed_mps'])
    old_steer=np.interp(aligned_times,previous['times'],previous['steering']);steer_delta=np.abs(s[:len(aligned_times),4]-old_steer)
    if np.any(steer_delta[active]>max_rate*(dt+p['alignment_horizon_s'])+1e-5):reasons.append(str(i)+':aligned_steering_discontinuity')
    # Independent of translations and global yaw; catches alternating future lateral intent.
    old_right=-np.asarray(pose[:3,1]);delta=float((new[-1]-old[-1])@old_right)
    previous_intents=[a['aligned_lateral_delta_m'] for a in audits[-4:] if a['aligned_lateral_delta_m'] is not None]+[delta]
    if _alternates(previous_intents,.03):reasons.append(str(i)+':alternating_aligned_lateral_reference')
   else:delta=None
   audits.append({'index':i,'emergency_negative_jerk_waiver':waiver,'aligned_lateral_delta_m':delta,'low_speed_geometry_masked_samples':int(low.sum())});previous={'time':t,'world':world,'times':times,'steering':s[:,4].copy(),'speeds':speed.copy()}
  except (ValueError,TypeError,KeyError,IndexError) as e:reasons.append(str(i)+':invalid_reference:'+str(e));break
 return {'passed':not reasons,'reasons':list(dict.fromkeys(reasons)),'plans':len(plans),'low_speed_geometry_masked_samples':low_speed_samples,'physical_steering_limits_never_waived':True,'audit':audits}

def for_metrics(metrics,config,simulation_only=False):
 if simulation_only:return {'passed':True,'simulation_fixture_only':True,'not_a_real_motion_certificate':True}
 vehicle=read(Path(__file__).resolve().parents[2]/'waymo23-native-import/configs/vehicle.json');cases={}
 for case,m in metrics.items():
  try:
   verify(m['report']);r=read(m['report']['path']);plans=[];risks={}
   for i,t in enumerate(r['trace']):
    plans.append({'timestamp':t['timestamp_s'],'ego_to_world':t['ego_to_world'],'states':t['control']['reference_states'],'action':t['action']})
    binding=t.get('host_risk_receipt')
    if binding:
     verify(binding);e=read(binding['path'])
     if e.get('schema')!='host_ttc_v1' or e.get('contract')!=m['contract'] or e.get('observation')!=t.get('perception_asset') or e.get('index')!=i:raise AgentError('Host TTC witness binding mismatch')
     risks[i]={'host_verified':True,'ttc_s':e['ttc_s'],'timestamp':e['timestamp']}
   cases[case]=assess(plans,vehicle,config['host_governance']['motion'],risks)
  except (AgentError,OSError,KeyError,TypeError,ValueError):cases[case]={'passed':False,'reasons':['missing_or_invalid_host_motion_evidence']}
 return {'passed':all(c['passed'] for c in cases.values()) and bool(cases),'cases':cases}
