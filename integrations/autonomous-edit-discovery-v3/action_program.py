"""Host-owned multi-stage actor motion language; no online policy inputs."""
import math
import numpy as np
ACTIONS=('lane_change','decelerate','accelerate','cruise','stop_hold')
LANES=('keep','ego_lane','original_lane')
PROGRAM='native551_car3_action_sequence'
PARAMETERS=('phases',)

def semantics():
 return dict(initial_speed_ratio=1.,initial_lane='original_lane',rules=dict(lane_change='Change lane, preserve entry speed ratio, duration>=2s. For combined speed/lane motion use accelerate or decelerate with lane intent.',decelerate='Target ratio at least0.2 below entry ratio; may also change lane over>=2s.',accelerate='Target ratio at least0.2 above entry ratio; may also change lane over>=2s.',cruise='Preserve entry ratio exactly and lane=keep.',stop_hold='Entry ratio and target ratio must BOTH be0; lane=keep. First decelerate to0, then hold.'),integration='Smooth quintic speed transition, NOT constant acceleration. Peak derivative is1.875 times average ratio change. Absolute acceleration also includes source-curve acceleration; host dense audit is authoritative.',feasibility_estimate='Ignoring source acceleration only, duration>=1.875*source_speed*abs(delta_ratio)/max_acceleration. This is guidance, not certification; leave additional margin.',no_automatic_rewriting=True)

class PhaseError(ValueError):
 def __init__(self,index,message,speed,lane):
  self.phase_index=index;self.entry_speed_ratio=speed;self.entry_lane=lane
  super().__init__('phase[%d]: %s; entry_speed_ratio=%s; entry_lane=%s'%(index,message,speed,lane))

def phase_schema():
 return dict(type='array',minItems=2,maxItems=6,items=dict(type='object',additionalProperties=False,required=['action','duration_s','target_speed_ratio','lane'],properties=dict(action=dict(type='string',enum=list(ACTIONS)),duration_s=dict(type='number',minimum=.8,maximum=6.,multipleOf=.1),target_speed_ratio=dict(type='number',minimum=0.,maximum=1.3,multipleOf=.1),lane=dict(type='string',enum=list(LANES)))))

def validate_phases(phases):
 if not isinstance(phases,list) or not 2<=len(phases)<=6:raise ValueError('Use2..6 meaningful action phases')
 clean=[];speed=1.;lane='original_lane';changes=0;speed_changes=0;total=0.
 for phase_index,phase in enumerate(phases,1):
  try:
   if not isinstance(phase,dict) or set(phase)!={'action','duration_s','target_speed_ratio','lane'}:raise ValueError('Only declared action fields are allowed')
   action=phase['action'];target=phase['target_speed_ratio'];duration=phase['duration_s'];to=phase['lane']
   if action not in ACTIONS or to not in LANES:raise ValueError('Unknown action or lane intent')
   for v,lo,hi in [(duration,.8,6.),(target,0.,1.3)]:
    if type(v) not in (int,float) or not math.isfinite(v) or not lo<=v<=hi or abs(v*10-round(v*10))>1e-6:raise ValueError('Invalid finite0.1-step duration/speed ratio')
   new_lane=lane if to=='keep' else to;delta=target-speed
   if action=='accelerate' and delta<.2-1e-9:raise ValueError('Acceleration must increase ratio by at least0.2')
   if action=='decelerate' and delta>-.2+1e-9:raise ValueError('Deceleration must decrease ratio by at least0.2')
   if action in ('cruise','stop_hold') and (abs(delta)>1e-9 or to!='keep'):raise ValueError('Hold phases cannot silently change speed or lane')
   if action=='stop_hold' and target!=0:raise ValueError('A stop hold requires prior zero speed')
   if action=='lane_change' and (new_lane==lane or duration<2.):raise ValueError('A lane change needs a new registered lane and at least2s')
   if new_lane!=lane and duration<2.:raise ValueError('Lateral transitions require at least2s')
   if action=='lane_change' and abs(delta)>1e-9:raise ValueError('Use decelerate/accelerate with lane intent for combined motion')
   changes+=int(new_lane!=lane or abs(delta)>=.2-1e-9);speed_changes+=int(abs(delta)>=.2-1e-9)
   total+=duration;speed=float(target);lane=new_lane
   clean.append(dict(action=action,duration_s=float(duration),target_speed_ratio=float(target),lane=to))
  except (ValueError,TypeError) as exc:
   raise PhaseError(phase_index,str(exc),speed,lane) from exc
 if total>18.3+1e-9:raise ValueError('Action sequence exceeds the admitted scene time window')
 if changes<2 or not speed_changes:raise ValueError('Need at least two substantive transitions including speed; displacement-only edits are not accepted')
 return clean

def smooth(u):return 10*u**3-15*u**4+6*u**5
def integral(u):return 2.5*u**4-3*u**5+u**6

class Timeline:
 def __init__(self,phases,start=1.5):self.phases=validate_phases(phases);self.start=float(start)
 def evaluate(self,clock):
  clock=np.atleast_1d(np.asarray(clock,dtype=float));elapsed=np.maximum(clock-self.start,0.)
  source=np.minimum(clock,self.start).copy();weight=np.zeros_like(clock);offset=0.;speed=1.;lane=0.
  for p in self.phases:
   duration=p['duration_s'];target=p['target_speed_ratio'];target_lane=lane if p['lane']=='keep' else float(p['lane']=='ego_lane')
   u=np.clip((elapsed-offset)/duration,0.,1.)
   source+=duration*(speed*u+(target-speed)*integral(u))
   weight+=(target_lane-lane)*smooth(u)
   offset+=duration;speed=target;lane=target_lane
  source+=np.maximum(elapsed-offset,0.)*speed
  return source,weight
