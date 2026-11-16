"""No averaging, no null scores, no unequal-length or replay promotion."""
import math
METRICS=('hdscore','pdms','rc')
def values(raw):
 if not isinstance(raw,dict):return None
 out={}
 for k in METRICS:
  v=raw.get(k);v=v.get('value') if isinstance(v,dict) else v
  if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):return None
  out[k]=float(v)
 return out

def compare(base,candidate,controls):
 def valid(x):return x.get('status')=='completed' and x.get('controls')==controls and x.get('fresh_history') is True and x.get('executed_path_scored') is True and x.get('safe') is True and values(x.get('scores')) is not None
 if not all(valid(x) for x in [base,candidate]):return {'passed':False,'comparable':False,'deltas':None,'reason':'Need complete safe fresh independent episodes and finite actual executed-path metrics'}
 b,n=values(base['scores']),values(candidate['scores']);d={k:n[k]-b[k] for k in METRICS}
 return {'passed':all(v>=-1e-9 for v in d.values()),'comparable':True,'deltas':d,'reason':'per_metric_nonregression'}

def replay_gate(r):
 # Existing failures may remain registered, but a newly introduced refusal blocks GPU.
 allowed={'552_v15':[24]}
 expected={'172_v15':22,'172_native':36,'551_original':189,'551_cutin':189,'552_v15':25}
 if set(r)!=set(expected) or any(len(r[k].get('rows',[]))!=n for k,n in expected.items()):return False
 return all('rows' in a and a.get('same_observation_counterfactual_only') is True and a.get('no_action_frames')==allowed.get(k,[]) for k,a in r.items())
