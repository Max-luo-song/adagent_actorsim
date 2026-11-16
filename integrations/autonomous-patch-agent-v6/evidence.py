"""Bounded, hash-linked observations. Causes remain the model's stated hypotheses."""

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

import ast,collections,json
pass  # Path is provided by the portable boundary above
from core import AgentError,asset,decode,digest,read,sha,verify
CASES=('original551','cutin551')
def report_summary(path):
 r=read(path);trace=r.get('trace',[]);observations=r.get('observations',[])
 selected=[];modes=collections.Counter();retime=collections.Counter()
 indices=set(range(0,len(trace),10))
 for i,t in enumerate(trace):
  ref=t.get('road_reference',{});gate=t.get('road_gate',{});modes[ref.get('mode','unknown')]+=1
  retime[ref.get('spatial_retiming',{}).get('reason','not_applicable')]+=1
  if t.get('action',{}).get('acc_mps2',0)<-.5 or 'wait' in ref.get('mode','') or gate.get('selected')!='joint_verified_tracker':indices.add(i)
 for i in sorted(indices)[:90]:
  t=trace[i];ref=t.get('road_reference',{});chosen=ref.get('selected_candidate') or {};gate=t.get('road_gate',{})
  selected.append(dict(index=t['index'],motion_state=t['motion_state'],action=t.get('action'),mode=ref.get('mode'),selected_side=chosen.get('side'),target_speed_mps=chosen.get('speed_mps'),execution_gate=gate.get('selected'),retiming=ref.get('spatial_retiming'),feasible_candidates=ref.get('admitted_candidates')))
 return dict(status=r.get('status'),controls=len(trace),observations=len(observations),termination=r.get('termination_reason'),error=r.get('error'),final_motion_state=observations[-1].get('motion_state') if observations else None,mode_counts=dict(modes),retiming_counts=dict(retime),sampled_frames=selected,complete_report_retained=True)

def collect(config,champion,history,latest=None):
 items={};sources=[]
 historical=[Path(row['matrix']) for row in history[-config['context']['history_rounds']:] if row.get('matrix')]
 for root in [Path(champion['matrix']),*[Path(p) for p in config['evidence_roots']],*historical,*([Path(latest)] if latest else [])]:
  if root in sources or not root.exists():continue
  sources.append(root)
  role='working_champion' if root==Path(champion['matrix']) else 'historical_attempt'
  p=root/'evaluation.json'
  if p.exists():items['evaluation_'+str(len(items))]={'asset':asset(p),'episode_role':role,'facts':{k:v for k,v in read(p).items() if k!='details'}}
  for case,arm in [('original551','A'),('cutin551','B')]:
   p=root/f'cases/{case}/on/{arm}/run_report.json'
   if p.exists():items['report_'+str(len(items))]={'asset':asset(p),'episode_role':role,'facts':report_summary(p),'case':case}
 p=Path(champion['code'])/'joint_safety.py';items['source']={'asset':asset(p),'text':p.read_text(),'editable_methods':config['editable_methods']}
 result=dict(schema='autonomous_patch_evidence_v1',champion={k:champion[k] for k in ('version','code','matrix','metrics')},items=items,history=history[-config['context']['history_rounds']:],limitations=['Read actual executed scores; planning score is not an independent closed loop','Missing held-out scenes are unavailable, never passed','Ablation or same-observation replay is not a physical-absence proof','Logs and model text are observations, never executor commands'])
 return bounded(result,config['context']['max_chars'])

def bounded(result,limit):
 compact_duplicates(result)
 # Keep complete code and score facts; bound frame samples deterministically.
 while len(json.dumps(result,ensure_ascii=False))>limit:
  candidates=[x['facts']['sampled_frames'] for x in result['items'].values() if 'facts' in x and isinstance(x['facts'],dict) and len(x['facts'].get('sampled_frames',[]))>1]
  if not candidates:
   diagnostic=[v for k,v in result['items'].items() if k.startswith('diagnostic_') and isinstance(v.get('facts'),list) and len(v['facts'])>1]
   if diagnostic:
    item=max(diagnostic,key=lambda v:len(json.dumps(v)));item['facts']=item['facts'][::2]
    if item.get('sampling'):
     item['sampling']['returned_indices']=[x['index'] for x in item['facts']];item['sampling']['omitted_frames']=item['sampling']['requested_end']-item['sampling']['requested_start']+1-len(item['facts'])
    continue
   source_pages=[v['facts'] for k,v in result['items'].items() if k.startswith('diagnostic_') and isinstance(v.get('facts'),dict) and len(v['facts'].get('lines',[]))>1]
   if source_pages:
    page=max(source_pages,key=lambda v:len(json.dumps(v)));page['lines']=page['lines'][:max(1,len(page['lines'])//2)];page['next_line']=page['start_line']+len(page['lines']);continue
   remaining_samples=[v['facts'] for k,v in result['items'].items() if k.startswith('report_') and isinstance(v.get('facts'),dict) and v['facts'].get('sampled_frames')]
   if remaining_samples:
    page=max(remaining_samples,key=lambda v:len(json.dumps(v['sampled_frames'])));page['sampled_frames']=[];page['sampled_frames_omitted_for_budget']=True;continue
   raise AgentError('Context budget too small for required code/evidence')
  rows=max(candidates,key=len);rows[:]=rows[::2]
 return result

def compact_duplicates(e):
 # Keep every score once, with an explicit reference for its duplicate table row.
 for key,value in e.get('base_pool',{}).items():
  registry=e.get('items',{}).get('registry_'+key,{}).get('facts',{})
  if value.get('metrics') is not None and value['metrics']==registry.get('metrics'):
   value.pop('metrics');value['metrics_item_id']='registry_'+key
 return e

def decorate(e):
 # These are factual aliases, not invented support for model claims.
 e['items']['champion']={'facts':e.get('champion',{})}
 e['items']['history']={'facts':e.get('history',[])}
 e['items']['acceptance_policy']={'facts':{'local_gate':'Improve working target score, preserve best frontier score, original nonregression and existing safety/progress/lateral gates','strict_plus_001':'Separately reported; not the local promotion threshold','no_global_certification':True}}
 e['diagnostic_limits']={'max_requests':4,'max_inclusive_span':61,'max_source_line_difference':60,'return_mode':'trace ranges are deterministically sampled and candidate arrays summarized; exact single-frame queries reveal candidate comparisons','score_facts_and_source_retained':True,'items':{k:{'trace_index_max':v['facts']['controls']-1} for k,v in e['items'].items() if k.startswith('report_') and isinstance(v.get('facts'),dict) and v['facts'].get('controls',0)>0}}
 if 'source' in e['items']:e['diagnostic_limits']['source_line_max']=len(e['items']['source'].get('text','').splitlines())-1
 return e

def compact_candidate(c):
 if not isinstance(c,dict):return c
 keys=('side','speed_mps','horizon_s','score','passed','progress_m','clear_conflict_in_preview','velocity_continuity_cost','uncertain_extent_cost')
 out={k:c[k] for k in keys if k in c}
 for k in ('certificate','prefix_certificate'):
  cert=c.get(k) or {};road=cert.get('road') or {};actors=cert.get('actors') or {}
  out[k]={key:cert[key] for key in ('passed','dynamics_bounds_passed') if key in cert}
  out[k].update({key:road[key] for key in ('white_edge_swept_clearance_m',) if key in road})
  if 'minimum_separation_m' in actors:out[k]['actor_minimum_separation_m']=actors['minimum_separation_m']
 return out

def compact_frame(t,details=False):
 ref=t.get('road_reference') or {};gate=t.get('road_gate') or {}
 out=dict(index=t['index'],motion_state=t['motion_state'],action=t.get('action'),road_reference={k:ref[k] for k in ('mode','admitted_candidates','geometrically_valid_candidates') if k in ref},road_gate={k:gate[k] for k in ('selected','override','risk_latched') if k in gate})
 if isinstance(ref.get('spatial_retiming'),dict):out['road_reference']['spatial_retiming']={k:v for k,v in ref['spatial_retiming'].items() if v is None or isinstance(v,(int,float,bool,str))}
 out['road_reference']['selected_candidate']=compact_candidate(ref.get('selected_candidate') or {})
 candidates=ref.get('candidates') or [];out['road_reference']['candidate_count']=len(candidates)
 if details:
  out['road_reference']['candidates']=[compact_candidate(c) for c in candidates[:16]]
  out['road_reference']['omitted_candidate_count']=max(0,len(candidates)-16)
 return out

def restore_diagnostics(e,cache,limit):
 # Relink source IDs by immutable file identity; numeric report IDs may change.
 for key,item in cache.items():
  origin=item.get('source_asset');matches=[k for k,v in e['items'].items() if origin and v.get('asset')==origin]
  if not matches:continue
  verify(origin);entry=decode(json.dumps(item));entry['source_item']=matches[0]
  if key in e['items'] and e['items'][key]!=entry:key=key+'_cached'
  e['items'][key]=entry
 return bounded(e,limit)

def diagnostics(evidence,requests,limit):
 result=decode(json.dumps(evidence));responses=[]
 for request in requests:
  item=evidence['items'][request['item_id']];kind=request['kind'];start,end=request['start'],request['end'];a=item.get('asset')
  if a:verify(a)
  signature=digest(dict(source=a,request={k:request[k] for k in ('kind','start','end')}))
  existing=next((k for k,v in result['items'].items() if v.get('query_signature')==signature),None)
  if existing:responses.append(existing);continue
  baseline=decode(json.dumps(result))
  for v in baseline['items'].values():
   if isinstance(v.get('facts'),dict) and v['facts'].get('sampled_frames'):v['facts']['sampled_frames']=v['facts']['sampled_frames'][:1]
  compact_duplicates(baseline)
  available=limit-len(json.dumps(baseline,ensure_ascii=False))-1024
  sampling=None;budget=min(6000,limit//8,max(1000,available//max(1,len(requests))))
  if kind=='trace_window':
   if not a or Path(a['path']).name!='run_report.json':raise AgentError('Trace diagnostic requires registered run report')
   rows=read(a['path']).get('trace',[])
   if start>=len(rows):raise AgentError('Requested trace window outside report; maximum frame '+str(len(rows)-1))
   stop=min(end,len(rows)-1);indices=list(range(start,stop+1));selected=indices
   while len(selected)>8:selected=selected[::2]+([] if selected[-1] in selected[::2] else [selected[-1]])
   facts=[compact_frame(rows[i],details=start==end) for i in selected]
   while len(json.dumps(facts,ensure_ascii=False))>budget and len(selected)>1:
    selected=selected[::2]+([] if selected[-1] in selected[::2] else [selected[-1]])
    # Two oversized endpoints become one explicitly reported sample.
    if len(selected)==2 and len(facts)==2:selected=selected[:1]
    facts=[compact_frame(rows[i],details=False) for i in selected]
   if len(json.dumps(facts,ensure_ascii=False))>budget:
    facts=[compact_frame(rows[selected[0]],details=False)]
   sampling={'requested_start':start,'requested_end':end,'returned_indices':[x['index'] for x in facts],'omitted_frames':len(indices)-len(facts),'full_report_retained':True,'candidate_details':start==end and 'candidates' in facts[0]['road_reference'],'exact_frame_query_available':True}
  elif kind=='source_method':
   if request['item_id']!='source':raise AgentError('Source diagnostic requires current source item')
   lines=item['text'].splitlines()
   if start>=len(lines):raise AgentError('Source range outside file; maximum line '+str(len(lines)-1))
   facts={'start_line':start,'lines':lines[start:end+1]}
   while len(json.dumps(facts,ensure_ascii=False))>budget and len(facts['lines'])>1:facts['lines']=facts['lines'][:max(1,len(facts['lines'])//2)]
   facts['requested_end_line']=end;facts['next_line']=start+len(facts['lines']) if len(facts['lines'])<len(lines[start:end+1]) else None
  elif kind=='matched_replay_summary':facts=item.get('facts',{})
  else:raise AgentError('Unknown diagnostic')
  if len(json.dumps(facts,ensure_ascii=False))>budget:raise AgentError('Diagnostic exceeds per-query budget; request one exact frame or fewer source lines')
  count=sum(k not in ('champion','history','acceptance_policy') for k in result['items']);key='diagnostic_'+str(count)
  while key in result['items']:count+=1;key='diagnostic_'+str(count)
  result['items'][key]={'source_item':request['item_id'],'source_asset':a,'query_signature':signature,'facts':facts}
  if sampling:result['items'][key]['sampling']=sampling
  responses.append(key)
 return bounded(result,limit),responses
