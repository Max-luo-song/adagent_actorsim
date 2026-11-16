"""Model-authored scenario parameters. Host owns compilation, trials and acceptance."""

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

import copy,hashlib,json,math,sys,time
pass  # Path is provided by the portable boundary above
from common_flow import AGENT,read,write,asset,verify
sys.path.insert(0,str(AGENT))
from core import AgentError,Deferred,BudgetExceeded,digest,decode
from provider import HTTPProvider,CompletedResponse,TransientAPI

PARAMETERS=('lane_change_duration_s','actor_deceleration_duration_s','target_lane_center_source_y_m')
PROGRAM='native551_car3_merge_brake'
MAX_EDITS=4


def schema(context_hash):
    number=lambda lo,hi:dict(type='number',minimum=lo,maximum=hi,multipleOf=.1)
    return dict(type='object',additionalProperties=False,required=['schema','context_sha256','decision','edits','note'],properties=dict(
        schema=dict(type='string',enum=['scene_edit_proposals_v1']),context_sha256=dict(type='string',enum=[context_hash]),
        decision=dict(type='string',enum=['propose','wait']),note=dict(type='string',maxLength=2000),
        edits=dict(type='array',maxItems=MAX_EDITS,items=dict(type='object',additionalProperties=False,
            required=['scene','program','parameters','hypothesis','expected_legal_response'],properties=dict(
                scene=dict(type='integer',enum=[551]),program=dict(type='string',enum=[PROGRAM]),
                parameters=dict(type='object',additionalProperties=False,required=list(PARAMETERS),properties=dict(
                    lane_change_duration_s=number(2.,6.),actor_deceleration_duration_s=number(2.,6.),target_lane_center_source_y_m=number(-1.,1.))),
                hypothesis=dict(type='string',maxLength=2000),expected_legal_response=dict(type='string',maxLength=2000))))))


def signature(parameters):
    # Normalize numeric representation only; never silently round an edit.
    values={k:(0. if float(parameters[k])==0 else float(parameters[k])) for k in PARAMETERS}
    return digest(dict(scene=551,program=PROGRAM,parameters=values))


def validate(value,context_hash,known):
    if not isinstance(value,dict) or set(value)!={'schema','context_sha256','decision','edits','note'}:raise AgentError('Exactly one bounded scene-edit proposal object is required')
    if value['schema']!='scene_edit_proposals_v1' or value['context_sha256']!=context_hash:raise AgentError('Proposal is not bound to this frozen scene context')
    if value['decision'] not in ('propose','wait') or not isinstance(value['note'],str) or len(value['note'])>2000:raise AgentError('Invalid scenario-design decision/note')
    edits=value['edits']
    if not isinstance(edits,list) or len(edits)>MAX_EDITS or (value['decision']=='propose' and not edits) or (value['decision']=='wait' and edits):raise AgentError('Propose1..4 edits or wait with no edits')
    seen=set(known);accepted=[]
    for edit in edits:
        if not isinstance(edit,dict) or set(edit)!={'scene','program','parameters','hypothesis','expected_legal_response'}:raise AgentError('Unknown edit field; code, commands and arbitrary paths are not supported')
        if type(edit['scene']) is not int or edit['scene']!=551 or edit['program']!=PROGRAM:raise AgentError('Scene or actor program is not admitted by the registered backend')
        for k in ('hypothesis','expected_legal_response'):
            if not isinstance(edit[k],str) or not edit[k].strip() or len(edit[k])>2000:raise AgentError('Bounded challenge hypothesis and legal response explanation required')
        p=edit['parameters']
        if not isinstance(p,dict) or set(p)!=set(PARAMETERS):raise AgentError('Only registered trajectory parameters may change')
        clean={}
        for k in PARAMETERS:
            v=p[k];lo,hi=(-1.,1.) if k==PARAMETERS[2] else (2.,6.)
            if type(v) not in (int,float) or not math.isfinite(v) or not lo<=v<=hi:raise AgentError('Nonfinite, boolean or out-of-range edit parameter: '+k)
            if abs(v*10-round(v*10))>1e-6:raise AgentError('Use declared0.1s/0.1m edit increments, not tiny perturbations')
            clean[k]=0. if v==0 else float(v)
        key=signature(clean)
        if key in seen:raise AgentError('Duplicate already proposed/evaluated edit; propose a genuinely new configuration')
        seen.add(key);accepted.append(dict(edit,parameters=clean))
    return accepted


class Designer:
    def __init__(self,folder,config,base_usage,provider=None,wall_remaining=None):
        self.folder=Path(folder);self.config=copy.deepcopy(config);self.base=copy.deepcopy(base_usage)
        self.provider=provider or HTTPProvider(self.config['api']);self.wall_remaining=wall_remaining or (lambda:self.config['budgets']['wall_seconds'])
        self.folder.mkdir(parents=True,exist_ok=True);self.ledger_path=self.folder/'usage.json'
        if not self.ledger_path.exists():write(self.ledger_path,dict(api_requests=0,charged_tokens=0,reported_tokens=0,prompt_tokens=0,completion_tokens=0,events=[]))
    def usage(self):return read(self.ledger_path)
    def charge(self,number,request):
        u=self.usage();b=self.config['budgets']
        if self.base.get('api_requests',0)+u['api_requests']+1>b['max_api_requests']:raise BudgetExceeded('Scenario design and patch analysis share the API-call budget')
        if self.base.get('charged_tokens',0)+u['charged_tokens']+number>b['max_api_tokens']:raise BudgetExceeded('Scenario design and patch analysis share the token budget')
        self.wall_remaining()
        u['api_requests']+=1;u['charged_tokens']+=number;u['events'].append(dict(request=str(request),reserved_tokens=number,reconciled=False))
        write(self.ledger_path,u)
    def reconcile(self,meta,request):
        u=self.usage();event=next((x for x in reversed(u['events']) if x['request']==str(request)),None)
        if not event or event['reconciled']:return
        if meta.get('usage_known',True):
            actual=meta['usage_tokens'];u['charged_tokens']+=actual-event['reserved_tokens'];u['reported_tokens']+=actual
            for k in ('prompt_tokens','completion_tokens'):u[k]+=meta.get('usage',{}).get(k,0)
        event.update(reconciled=True,usage_known=meta.get('usage_known',True));write(self.ledger_path,u)
    def propose(self,context,batch=1,max_attempts=3):
        missing=self.provider.readiness()
        if missing:raise Deferred('Scene-design API configuration pending: '+', '.join(missing))
        h=digest(context);shape=schema(h);batch_dir=self.folder/('batch_%03d'%batch);batch_dir.mkdir(exist_ok=True)
        context_file=batch_dir/'context.json'
        if context_file.exists() and read(context_file)!=context:raise AgentError('Frozen scenario context changed on resume')
        write(context_file,context)
        accepted=batch_dir/'proposal.json'
        if accepted.exists():
            saved=read(accepted);verify(saved['context']);verify(saved['response']);validate(saved['proposal'],h,context['known_parameter_signatures']);return saved
        feedback=[]
        system=('You design challenging but legal simulator edits, not controller patches. Return only the requested JSON. '
                'The fixed global controller is v15. Use only admitted scenes, actor programs and parameters. '
                'Keep explanations concise. Vary meaningful merge/braking/offset conditions and avoid duplicates, inevitable collisions, '
                'code, shell commands, filesystem paths, score changes, controller-specific scene/track branches and claimed successes. '
                'A hypothesis is not a measured failure. Host physics/clearance/feasible-path audits and real closed loops decide. '
                'Do not inspect or optimize the held-out evaluation scenes. If no defensible new edit exists, choose wait.')
        for attempt in range(1,max_attempts+1):
            self.wall_remaining();record=batch_dir/('request_%03d.json'%attempt)
            # Recover a fully received model response before sending another request.
            cached=read(record) if record.exists() else None
            if cached and cached.get('metadata'):
                self.reconcile(cached['metadata'],record)
            if cached and 'content' in cached and cached.get('metadata',{}).get('finish_reason','stop')=='stop':
                try:value=decode(cached['content']) if isinstance(cached['content'],str) else cached['content']
                except (AgentError,ValueError) as exc:
                    feedback.append(dict(attempt=attempt,error='Cached incomplete/invalid JSON: '+str(exc)));write(batch_dir/'feedback.json',feedback);continue
                meta=cached['metadata']
            elif cached:
                feedback.append(dict(attempt=attempt,error='Previous incomplete response retained; no edit applied'));continue
            else:
                if any(e['request']==str(record) for e in self.usage()['events']):
                    write(record,dict(status='interrupted_unknown_usage',note='Reservation retained; next bounded attempt may proceed without replaying this request'))
                    feedback.append(dict(attempt=attempt,error='Prior request interrupted; unknown usage retained'));continue
                messages=[dict(role='system',content=system),dict(role='user',content=json.dumps(dict(scene_context=context,context_sha256=h,validation_feedback=feedback,required_response_schema=shape),ensure_ascii=False))]
                write(record.with_suffix('.request.json'),dict(messages=messages,context=asset(context_file),credentials_persisted=False))
                if isinstance(self.provider,HTTPProvider):self.provider.config=dict(self.config['api'],max_output_tokens=min(8192,self.config['api']['max_output_tokens']),timeout_seconds=min(self.config['api']['timeout_seconds'],self.wall_remaining()))
                try:value,meta=self.provider.complete(messages,shape,lambda n:self.charge(n,record),record)
                except CompletedResponse as exc:
                    self.reconcile(exc.metadata,record);feedback.append(dict(attempt=attempt,error=str(exc)));write(batch_dir/'feedback.json',feedback);continue
                except TransientAPI:
                    # Transport retry is operator-visible and bounded by the shared ledger.
                    raise
                self.reconcile(meta,record)
            try:edits=validate(value,h,context['known_parameter_signatures'])
            except (AgentError,ValueError,TypeError) as exc:
                feedback.append(dict(attempt=attempt,error=str(exc)));write(batch_dir/'feedback.json',feedback);continue
            proposal=dict(proposal=value,context=asset(context_file),response=asset(record),author=meta,real_model_response=bool(meta.get('external_API_called')) and not meta.get('mock_model',False),hypotheses_are_not_measured_failures=True)
            write(accepted,proposal);return proposal
        raise AgentError('Scene proposal attempts exhausted; no edit was applied')


def compile_entries(proposal,batch):
    if proposal.get('real_model_response') is not True:raise AgentError('Live discovery cannot consume a fixture/model-simulated proposal')
    verify(proposal['context']);verify(proposal['response']);record=read(proposal['response']['path'])
    if not record.get('metadata',{}).get('external_API_called') or record.get('metadata',{}).get('mock_model') or record.get('metadata',{}).get('finish_reason')!='stop':raise AgentError('Proposal lacks a real provider-response receipt')
    recorded=record.get('content');recorded=decode(recorded) if isinstance(recorded,str) else recorded
    if recorded!=proposal['proposal']:raise AgentError('Compiled proposal differs from provider response')
    value=proposal['proposal'];h=read(proposal['context']['path']);edits=validate(value,digest(h),h['known_parameter_signatures'])
    return [dict(id='llm_s551_b%03d_e%02d_%s'%(batch,i+1,signature(e['parameters'])[:8]),**e['parameters']) for i,e in enumerate(edits)]
