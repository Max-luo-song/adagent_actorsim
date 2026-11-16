"""Provider-neutral Agent adapter; transport is a callable returning JSON.

The adapter never invokes an LLM, retries an edit, or executes shell content.
A future model only needs to select one of the exported tools and supply args.
"""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

import copy,json,re,uuid
from task_contracts import TASK_SCHEMA


def tool_specs():
    return [dict(name='get_scene_capabilities',description='Query supported scene tasks, weather and driver adapters before planning.',parameters={'type':'object','properties':{},'additionalProperties':False}),
        dict(name='run_scene_task',description='Execute one declarative scene task. Returns observations, artifacts, actual execution evidence and current scene context.',parameters={'type':'object','required':['task'],'additionalProperties':False,'properties':{'task':TASK_SCHEMA}})]


class JsonlTransport:
    """Connect to an existing scene-task-stdio process via text pipes."""
    def __init__(self,reader,writer):self.reader=reader;self.writer=writer
    def __call__(self,request):
        self.writer.write(json.dumps(request,allow_nan=False)+'\n');self.writer.flush()
        line=self.reader.readline()
        if not line:raise ConnectionError('scene service closed before a response')
        def reject(value):raise ValueError('nonfinite service response: '+value)
        return _hugsim_expand(json.loads(line,parse_constant=reject))


class SceneAgentClient:
    def __init__(self,transport):
        if not callable(transport):raise TypeError('transport must be callable(request)->response')
        self.transport=transport;self.context={};self.sent={}
    def call(self,action,args=None,*,request_id=None):
        args=_hugsim_expand(json.loads(json.dumps({} if args is None else args,allow_nan=False)));identity=request_id or 'agent_'+uuid.uuid4().hex
        if not isinstance(args,dict):raise ValueError('tool arguments must be an object')
        if not isinstance(identity,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}',identity):raise ValueError('invalid request_id')
        fingerprint=json.dumps(dict(action=action,args=args),sort_keys=True,separators=(',',':'))
        if identity in self.sent:
            old,request=self.sent[identity]
            if old!=fingerprint:raise ValueError('request_id already used for different arguments')
        else:
            actual=copy.deepcopy(args)
            versioned=action in ('edit','validate_edit','preview','observe','reset','validate_ego_trajectory','preview_ego_trajectory','validate_ego_execution','execute_ego_trajectory','preview_actor_trajectory') or (action in ('load_scene','run_scene_task') and self.context.get('generation') is not None)
            if versioned:
                if self.context.get('generation') is not None:
                    actual.setdefault('expected_generation',self.context['generation']);actual.setdefault('expected_revision',self.context['edit_revision'])
            request=dict(schema='agent_request_v1',request_id=identity,action=action,args=actual)
            self.sent[identity]=(fingerprint,copy.deepcopy(request))
        response=self.transport(copy.deepcopy(request))
        if not isinstance(response,dict) or response.get('schema')!='agent_response_v1' or response.get('request_id')!=identity:raise ValueError('invalid response from transport')
        self.context=copy.deepcopy(response['context'])
        return response
    def run_task(self,task,*,request_id=None):
        if not self.context:self.call('get_capabilities')
        return self.call('run_scene_task',{'task':task},request_id=request_id)
    def dispatch(self,name,arguments,*,request_id=None):
        if name=='get_scene_capabilities':
            if arguments!={}:raise ValueError('capability query takes no arguments')
            return self.call('get_capabilities',request_id=request_id)
        if name=='run_scene_task':
            if not isinstance(arguments,dict) or set(arguments)!={'task'}:raise ValueError('run_scene_task requires task only')
            return self.run_task(arguments['task'],request_id=request_id)
        raise ValueError('unknown scene agent tool')
