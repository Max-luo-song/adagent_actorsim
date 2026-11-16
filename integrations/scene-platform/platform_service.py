"""Two portable tool calls. The caller can be a mock LLM or a future real agent."""

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

import sys,json
from platform_io import read,write,fingerprint,identity
from platform_contracts import tool_specs

class Service:
    def __init__(self,runner):self.runner=runner
    def dispatch(self,request):
        rid=request.get('request_id');identity(rid)
        if set(request)!={'request_id','tool','arguments'}:raise ValueError('Expected request_id, tool, arguments')
        tool=request['tool'];args=request['arguments']
        if tool=='get_scene_capabilities':
            if args!={}:raise ValueError('Capabilities takes no arguments')
            return dict(request_id=rid,ok=True,result=self.runner.catalog.capabilities(),replayed=False)
        if tool!='run_scene_task' or not isinstance(args,dict) or set(args)!={'task'}:raise ValueError('Unknown tool or arguments')
        reply=self.runner.run(args['task'],rid)
        return dict(request_id=rid,ok=True,**reply)

def stdio(runner):
    service=Service(runner)
    for line in sys.stdin:
        try:
            request=_hugsim_expand(json.loads(line));response=service.dispatch(request)
        except Exception as exc:response=dict(ok=False,error=str(exc))
        print(json.dumps(response,ensure_ascii=False,allow_nan=False),flush=True)

def simulate(goal,runner,output,prepare_only=False):
    """Deterministic mock LLM: query capabilities, then emit a bound task tool call."""
    pass  # Path is provided by the portable boundary above
    from platform_contracts import compile_task
    output=Path(output);service=Service(runner);calls=[];responses=[]
    rid='mock_'+fingerprint(goal)[:24]
    query=dict(request_id=rid+'_caps',tool='get_scene_capabilities',arguments={})
    calls.append(query);capabilities=service.dispatch(query);responses.append(capabilities)
    task=goal['task'];compile_task(task,runner.catalog)
    request=dict(request_id=rid,tool='run_scene_task',arguments=dict(task=task));calls.append(request)
    write(output/'tool_calls.json',calls);write(output/'tool_responses.json',responses)
    if prepare_only:
        job=runner.prepare(task,rid)
        result=dict(mode='prepared_only',LLM_invoked=False,run_tool_dispatched=False,controls_executed=False,job=str(job),
            capabilities_tool_dispatched=True,compiled_task=str(job/'compiled_task.json'))
    else:
        try:
            response=service.dispatch(request)
            result=dict(mode='actual_tools_dispatched',LLM_invoked=False,run_tool_dispatched=True,**response)
        except Exception as exc:
            response=dict(request_id=rid,ok=False,error=str(exc))
            state_path=runner.store/'jobs'/rid/'state.json'
            state=read(state_path)['state'] if state_path.exists() else 'unprepared'
            result=dict(mode='actual_tools_dispatched',LLM_invoked=False,run_tool_dispatched=True,
                request_id=rid,status='deferred' if state=='prepared' else 'failed',error=str(exc),
                controls_executed=False if state=='prepared' else None,job_state=state,
                job=str(runner.store/'jobs'/rid))
        responses.append(response);write(output/'tool_responses.json',responses)
    write(output/'simulation_receipt.json',result);return result
