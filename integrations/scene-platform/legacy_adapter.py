"""Delegate v1 to its existing validators/executor, preserving measured semantics."""
import subprocess,sys
from platform_io import read,write,asset

def execute_legacy(stage,task,profile,job,event):
    command=[sys.executable,'-u','-B',profile['entry'],'--task',str(job/'task.json')]
    if stage=='check':command.append('--check-only')
    else:command.extend(['--output',str(job/'legacy')])
    event('legacy_scene_task_adapter','checking' if stage=='check' else 'started',dict(schema='scene_task_v1'))
    with (job/('legacy_'+stage+'.log')).open('w') as log:
        code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
    if stage=='check':
        if code:raise RuntimeError('Existing v1 validator rejected task; inspect legacy_check.log')
        import torch
        write(job/'preflight.json',dict(verified=True,cuda_available=torch.cuda.is_available(),
            execution_performed=False,control_executed=False,authoritative_existing_preflight=asset(job/'legacy_check.log')))
        return
    path=job/'legacy/task_result.json'
    if not path.exists():raise RuntimeError('Legacy worker produced no task result; inspect legacy_run.log')
    result=read(path)
    # Preserve schema v1 and execution flags. Never label reference tracking as UniAD.
    result['platform_dispatch']=dict(adapter='legacy_scene_task_v1',LLM_invoked=False,uniad_closed_loop=False,
        original_result=asset(path),exit_code=code)
    if code:result['status']='failed'
    write(job/'task_result.json',result);event('legacy_scene_task_adapter','finished',dict(status=result['status']))
