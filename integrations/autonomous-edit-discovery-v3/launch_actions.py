"""GPU-terminal bootstrap, explicit additive allocation with no counter reset."""
import argparse,fcntl,json,math,shutil,subprocess,sys,time
from common_flow import *
from migration import OLD
sys.path.insert(0,str(AGENT))
from provider import HTTPProvider

def additions(previous,ledger,wall_seconds,gpu_attempts,now):
 started=ledger.get('started_epoch');elapsed=max(0.,now-started) if started is not None else 0.
 return dict(add_wall_seconds=max(0,math.ceil(elapsed+wall_seconds-previous['explicit_budget_extension']['wall_seconds'])),add_gpu_attempts=gpu_attempts)

def record_launch(status,**extra):
 value=dict(stage=status,updated_epoch=time.time(),**extra);write(OUT/'startup.json',value);print(json.dumps(value),flush=True)

def extend_window(seconds):
 if type(seconds) is not int or seconds<=0:raise ValueError('Positive explicit run window required')
 if (OUT/'handoff.json').exists():raise ValueError('Patch campaign has a frozen budget; cannot extend only its discovery parent')
 with (OUT/'worker.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (OUT/'owner.json').exists():
   sys.path.insert(0,str(DISCOVERY));from service import alive
   if alive(read(OUT/'owner.json')):raise ValueError('Coordinator is already running; resume without a new window')
  m=read(OUT/'mission.json');ledger=read(OUT/'ledger.json');delta=additions(m,ledger,seconds,0,time.time())['add_wall_seconds']
  event=OUT/'budget_extensions'/('window_'+str(time.time_ns()));event.mkdir(parents=True)
  (event/'previous_mission.json').write_bytes((OUT/'mission.json').read_bytes())
  receipt=dict(explicit_window_seconds=seconds,added_wall_seconds=delta,added_GPU_attempts=0,added_API_requests=0,previous_mission=asset(event/'previous_mission.json'),timer_reset=False,counters_reset=False)
  write(event/'extension.json',receipt);m['explicit_budget_extension']['wall_seconds']+=delta;m.setdefault('budget_extension_history',[]).append(asset(event/'extension.json'));write(OUT/'mission.json',m)
  print(json.dumps(receipt),flush=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--wall-seconds',type=int,default=14400);parser.add_argument('--gpu-attempts',type=int,default=12);parser.add_argument('--new-window-seconds',type=int);args=parser.parse_args()
 if args.new_window_seconds is not None and args.new_window_seconds<=0:raise SystemExit('Positive explicit run window required')
 if args.wall_seconds<=0 or args.gpu_attempts<=0:raise SystemExit('Positive run allocation required')
 missing=HTTPProvider(read(AGENT/'config.example.json')['api']).readiness()
 if missing:record_launch('waiting_api_launch',missing_api_variables=missing,coordinator_started=False);return 2
 if not shutil.which('nvidia-smi'):record_launch('GPU_terminal_required',coordinator_started=False);return 2
 subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],check=True)
 if args.new_window_seconds is not None:
  if (OUT/'mission.json').exists():extend_window(args.new_window_seconds)
  else:args.wall_seconds=args.new_window_seconds
 command=['bash',str(AGENT/'llm_tasks.sh'),'resume']
 if not (OUT/'mission.json').exists() and not (OUT/'launch_options.json').exists():
  allocation=additions(read(OLD/'mission.json'),read(OLD/'ledger.json'),args.wall_seconds,args.gpu_attempts,time.time())
  command+=['--add-wall-seconds',str(allocation['add_wall_seconds']),'--add-gpu-attempts',str(allocation['add_gpu_attempts'])]
  print(json.dumps(dict(new_run_wall_window_seconds=args.wall_seconds,explicit_additive_allocation=allocation,prior_tokens_and_attempts_preserved=True)),flush=True)
 else:print('Resuming stored allocation; prior tokens and GPU counters preserved',flush=True)
 record_launch('launch_requested',coordinator_started=False)
 result=subprocess.run(command,cwd=ROOT)
 record_launch('launch_dispatched' if result.returncode==0 else 'launch_failed',exit_code=result.returncode,coordinator_started='check owner.json and live process')
 return result.returncode
if __name__=='__main__':
 try:result=main()
 except Exception as exc:
  record_launch('launch_failed',error_type=type(exc).__name__,error=str(exc),coordinator_started=False);raise
 raise SystemExit(result)
