"""A native episode consumes the existing campaign budget, never a fresh counter."""
import contextlib,fcntl,time,signal
from scene_contracts import *

def campaign():return Path(read(P26/'autonomous_edit_discovery_v3/handoff.json')['campaign'])
def readiness():
 root=campaign();state=read(root/'state.json');cfg=read(root/'config.json');left=cfg['budgets']['wall_seconds']-(time.time()-state['started_epoch'])
 return {'campaign':str(root),'remaining_wall_seconds':left,'GPU_attempts':state['usage'].get('gpu_attempts',0),'GPU_limit':cfg['budgets']['max_gpu_attempts'],'ready':left>0 and state['usage'].get('gpu_attempts',0)<cfg['budgets']['max_gpu_attempts']}
@contextlib.contextmanager
def reserve(folder):
 root=campaign()
 with (root/'coordinator.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);r=readiness()
  if not r['ready']:raise RuntimeError('Existing shared wall/GPU budget exhausted; explicit additive allocation required, no counter reset')
  state=read(root/'state.json')
  if state.get('active_round') or (state.get('generalization') or {}).get('current_job'):raise RuntimeError('Existing campaign still has an active round/job')
  initial=state['usage'].copy();state['usage']['gpu_attempts']=state['usage'].get('gpu_attempts',0)+1;state['usage']['gpu_dispatches']=state['usage'].get('gpu_dispatches',0)+1
  write(Path(folder)/'budget_receipt.json',{'campaign':str(root),'usage_before':initial,'usage_after':state['usage'],'started_epoch_preserved':state['started_epoch'],'reserved_epoch':time.time(),'budget_reset':False});write(root/'state.json',state)
  previous=signal.getsignal(signal.SIGALRM)
  def expired(*args):raise TimeoutError('Shared cumulative GPU stage wall allowance exhausted')
  signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,r['remaining_wall_seconds'])
  try:yield r['remaining_wall_seconds']
  finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)

def extend_window(seconds):
 """Explicit additive wall allowance only; counters and old starts survive."""
 if type(seconds) is not int or seconds<=0:raise ValueError('Positive explicit wall window required')
 root=campaign()
 with (root/'coordinator.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);state=read(root/'state.json');cfg=read(root/'config.json')
  if state.get('active_round'):raise RuntimeError('Cannot revise an active round budget')
  folder=root/'budget_extensions'/('cross_scene_'+str(time.time_ns()));folder.mkdir(parents=True)
  for name in ('config.json','host_freeze.json'):(folder/('previous_'+name)).write_bytes((root/name).read_bytes())
  old=cfg['budgets']['wall_seconds'];new=max(old,time.time()-state['started_epoch']+seconds)
  receipt={'explicit_window_seconds':seconds,'added_wall_seconds':new-old,'old_start_preserved':state['started_epoch'],'tokens_and_attempts_unchanged':state['usage'],'previous_config':asset(folder/'previous_config.json'),'previous_freeze':asset(folder/'previous_host_freeze.json'),'counter_reset':False}
  write(folder/'extension.json',receipt);cfg['budgets']['wall_seconds']=new;cfg.setdefault('budget_extension_history',[]).append(asset(folder/'extension.json'));write(root/'config.json',cfg)
  freeze=read(root/'host_freeze.json');freeze['config_sha256']=sha(root/'config.json');write(root/'host_freeze.json',freeze)
  # The discovery parent's older timer must not undo an explicit child renewal.
  missionfile=P26/'autonomous_edit_discovery_v3/mission.json';ledgerfile=missionfile.parent/'ledger.json'
  if missionfile.exists() and ledgerfile.exists():
   mission=read(missionfile);ledger=read(ledgerfile);(folder/'previous_discovery_mission.json').write_bytes(missionfile.read_bytes())
   started=ledger.get('started_epoch');elapsed=time.time()-started if started is not None else 0.
   mission['explicit_budget_extension']['wall_seconds']=max(mission['explicit_budget_extension']['wall_seconds'],elapsed+seconds);mission.setdefault('budget_extension_history',[]).append(asset(folder/'extension.json'));write(missionfile,mission)
  return receipt
