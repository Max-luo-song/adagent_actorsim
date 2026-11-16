import argparse,fcntl,subprocess,sys,os,traceback,time,datetime,shutil
from common import *

def pipeline(device):
 if device=='cuda':(OUT/'CPU_YIELD_TO_GPU').touch()
 with (OUT/'pipeline.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if device=='cpu' and (OUT/'CPU_YIELD_TO_GPU').exists():print('CPU yielded to GPU request',flush=True);return
  def status(stage,**kw):write(OUT/'progress.json',{'stage':stage,'device':device,'pid':os.getpid(),'updated_epoch':time.time(),'patch_validation_started':False,**kw})
  def run(script,*args,logname):
   with (OUT/logname).open('a') as log:
    result=subprocess.run([sys.executable,'-u','-B',str(LOCAL/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','MPLCONFIGDIR':str(OUT/'matplotlib_cache')})
   if result.returncode==75:status('yielded_for_gpu',scene=current_scene);return False
   if result.returncode:raise RuntimeError('Static stage failed '+script+'; inspect '+str(OUT/logname))
   return True
  current_scene=None
  try:
   status('offline_tests');run('tests.py',logname='offline_tests.log')
   for scene in (552,3,31,233):
    current_scene=scene;folder=ASSETS/str(scene)/'semantic19'
    if not (folder/'semantic_manifest.json').exists():
     status('semantic_export',scene=scene)
     progress=folder/'progress.json'
     if progress.exists() and read(progress).get('device')!=device:
      archived=folder.with_name(folder.name+'_partial_'+str(time.time_ns()));folder.rename(archived);write(OUT/('device_switch_'+str(time.time_ns())+'.json'),{'scene':scene,'archived_partial':str(archived),'new_device':device,'reason':'different inference host contract; partial original artifacts retained'})
     if not run('semantic_worker.py','--scene',scene,'--device',device,logname='semantic_scene%d.log'%scene):return
    status('map_build',scene=scene);run('map_job.py','--scene',scene,logname='map_scene%d.log'%scene)
    status('full_footprint_scan',scene=scene);run('scan.py','--scene',scene,logname='scan_scene%d.log'%scene)
   status('pool_draft');run('finish.py',logname='finish.log');status('completed_static_only')
  except BaseException as exc:
   status('static_infrastructure_error',scene=current_scene,error=repr(exc));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args();pipeline(a.device)
