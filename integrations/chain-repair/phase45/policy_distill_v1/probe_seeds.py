import inspect,json,pickle,sys,time
import numpy as np
from paths import *
from run_cutin import configure
parser=__import__('argparse').ArgumentParser();parser.add_argument('--steps',nargs='+',type=int,default=[22,23]);args=parser.parse_args();driver=configure();opt=driver.optimized_controller
class Captured(Exception):pass
for index in args.steps:
 with (HERE/'validation/inputs'/('step_%03d.pkl'%index)).open('rb') as f:data=pickle.load(f)
 saved={}
 def capture(fun,x0,**kwargs):saved.update(inspect.currentframe().f_back.f_locals);raise Captured()
 actual=opt.minimize;opt.minimize=capture
 try:opt.optimize(data['reference'],data['occupancy'],data['info'],**data['kwargs'])
 except Captured:pass
 finally:opt.minimize=actual
 rows=[]
 for name,seed in saved['seeds']:
  if name!='linear_reachable_stop':seed=opt.bounded_seed(seed,saved['speed'],saved['steer'],saved['control_interval_s'],saved['vehicle'])
  states,_=saved['simulate'](seed,saved['speed'],saved['steer'],saved['offset']);minimum=float(saved['constraints'](seed.ravel()).min());rows.append(dict(seed=name,objective=float(saved['objective'](seed.ravel())),minimum=minimum,final_speed=float(states[-1,3])))
 print(index,json.dumps(sorted(rows,key=lambda r:(r['minimum']<-1e-5,r['objective'])),indent=2),flush=True)
