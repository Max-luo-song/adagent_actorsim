"""Check all saved current problems without re-solving unaffected starts."""
import inspect,json,pickle
import numpy as np
from paths import *
from run_cutin import configure
from admission import STEPS
class Captured(Exception):pass
def validate():
 driver=configure();opt=driver.optimized_controller;rows=[]
 from ranked_search import candidates
 for index in range(28):
  with (HERE/'validation/inputs'/('step_%03d.pkl'%index)).open('rb') as handle:data=pickle.load(handle)
  original=data['source_result'];saved={}
  if index in STEPS:rows.append(dict(step=index,scope='full_solve_benchmark_and_independent_geometry'));continue
  def capture(fun,x0,**kwargs):saved.update(inspect.currentframe().f_back.f_locals);raise Captured()
  actual=opt.minimize;opt.minimize=capture;result=None
  try:result=opt.optimize(data['reference'],data['occupancy'],data['info'],**data['kwargs'])
  except Captured:pass
  finally:opt.minimize=actual
  if not original.get('trials'):
   if saved or result is None or result.get('trials'):raise ValueError('Continuation unexpectedly requires solving: '+str(index))
   if not np.array_equal(result['selected']['controls'],original['selected']['controls']):raise ValueError('Continuation controls changed: '+str(index))
   rows.append(dict(step=index,scope='actual_current_continuation_recomputed',controls_bitwise_equal=True));continue
  if not saved or not saved['consensus']:raise ValueError('Missing primary schedule: '+str(index))
  lookup={row['seed']:row for row in original['trials']};visited=[]
  def forbidden(*args):raise ValueError('Previously-unexpanded problem unexpectedly expands: '+str(index))
  for name,seed,budget,stage in candidates(saved['consensus'],saved['seeds'],220,forbidden):
   if budget!=220 or stage!='consensus_primary':raise ValueError('Changed primary solve budget')
   visited.append(name);saved['consensus'].observe(lookup[name])
  if visited!=[row['seed'] for row in original['trials']]:raise ValueError('Primary schedule changed: '+str(index))
  rows.append(dict(step=index,scope='unchanged_primary_schedule_and_original_convergence_rows',primary_order=visited,iteration_budgets_unchanged=True))
 result=dict(passed=True,cases=len(rows),rows=rows,gpu_executed=False,scope='Six full solves independently checked elsewhere; actual continuation replay; unchanged primary scheduling proved with archived convergence rows. Not a new closed loop.')
 (HERE/'schedule_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,cases=len(rows),counts={kind:sum(r['scope']==kind for r in rows) for kind in {r['scope'] for r in rows}})),flush=True)
if __name__=='__main__':validate()
