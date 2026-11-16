"""Host-owned one-use validation pools, released only after durable retirement."""
from common_flow import *

def configuration(m,taskfile):
 task=read(taskfile);folder=OUT/'private_validation';folder.mkdir(parents=True,exist_ok=True);folder.chmod(0o700)
 catalog=folder/'pool_catalog.json'
 if not catalog.exists():
  scenes=[s for s in (172,552,621) if s!=task['scene']]
  parent=read(m['policies']['global'])
  cases=[dict(id='registered_'+str(s),scene=s,ready=False,baseline=None,adapter=None,unavailable_reason='Matched scene runtime/map/native-score baseline and counterexample training adapter require admission') for s in scenes]
  pool=folder/'pool_001.json';write(pool,dict(schema='one_use_cross_scene_validation_pool_v1',baseline_controller_pins=parent['code_pins'],cases=cases))
  write(catalog,dict(schema='one_use_validation_pool_catalog_v1',pools=[asset(pool)],replacement_policy='New scenes only; consumed or training scenes never return to validation'))
 return dict(required=True,mode='counterexample_cross_scene',feedback_policy='retire_then_release',training_scene=task['scene'],pool_catalog=asset(catalog),global_parent=asset(P26/'global_patch.json'))
