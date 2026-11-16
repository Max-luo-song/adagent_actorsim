
# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

from bootstrap import *
import unittest,json,tempfile,textwrap,types,numpy as np
from unittest.mock import patch
from occupancy_protocol import unpack
class WorkerIOTests(unittest.TestCase):
 def test_actual_resident_export_block_completes_all_availability_states(self):
  source=(HERE/'protocol_branch_worker.py').read_text();start=source.index('                result = scope["_results"][0]');end=source.index('            except BaseException as exc:',start);code=compile(textwrap.dedent(source[start:end]),str(HERE/'protocol_branch_worker.py')+':export','exec')
  from closed172_common import atomic_json
  examples=[(None,0,'unavailable'),(np.zeros((1,5,1,200,200)),1,'valid_empty'),(np.ones((1,5,1,200,200)),1,'valid_nonempty'),(np.full((1,5,1,200,200),np.nan),1,'invalid')]
  with tempfile.TemporaryDirectory() as tmp:
   job=Path(tmp)
   for index,(occupancy,queries,status) in enumerate(examples):
    out=job/f'step_{index:03d}';out.mkdir();atomic_json(out/'occupancy_head_audit.json',dict(motion_query_count=queries))
    result=dict(occ=dict(seg_out=occupancy),planning=dict(result_planning=dict(sdc_traj=np.zeros((6,2)))),scene_reliability={})
    fake_torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:False))
    namespace=dict(scope={'_results':[result],'_elapsed':.1},output=out,array=np.asarray,atomic_json=atomic_json,np=np,json=json,core=types.SimpleNamespace(with_seg_head=True,seg_head=types.SimpleNamespace(pc_range=[-50,-50,-5,50,50,3])),JOB=job,index=index,timestamp=index*.1,torch=fake_torch,mark=lambda *args:None,current={},obs={})
    with patch('drivable_area.export_prediction',return_value={'drivable_mask':np.ones((200,200),dtype=np.uint8)}):exec(code,namespace)
    done=_hugsim_expand(json.loads((job/f'done_{index:03d}.json').read_text()));self.assertEqual(done['status'],'ok');self.assertEqual(namespace['index'],index+1)
    with np.load(done['perception']) as saved:array,contract=unpack(saved)
    self.assertEqual(contract['status'],status)
    if status in ('unavailable','invalid'):self.assertIsNone(array)
 def test_resident_wrapper_chain_is_connected_to_new_export(self):
  import worker as wrapper
  direct=wrapper.worker.serve;origin=wrapper.origin;global_worker=wrapper.global_worker
  old=(origin.worker,origin.original_serve,global_worker.original_serve)
  try:
   wrapper.install()
   self.assertIs(global_worker.original_serve,direct)
   self.assertIs(origin.worker,wrapper.worker)
   self.assertIs(origin.original_serve,wrapper.serve)
   self.assertIs(wrapper.worker.serve,origin.serve)
  finally:
   origin.worker,origin.original_serve,global_worker.original_serve=old;wrapper.worker.serve=direct
if __name__=='__main__':unittest.main(verbosity=2)
