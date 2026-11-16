
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

import copy,json,sys,unittest
pass  # Path is provided by the portable boundary above
import numpy as np
LOCAL=Path(__file__).resolve().parents[1];sys.path[:0]=[str(LOCAL),str(LOCAL/'tests'),str(LOCAL.parent/'light-map551-v4'),str(LOCAL.parent/'light-cutin-diagnostic')]
from test_repair import V,mapping,info,Motion,intent
from evidence_packet import build,validate
from joint_safety import JointReference

def packet(box,inf,visible=True,support=10,refine_x=None):
 raw=np.asarray([box],np.float32);candidate=raw.copy();gravity=candidate.copy();gravity[:,2]+=gravity[:,5]/2;proposed=candidate.copy()
 if refine_x is not None:proposed[0,0]=refine_x
 arrays={'boxes':raw,'scores':np.array([.7],np.float32),'labels':np.array([0]),'candidate_boxes':candidate,'candidate_scores':np.array([.7],np.float32),'candidate_labels':np.array([0]),'geometry_candidate_boxes':proposed}
 origin={'decoded_gravity_boxes':gravity.tolist(),'selected_track_ids':[11],'decode_input_has_special_ego':False}
 evidence={'timestamp':inf['timestamp'],'scene_token':inf['scene_token'],'queries':[{'index':0,'neural_id':11,'visible_center':visible,'support_pixels':support,'reliable_pixels':20 if support else 0,'protected_observation':True}]}
 refinement={'candidates':[{'candidate_index':0,'accepted':refine_x is not None,'original_box':candidate[0].tolist(),'box':proposed[0].tolist()}]}
 return build(arrays,origin,evidence,refinement,inf,V)

class ObservationContractTests(unittest.TestCase):
 def test_live_mode_rejects_missing_packet(self):
  r=JointReference(V,mapping());r.require_box_observation_contract=True
  with self.assertRaisesRegex(ValueError,'Live box observation contract missing'):r.update_tracks({},info(Motion(.1,2.7,0.),0))
 def test_raw_arrays_preserved_and_refinement_is_separate(self):
  inf=info(Motion(.1,2.7,0.),0);p=packet([0,8,0,2,4.5,1.5,0,0,0],inf,refine_x=2.)
  self.assertEqual(p['boxes'][0,0],0);self.assertEqual(p['validated_memory_boxes'][0,0],2.);self.assertTrue(p['validated_memory_mask'][0]);self.assertFalse(p['box_observation_contract']['raw_neural_modified'])
 def test_observed_elevated_obstacle_is_not_erased_by_current_ego_height(self):
  inf=info(Motion(.1,2.7,0.),0);p=packet([0,8,3,2,4.5,1.5,0,0,0],inf)
  self.assertTrue(p['validated_memory_mask'][0]);self.assertFalse(p['box_observation_contract']['rows'][0]['vertical_overlap_possible'])
 def test_contract_tampering_or_scene_swap_rejected(self):
  inf=info(Motion(.1,2.7,0.),0);p=packet([0,8,0,2,4.5,1.5,0,0,0],inf)
  with self.assertRaises(ValueError):validate(p,dict(inf,scene_token='other'))
  p['boxes'][0,0]=1
  with self.assertRaises(ValueError):validate(p,inf)
 def test_protected_history_does_not_equal_current_observation(self):
  r=JointReference(V,mapping());m=Motion(.1,2.7,0.)
  for i in range(4):
   inf=info(m,i*.1);a=packet([0,8,3,2,4.5,1.5,0,0,0],inf,visible=False,support=0);r.observed({},inf,a);r.update_tracks(a,inf)
  self.assertEqual(len(r.actor_boxes()[0]),0);self.assertEqual(len(r._weak_geometry),1)
  states=np.zeros((10,5));states[:,1]=8;self.assertGreater(r.uncertainty_cost(states,np.empty((0,9)),[]),0)
 def test_qualified_refinement_reaches_actual_memory(self):
  r=JointReference(V,mapping());m=Motion(.1,2.7,0.)
  for i in range(4):
   inf=info(m,i*.1);a=packet([0,8,0,2,4.5,1.5,0,0,0],inf,refine_x=2.);r.observed({},inf,a);r.update_tracks(a,inf)
  self.assertEqual(len(r.actor_boxes()[0]),1);self.assertAlmostEqual(r.tracks[0]['box'][0],2.)
 def test_supported_obstacle_survives_occlusion_without_renewing_geometry(self):
  r=JointReference(V,mapping());m=Motion(.1,2.7,0.)
  for i in range(4):
   inf=info(m,i*.1);a=packet([0,8,0,2,4.5,1.5,0,0,0],inf);r.observed({},inf,a);r.update_tracks(a,inf)
  r.tracks[0]['latched']=True;before=copy.deepcopy(r.tracks[0]);inf=info(m,2.)
  a=packet([.1,8,3,2,4.5,1.5,0,0,0],inf,visible=False,support=0);r.observed({},inf,a);r.update_tracks(a,inf)
  self.assertEqual(r.tracks[0]['last'],before['last']);np.testing.assert_array_equal(r.tracks[0]['box'],before['box']);self.assertEqual(len(r.actor_boxes()[0]),1)
  state=np.array([[0,8,0,0,0]],float);self.assertFalse(r.actor_certificate(state,r.actor_boxes()[0])['passed'])
if __name__=='__main__':unittest.main()
