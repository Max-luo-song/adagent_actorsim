
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

import json,sys,tempfile,types,unittest
pass  # Path is provided by the portable boundary above
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
LOCAL=Path(__file__).resolve().parents[1];sys.path.insert(0,str(LOCAL))
from hooks import install

class Head(nn.Module):
    def __init__(self):
        super().__init__();self.mlp_fuser=nn.Sequential(nn.Linear(768,8),nn.LayerNorm(8),nn.ReLU())
        self.pos_embed=nn.Embedding(1,8);self.with_adapter=True;self.bev_adapter=nn.Conv2d(8,8,1)
        self.attn_module=nn.TransformerDecoder(nn.TransformerDecoderLayer(8,2,16,dropout=0),1)
        self.reg_branch=nn.Linear(8,12)
    def forward_test(self,bev_embed,outs_motion):
        q=self.mlp_fuser(outs_motion['fuser']).max(1,keepdim=True)[0].permute(1,0,2)+self.pos_embed.weight[None]
        m=bev_embed+outs_motion['bev_pos'];m=m+self.bev_adapter(m);m=m.flatten(2).permute(2,0,1)
        return self.reg_branch(self.attn_module(q,m)).reshape(-1,6,2).cumsum(1)

class HookTests(unittest.TestCase):
    def test_history_comparisons_end_at_same_source_frame(self):
        from pipeline import cases
        rows=cases();streams={}
        for row in rows:
            if row['sequence']:streams.setdefault(row['sequence'],[]).append(row)
        self.assertEqual(len(rows),16)
        for name,stream in streams.items():
            self.assertEqual([r['reset'] for r in stream],[True,False,False])
            self.assertEqual(stream[-1]['frame'],20);self.assertEqual(stream[-1]['timestamp'],1.)
            self.assertTrue(all(r['rgb']=='source' and r['speed']=='recorded' for r in stream))
        self.assertEqual([r['frame'] for r in streams['history_01s']],[18,19,20])
        self.assertEqual([r['frame'] for r in streams['history_05s']],[10,15,20])

    def test_original_return_unchanged_and_no_reset_inside_history(self):
        torch.manual_seed(13);head=Head().eval();memory=torch.randn(1,8,2,2);outs=dict(fuser=torch.randn(1,2,768),bev_pos=torch.randn(1,8,2,2))
        class Core:
            def __init__(self):
                self.planning_head=head;self.motion_head=SimpleNamespace(forward_test=lambda outs_track:None);self.cached=True
                self.gate=SimpleNamespace(can_bus_mlp=True,use_can_bus=False,use_cams_embeds=False)
            def named_modules(self):return [('encoder',self.gate)]
            def forward_test(self,img_metas):
                value=head.forward_test(memory,outs);self.cached=True;return value
        core=Core();resets=[];fake=types.ModuleType('resident_session_state')
        def reset(c):resets.append(1);c.cached=False
        def snapshot(c):return dict(bev_present=c.cached,tracking_present=c.cached,tracking_pose_present=c.cached,metadata_bev_present=c.cached)
        fake.reset_episode=reset;fake.snapshot=snapshot
        def read(p):return _hugsim_expand(json.loads(Path(p).read_text()))
        def write(p,x):Path(p).write_text(json.dumps(x))
        weights={k:v.clone() for k,v in head.state_dict().items()}
        with torch.no_grad():expected=core.forward_test([[dict(can_bus=np.zeros(18),scene_token='one')]])
        with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,resident_session_state=fake):
            folder=Path(tmp);case=dict(name='render_base',reset=True,timestamp=0.,use_can_bus=False,pretrained_cameras=False,planning_bev='current')
            write(folder/'case.json',case);protocol=SimpleNamespace(PLANNING_BEV='current')
            cleanup=install(core,lambda:folder,protocol,read,write)
            try:
                with torch.no_grad():actual=core.forward_test(img_metas=[[dict(can_bus=np.zeros(18),scene_token='one')]])
                self.assertTrue(torch.equal(expected,actual));self.assertEqual(len(resets),1)
                with np.load(folder/'planner_ablations.npz') as z:np.testing.assert_array_equal(z['original'],actual[0])
                case.update(reset=False,name='history_next',timestamp=.5,planning_bev='history');write(folder/'case.json',case)
                with torch.no_grad():core.forward_test(img_metas=[[dict(can_bus=np.zeros(18),scene_token='one')]])
                self.assertEqual(len(resets),1);self.assertEqual(protocol.PLANNING_BEV,'history')
                self.assertTrue(read(folder/'planner_features.json')['query_reconstruction_verified'])
                for k,v in head.state_dict().items():self.assertTrue(torch.equal(v,weights[k]))
            finally:cleanup()

if __name__=='__main__':unittest.main()
