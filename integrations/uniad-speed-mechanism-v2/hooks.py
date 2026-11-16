"""Offline diagnostic hooks. Original forward results are returned without modification."""

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

import inspect
pass  # Path is provided by the portable boundary above
import numpy as np

def install(core,current_folder,protocol,read,write):
    import torch
    from resident_session_state import reset_episode,snapshot
    original=core.forward_test;head=core.planning_head;attention=head.attn_module.forward
    motion=core.motion_head.forward_test;context={}
    def stats(x):
        x=x.detach().float()
        return dict(shape=list(x.shape),mean=float(x.mean()),std=float(x.std()),norm=float(x.norm()),finite=bool(torch.isfinite(x).all()))
    def save_fuser(module,args):context['fuser']=args[0].detach().clone()
    handle=head.mlp_fuser.register_forward_pre_hook(save_fuser)
    def plan_from_attention(out):return head.reg_branch(out).reshape(-1,6,2).cumsum(dim=1)[0]
    def observed_attention(query,memory,*args,**kwargs):
        result=attention(query,memory,*args,**kwargs)
        fuser=context['fuser'];fused=head.mlp_fuser(fuser).max(1,keepdim=True)[0].permute(1,0,2)+head.pos_embed.weight[None]
        if not torch.allclose(fused,query,rtol=0,atol=1e-5):raise ValueError('Planning query reconstruction mismatch')
        # The hook fires again for the numerical replay; retain the original capture.
        context['fuser']=fuser
        no_sdc=fuser.clone();no_sdc[...,:512]=0
        query_no_sdc=head.mlp_fuser(no_sdc).max(1,keepdim=True)[0].permute(1,0,2)+head.pos_embed.weight[None]
        context['fuser']=fuser
        position_grid=context['bev_pos'].to(memory)
        if head.with_adapter:position_grid=position_grid+head.bev_adapter(position_grid)
        position=position_grid.flatten(2).permute(2,0,1)
        if position.shape!=memory.shape:raise ValueError('Position-only memory layout mismatch')
        values=dict(original=plan_from_attention(result),
            no_sdc_features=plan_from_attention(attention(query_no_sdc,memory,*args,**kwargs)),
            position_only_memory=plan_from_attention(attention(query,position,*args,**kwargs)))
        folder=current_folder()
        np.savez_compressed(folder/'planner_ablations.npz',**{k:v.detach().cpu().numpy() for k,v in values.items()})
        write(folder/'planner_features.json',dict(query=stats(query),memory=stats(memory),
            sdc_motion_features=stats(fuser[...,:256]),sdc_track_features=stats(fuser[...,256:512]),
            navigation_features=stats(fuser[...,512:]),query_reconstruction_verified=True,
            max_plan_delta_without_sdc_m=float((values['original']-values['no_sdc_features']).abs().max()),
            max_plan_delta_position_only_memory_m=float((values['original']-values['position_only_memory']).abs().max()),
            ablations_reference_only=True,original_attention_result_returned=True))
        if read(folder/'case.json')['name'] in ('render_base','source_rgb'):
            np.savez(folder/'planner_inputs.npz',query=query.detach().cpu().numpy(),memory=memory.detach().cpu().numpy(),
                fuser=fuser.detach().cpu().numpy(),bev_pos=context['bev_pos'].detach().cpu().numpy())
        return result
    planning_test=head.forward_test
    def observed_planning(*args,**kwargs):
        bound=inspect.signature(planning_test).bind(*args,**kwargs)
        context['bev_pos']=bound.arguments['outs_motion']['bev_pos'].detach()
        return planning_test(*args,**kwargs)
    def observed_motion(*args,**kwargs):
        bound=inspect.signature(motion).bind(*args,**kwargs);track=bound.arguments['outs_track']
        box=track['sdc_track_bbox_results'][0][0].tensor
        original_box=box.detach().clone();case=read(current_folder()/'case.json')
        if case.get('override_sdc_velocity'):
            box[:,7:9]=box.new_tensor([0.,16.])
        try:return motion(*args,**kwargs)
        finally:box.copy_(original_box)
    def forward(*args,**kwargs):
        folder=current_folder();case=read(folder/'case.json');bound=inspect.signature(original).bind(*args,**kwargs)
        meta=bound.arguments['img_metas']
        while isinstance(meta,list):meta=meta[0]
        before=snapshot(core)
        if case['reset']:
            reset_episode(core);cleared=snapshot(core)
            if any(cleared[k] for k in ['bev_present','tracking_present','tracking_pose_present','metadata_bev_present']):raise ValueError('Incomplete cold reset')
            # Preserve the adapter's current-scene clock for the subsequent real history frames.
            core._adapter_scene_token=meta['scene_token'];core._adapter_timestamp=float(case['timestamp'])
        else:
            cleared=None
            if not before['bev_present'] or not before['tracking_present']:raise ValueError('Required real history missing')
        gates={};cameras={}
        for name,m in core.named_modules():
            if hasattr(m,'use_can_bus'):m.use_can_bus=case['use_can_bus'];gates[name]=bool(m.use_can_bus)
            if hasattr(m,'use_cams_embeds'):m.use_cams_embeds=case['pretrained_cameras'];cameras[name]=bool(m.use_cams_embeds)
        protocol.PLANNING_BEV=case['planning_bev']
        write(folder/'mechanism_audit.json',dict(case=case,before=before,cold_state=cleared,actual_CAN_gates=gates,
            actual_camera_embeddings=cameras,actual_planning_bev=protocol.PLANNING_BEV,
            incoming_can_bus=meta['can_bus'].tolist(),closed_loop_driving=False,model_weights_changed=False))
        return original(*args,**kwargs)
    core.forward_test=forward;head.attn_module.forward=observed_attention;head.forward_test=observed_planning;core.motion_head.forward_test=observed_motion
    def cleanup():
        core.forward_test=original;head.attn_module.forward=attention;head.forward_test=planning_test;core.motion_head.forward_test=motion;handle.remove()
    return cleanup
