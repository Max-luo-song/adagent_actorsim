"""Frozen same-camera lighting fallback for source-reconstruction held-out times."""
from contextlib import contextmanager
import bisect
import torch

MODES=('direct','interpolate','camera_mean')


def build_tables(model,cameras,train_frames,time_denominator):
    frames=tuple(sorted(set(int(f) for f in train_frames)))
    if not frames or time_denominator<=0:raise ValueError('Training frames and original lighting clock required')
    weight=model.appearance_embedding.weight
    tables={}
    with torch.no_grad():
        for cam,camera in cameras.items():
            if frames[0]<0 or frames[-1]>=len(camera.unique_img_idx):raise ValueError('Training frame outside camera clock')
            source_ids=camera.unique_img_idx.detach().cpu().tolist() if isinstance(camera.unique_img_idx,torch.Tensor) else list(camera.unique_img_idx)
            id_values=[int(source_ids[f]) for f in frames]
            ids=torch.tensor(id_values,device=weight.device,dtype=torch.long)
            if bool((ids<0).any()) or bool((ids>=len(weight)).any()):raise ValueError('Training image ID outside checkpoint')
            times=torch.tensor(frames,device=weight.device,dtype=weight.dtype)[:,None]/time_denominator
            coefficients=model(times,ids).detach().clone()
            if not torch.isfinite(coefficients).all():raise ValueError('Nonfinite lighting table')
            tables[cam]=dict(frames=frames,image_ids=tuple(id_values),coefficients=coefficients,
                mean=coefficients.mean(dim=0),time_denominator=time_denominator)
    return tables


def fallback_coefficients(tables,mode,frame,cam):
    if mode not in MODES:raise ValueError('Unknown lighting mode')
    table=tables[cam];frames=table['frames']
    if mode=='direct' or frame in frames:
        return None,dict(mode=mode,used_fallback=False,frame=frame,cam=cam)
    if mode=='camera_mean':
        return table['mean'][None],dict(mode=mode,used_fallback=True,frame=frame,cam=cam,source_count=len(frames),
            source_frames=list(frames),source_image_ids=list(table['image_ids']))
    pos=bisect.bisect_left(frames,frame)
    left=max(0,pos-1);right=min(len(frames)-1,pos)
    fraction=0. if left==right else (frame-frames[left])/(frames[right]-frames[left])
    result=(1-fraction)*table['coefficients'][left]+fraction*table['coefficients'][right]
    return result[None],dict(mode=mode,used_fallback=True,frame=frame,cam=cam,
        source_frames=[frames[left],frames[right]],source_image_ids=[table['image_ids'][left],table['image_ids'][right]],
        weights=[1-fraction,fraction])


@contextmanager
def override_lighting(model,coefficients):
    if coefficients is None:
        yield
        return
    def hook(module,args,output):
        if output.shape!=coefficients.shape or output.device!=coefficients.device or output.dtype!=coefficients.dtype:
            raise ValueError('Lighting override shape/device/dtype mismatch')
        return coefficients
    handle=model.register_forward_hook(hook)
    try:yield
    finally:handle.remove()
