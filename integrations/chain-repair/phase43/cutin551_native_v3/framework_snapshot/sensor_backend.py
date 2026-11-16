"""Camera-Z/alpha sensor contract for the edited framework scene only.

Uses the trainer RGB+ED output before video ego-depth decoration. No synthetic
variance, surface, boxes, or ground. Runtime rejects a changed renderer mode.
"""
import numpy as np
from backend import FrameworkSceneBackend,CAM_ID_TO_NAME


def sensor_plane(value,shape,name):
    value=np.asarray(value,dtype=np.float32)
    if value.shape==(*shape,1):value=value[...,0]
    if value.shape!=shape or not np.isfinite(value).all() or np.any(value<0):
        raise ValueError('Invalid '+name)
    if name=='alpha' and np.any(value>1.00001):raise ValueError('Invalid alpha range')
    return value


class ClosedLoopFrameworkBackend(FrameworkSceneBackend):
    def render(self,pose,timestamp):
        if self.closed:raise RuntimeError('backend closed')
        pose=np.asarray(pose,dtype=float)
        if pose.shape!=(4,4) or not np.isfinite(pose).all():raise ValueError('Invalid pose')
        torch=self.torch
        self.trainer.set_eval()
        frame=min(max(int(getattr(self,'_actor_source_frame',round(timestamp*self.spec.fps))),0),self.frames-1)
        traj=torch.from_numpy(pose).float().cuda().unsqueeze(0)
        cameras={}
        # Verify the actual invocation, not only the source render-mode string.
        original=self.trainer.render_gaussians
        modes=[]
        def checked(*args,**kwargs):
            mode=kwargs.get('render_mode')
            if mode!='RGB+ED':raise RuntimeError('Unexpected sensor render mode: '+str(mode))
            modes.append(mode)
            return original(*args,**kwargs)
        self.trainer.render_gaussians=checked
        try:
            with torch.no_grad():
                for cid,name in CAM_ID_TO_NAME.items():
                    camera=self.cameras_data[cid]
                    if camera.is_fisheye:raise RuntimeError('Pinhole control sensor required')
                    data=self.dataset.prepare_online_render_data(traj=traj,target_cam_data=camera,frame_id=frame)
                    if len(data)!=1:raise RuntimeError('Expected exactly one current frame')
                    item=data[0]
                    for group in ('cam_infos','image_infos'):
                        for key,value in item[group].items():item[group][key]=value.cuda(non_blocking=True)
                    count=len(modes)
                    outputs,_=self.trainer(image_infos=item['image_infos'],camera_infos=item['cam_infos'],novel_view=True,render_only=True)
                    if len(modes)==count:raise RuntimeError('Sensor bypassed checked rasterization')
                    rgb=outputs['rgb'].clamp(0,1)
                    mask=item['image_infos'].get('egocar_masks')
                    if mask is not None and getattr(camera,'ego_template',None) is not None:
                        rgb=torch.where(mask[...,None],camera.ego_template,rgb)
                    rgb=(rgb.detach().cpu().numpy()*255).clip(0,255).astype(np.uint8)
                    shape=rgb.shape[:2]
                    depth=sensor_plane(outputs['depth'].detach().cpu().numpy(),shape,'depth')
                    alpha=sensor_plane(outputs['opacity'].detach().cpu().numpy(),shape,'alpha').copy()
                    # A composited ego template has no measured geometric depth.
                    if mask is not None:alpha[np.asarray(mask.detach().cpu(),dtype=bool).reshape(shape)]=0
                    K=camera.intrinsics[0].detach().cpu().numpy()
                    cameras[name]=dict(rgb=rgb,depth=depth,depth_alpha=alpha,
                        depth_encoding='expected_camera_z_v1',K=K,
                        camera_to_world=pose@np.asarray(camera.cam_to_ego,dtype=float),source_camera_id=cid)
        finally:self.trainer.render_gaussians=original
        self._last=(float(timestamp),pose.copy())
        return dict(cameras=cameras,ego_state=dict(speed_mps=None,steering_rad=None,note='frozen HUGSIM Motion owns state'))
