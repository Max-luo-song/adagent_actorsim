"""Fixed-scene appearance repair and layer ablations; no weather/UniAD needed.

Original checkpoints/evidence are read-only. Render completion and SH
consistency are engineering checks, never automatic visual-quality approval.
"""

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

import argparse
import copy
import hashlib
import json
pass  # Path is provided by the portable boundary above
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
import torch

HERE=Path(__file__).resolve().parent
BRIDGE=HERE.parent
sys.path[:0]=[str(BRIDGE/'framework_backend'),str(HERE)]
import run_arm_fwbackend as fw
from scene_edit_session import UnifiedSceneSession
from backend import FrameworkSceneBackend
from native_radiance import radiance_state, restore_radiance
from native_edit import rotate_native
from asset_radiance import evaluator

MANIFEST=BRIDGE.parent.parent/'phase4/dynamic552_frame80/scene_manifest.json'
CAMERAS=('CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT')
FIELDS=('_means','_scales','_quats','_rgbs','_opacities')


def source():
    from scene_api import SceneSpec
    return SceneSpec.read(MANIFEST)


def coverage_audit(data,spec):
    ids=data['points_ids'].reshape(-1)==1
    means=data['_means'][ids].numpy()
    weights=data['_opacities'][ids].sigmoid().reshape(-1).numpy()
    half_length=float(data['instances_size'][1,0])/2
    rear=means[:,0]<-half_length+1
    front=means[:,0]>half_length-1
    inp=Path(spec.source['inputs']);start=spec.source['start_timestep']
    first=np.linalg.inv(np.loadtxt(inp/'ego_pose'/f'{start:03d}.txt'))
    valid=data['instances_fv'][:,1].numpy().astype(bool);angles=[]
    extrinsics=[np.loadtxt(inp/'extrinsics'/f'{i}.txt') for i in (0,1,2)]
    for frame in np.flatnonzero(valid):
        ego=first@np.loadtxt(inp/'ego_pose'/f'{start+frame:03d}.txt')
        root=Rotation.from_quat(data['instances_quats'][frame,1].numpy()[[1,2,3,0]]).as_matrix()
        for ext in extrinsics:
            camera=ego@ext
            local=(camera[:3,3]-data['instances_trans'][frame,1].numpy())@root
            angles.append(np.arctan2(local[1],local[0]))
    angles=np.asarray(angles);center=np.angle(np.exp(1j*angles).mean())
    offsets=np.angle(np.exp(1j*(angles-center)))
    return dict(rear_end_within_1m_fraction=float(rear.mean()),front_end_within_1m_fraction=float(front.mean()),
        rear_end_opacity_weighted_fraction=float(weights[rear].sum()/weights.sum()),
        front_end_opacity_weighted_fraction=float(weights[front].sum()/weights.sum()),valid_frames=int(valid.sum()),
        camera_center_azimuth_mean_degrees=float(np.rad2deg(center)),
        camera_center_azimuth_span_degrees=float(np.rad2deg(offsets.max()-offsets.min())),
        scope='Gaussian-center density and source camera-center directions; not measured pixel visibility or a causal geometry-quality proof')


def cpu_audit(spec):
    """Actual source coefficients, not synthetic imagery or GPU proxy renders."""
    data=torch.load(spec.source['checkpoint'],map_location='cpu',weights_only=False,mmap=True)['models']['RigidNodes']
    model=SimpleNamespace(**{k:data[k].clone() for k in ('instances_trans','instances_quats','_features_rest')})
    model.point_ids=data['points_ids'].clone()
    backend=SimpleNamespace(torch=torch,_last=None,trainer=SimpleNamespace(models={'RigidNodes':model}),
        env=SimpleNamespace(models={'RigidNodes':dict(instances_trans=model.instances_trans,instances_quats=model.instances_quats)}))
    ids=(model.point_ids.reshape(-1)==1).nonzero(as_tuple=True)[0]
    chosen=ids[torch.linspace(0,len(ids)-1,min(2048,len(ids))).long()]
    old=torch.cat((data['_features_dc'][chosen,None],data['_features_rest'][chosen]),1).numpy().transpose(0,2,1)
    before=radiance_state(model,1)
    q=model.instances_quats.clone();t=model.instances_trans.clone()
    rotate_native(backend,dict(schema='framework_native_edit_v1',operation='rotate',group='RigidNodes',instance=1,yaw_degrees=15.))
    new=torch.cat((data['_features_dc'][chosen,None],model._features_rest[chosen]),1).numpy().transpose(0,2,1)
    rng=np.random.default_rng(42);directions=rng.normal(size=(len(chosen),3));directions/=np.linalg.norm(directions,axis=-1,keepdims=True)
    rotation=Rotation.from_euler('z',15,degrees=True).as_matrix()
    expected=evaluator()(3,old,directions)
    legacy=evaluator()(3,old,directions@rotation.T)
    repaired=evaluator()(3,new,directions@rotation.T)
    restore_radiance(torch,model,before)
    scales=data['_scales'][ids].exp().numpy()
    extents=np.quantile(data['_means'][ids].abs().numpy(),[.5,.95,.99,1.],axis=0)
    checks=dict(actual_target_has_gaussians=len(ids)>0,
        sh_rigid_rotation_max_error_below_2e5=float(np.max(np.abs(repaired-expected)))<2e-5,
        original_world_fixed_sh_changes_under_rigid_rotation=float(np.mean(np.abs(legacy-expected)))>1e-5,
        root_centers_unchanged=bool(torch.equal(model.instances_trans,t)),
        appearance_reset_bit_exact=bool(torch.equal(model._features_rest,data['_features_rest'])))
    return dict(passed=all(checks.values()),checks=checks,checkpoint=str(spec.source['checkpoint']),
        target='RigidNodes:1',target_gaussians=len(ids),sampled_gaussians=len(chosen),
        legacy_radiance_mae=float(np.mean(np.abs(legacy-expected))),
        repaired_radiance_mae=float(np.mean(np.abs(repaired-expected))),
        repaired_radiance_max_abs=float(np.max(np.abs(repaired-expected))),
        radiance_error_is_not_image_quality_score=True,
        scale_max_axis_m_quantiles={str(v):float(np.quantile(scales.max(-1),v)) for v in (.5,.9,.99,1.)},
        anisotropy_ratio_quantiles={str(v):float(np.quantile(scales.max(-1)/np.maximum(scales.min(-1),1e-12),v)) for v in (.5,.9,.99,1.)},
        local_abs_extent_quantiles_xyz=extents.tolist(),box_lwh=data['instances_size'][1].tolist(),
        coverage=coverage_audit(data,spec),
        source_disk_unchanged=True,gpu_render_executed=False,visual_quality_accepted=False)


def native_render_center(trainer, frame=80, instance=1):
    """Use the same temporal center as RigidNodes.transform_means.

    Test frames use adjacent valid-frame root positions; the annotation center
    at the requested frame is not necessarily the rendered Gaussian pivot.
    """
    model=trainer.models['RigidNodes']
    interpolate=(frame in trainer.test_set_indices and frame-1>0 and frame+1<model.instances_trans.shape[0]
                 and bool(model.instances_fv[frame-1,instance]) and bool(model.instances_fv[frame+1,instance]))
    center=(model.instances_trans[frame-1,instance]+model.instances_trans[frame+1,instance])*.5 if interpolate else model.instances_trans[frame,instance]
    return center.detach().cpu().numpy(),bool(interpolate)


class LayerCapture:
    def __init__(self,backend):
        self.backend=backend;self.trainer=backend.trainer
        self.original=self.trainer.render_gaussians;self.rows=[];self.target_kind="native";self.capture_groups=False
        self.trainer.render_gaussians=self.render
    def close(self):self.trainer.render_gaussians=self.original
    def render(self,gs,cam,**kwargs):
        output,fn=self.original(gs=gs,cam=cam,**kwargs)
        # Camera0 is front. Capture raw pre-affine layers only for this camera.
        if not self.rows:
            trainer=self.trainer
            if self.target_kind == 'managed':
                target=trainer.pts_labels==trainer.managed_dynamic_label
                if not bool(target.any()):raise RuntimeError('managed target absent from collection')
            else:
                model=trainer.models['RigidNodes']
                positions=(trainer.pts_labels==trainer.gaussian_classes['RigidNodes']).nonzero(as_tuple=True)[0]
                if len(model.point_ids)!=len(model.filter_mask):
                    raise RuntimeError('native Gaussian IDs and collection differ')
                ids=model.point_ids[model.filter_mask].reshape(-1)
                if len(ids)!=len(positions):raise RuntimeError('native Gaussian IDs and collection differ')
                target=trainer.pts_labels.new_zeros(len(trainer.pts_labels),dtype=torch.bool)
                target[positions]=ids==1
            info=trainer.info
            row={}
            try:
                for name,keep,gray in (('target',target,False),('target_gray',target,True),('without_target',~target,False)):
                    view=copy.copy(gs)
                    view._opacities=gs.opacities*keep[:,None]
                    if gray:view._rgbs=torch.full_like(gs.rgbs,.6)
                    raw,_=self.original(gs=view,cam=cam,**kwargs)
                    row[name]=raw['rgb_gaussians'].detach().cpu().numpy()
                    row[name+'_alpha']=raw['opacity'].detach().cpu().numpy().squeeze(-1)
                if self.capture_groups:
                    masks={name:trainer.pts_labels==label for name,label in trainer.gaussian_classes.items() if name!='RigidNodes'}
                    masks['other_RigidNodes']=(trainer.pts_labels==trainer.gaussian_classes['RigidNodes']) & ~target
                    for name,mask in masks.items():
                        view=copy.copy(gs);view._opacities=gs.opacities*mask[:,None]
                        raw,_=self.original(gs=view,cam=cam,**kwargs)
                        row['group_'+name]=raw['rgb_gaussians'].detach().cpu().numpy()
                        row['group_'+name+'_alpha']=raw['opacity'].detach().cpu().numpy().squeeze(-1)
                row['depth']=output['depth'].detach().cpu().numpy()
            finally:trainer.info=info
            self.rows.append(row)
        return output,fn


def save_rgb(path,value):
    Image.fromarray(np.clip(np.asarray(value)*255,0,255).astype(np.uint8)).save(path)


def render_audit(spec,out):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for actual raster validation')
    inp=Path(spec.source['inputs']);start=spec.source['start_timestep']
    pose=np.linalg.inv(np.loadtxt(inp/'ego_pose'/f'{start:03d}.txt'))@np.loadtxt(inp/'ego_pose'/f'{start+80:03d}.txt')
    captures={};checks={};journal=[]
    with UnifiedSceneSession(FrameworkSceneBackend) as session:
        session.load_scene(MANIFEST)
        capture=LayerCapture(session.backend)
        center,center_interpolated=native_render_center(session.backend.trainer)
        delta=Rotation.from_euler('z',15,degrees=True).as_matrix()
        orbit=pose.copy();orbit[:3,:3]=delta@pose[:3,:3];orbit[:3,3]=center+delta@(pose[:3,3]-center)
        novel=pose.copy();novel[:3,3]+=pose[:3,1]*.5
        variants=[('baseline',0,'original',pose),('legacy_yaw15',15,'legacy',pose),
            ('repaired_yaw15',15,'repaired',pose),('dc_yaw15',15,'dc',pose),
            ('legacy_orbit15',15,'legacy',orbit),('repaired_orbit15',15,'repaired',orbit),
            ('repaired_yaw5',5,'repaired',pose),('repaired_yaw30',30,'repaired',pose),
            ('novel_lateral05_yaw0',0,'original',novel),('baseline_reset',0,'original',pose)]
        try:
            for name,yaw,appearance,ego in variants:
                session.reset()
                if yaw:
                    session.edit(dict(schema='scene_edit_v1',mode='diagnostic_preview',operation='rotate',
                        object_id='native:RigidNodes:1',yaw_degrees=yaw))
                model=session.backend.trainer.models['RigidNodes']
                if appearance=='legacy':restore_radiance(torch,model,session.editor.original_radiance[('RigidNodes',1)])
                if appearance=='dc':
                    state=radiance_state(model,1)
                    with torch.no_grad():model._features_rest[state[0]]=0
                folder=out/name;folder.mkdir()
                capture.rows.clear()
                packet=session.render(ego,8.)
                for cam,c in packet['cameras'].items():Image.fromarray(c['rgb']).save(folder/f'{cam}.png')
                row=capture.rows[0];captures[name]=row
                for layer in ('target','target_gray','without_target'):
                    save_rgb(folder/f'{layer}_raw.png',row[layer])
                    save_rgb(folder/f'{layer}_alpha.png',row[layer+'_alpha'])
                np.save(folder/'depth.npy',row['depth'])
                journal.append(dict(name=name,yaw_degrees=yaw,appearance=appearance,
                    ego_to_world=ego.tolist(),journal=session.edit_journal(),
                    ablation_only=appearance in ('legacy','dc'),
                    layers='raw Gaussian RGB/alpha before sky/affine; full CAM images use normal production path'))
                print('Render quality case saved:',name,flush=True)
            # Independent calibrated vehicle package, separate from native bus.
            asset_command=_hugsim_expand(json.loads((HERE/'managed_vehicle_command.json').read_text()))
            asset_pose=np.asarray(asset_command['asset_to_world'],dtype=float)
            managed_orbit=pose.copy()
            managed_orbit[:3,:3]=delta@pose[:3,:3]
            managed_orbit[:3,3]=asset_pose[:3,3]+delta@(pose[:3,3]-asset_pose[:3,3])
            for name,yaw,ego in (('managed_original',0,pose),('managed_yaw15',15,pose),
                                 ('managed_novel_lateral05',0,novel),('managed_orbit15',15,managed_orbit)):
                session.reset();session.edit(asset_command)
                if yaw:
                    asset_delta=asset_pose.copy();asset_delta[:3,:3]=delta@asset_pose[:3,:3]
                    session.edit(dict(schema='scene_edit_v1',mode='diagnostic_preview',operation='set_pose',
                        object_id=asset_command['object_id'],asset_to_world=asset_delta.tolist()))
                capture.target_kind='managed';capture.rows.clear()
                folder=out/name;folder.mkdir();packet=session.render(ego,8.)
                for cam,c in packet['cameras'].items():Image.fromarray(c['rgb']).save(folder/f'{cam}.png')
                row=capture.rows[0];captures[name]=row
                for layer in ('target','target_gray','without_target'):
                    save_rgb(folder/f'{layer}_raw.png',row[layer])
                    save_rgb(folder/f'{layer}_alpha.png',row[layer+'_alpha'])
                np.save(folder/'depth.npy',row['depth'])
                journal.append(dict(name=name,yaw_degrees=yaw,appearance='managed_local_SH',
                    ego_to_world=ego.tolist(),journal=session.edit_journal(),ablation_only=False,
                    layers='managed asset isolated from same shared Gaussian raster; raw pre-affine'))
                print('Render quality case saved:',name,flush=True)
            asset_mask=captures['managed_original']['target_alpha']>.2
            checks['managed_target_visible_in_isolated_layer']=bool(asset_mask.any())
            managed_error=float(np.abs(captures['managed_original']['target']-captures['managed_orbit15']['target'])[asset_mask].mean())
            checks['managed_orbit_target_rgb_mae_below_001']=managed_error<.01
            checks['legacy_and_repaired_geometry_depth_identical']=bool(np.array_equal(captures['legacy_yaw15']['depth'],captures['repaired_yaw15']['depth']))
            checks['legacy_and_repaired_alpha_identical']=bool(np.array_equal(captures['legacy_yaw15']['target_alpha'],captures['repaired_yaw15']['target_alpha']))
            checks['reset_target_rgb_identical']=bool(np.array_equal(captures['baseline']['target'],captures['baseline_reset']['target']))
            checks['reset_background_rgb_identical']=bool(np.array_equal(captures['baseline']['without_target'],captures['baseline_reset']['without_target']))
            # Camera and object undergo the same rigid transform. Target-only
            # radiance should be invariant; scene/background are allowed to change.
            mask=captures['baseline']['target_alpha']>.2
            old_error=np.abs(captures['baseline']['target']-captures['legacy_orbit15']['target'])[mask]
            new_error=np.abs(captures['baseline']['target']-captures['repaired_orbit15']['target'])[mask]
            orbit_errors=dict(pivot_world=center.tolist(),pivot_interpolated=center_interpolated,legacy_mae=float(old_error.mean()),repaired_mae=float(new_error.mean()),
                alpha_mae=float(np.abs(captures['baseline']['target_alpha']-captures['repaired_orbit15']['target_alpha']).mean()))
            checks['orbit_target_radiance_improves']=orbit_errors['repaired_mae']<orbit_errors['legacy_mae']
            checks['orbit_target_rgb_mae_below_001']=orbit_errors['repaired_mae']<.01
            checks['all_fourteen_cases_saved']=len(journal)==14
            orbit_errors['managed_repaired_mae']=managed_error
        finally:capture.close()
    (out/'cases.json').write_text(json.dumps(journal,indent=2))
    # Contact sheet uses original render PNGs, with clear ablation labels.
    thumbs=[]
    for name in ('baseline','legacy_yaw15','repaired_yaw15','dc_yaw15'):
        im=Image.open(out/name/'CAM_FRONT.png').convert('RGB');h=round(im.height*640/im.width)
        tile=Image.new('RGB',(640,h+30));tile.paste(im.resize((640,h)),(0,30))
        ImageDraw.Draw(tile).text((8,8),name,fill='white');thumbs.append(tile)
    sheet=Image.new('RGB',(1280,thumbs[0].height*2))
    for i,im in enumerate(thumbs):sheet.paste(im,((i%2)*640,(i//2)*im.height))
    sheet.save(out/'appearance_comparison.jpg',quality=94)
    thumbs=[]
    for name in ('managed_original','managed_yaw15','managed_novel_lateral05','managed_orbit15'):
        im=Image.open(out/name/'CAM_FRONT.png').convert('RGB');h=round(im.height*640/im.width)
        tile=Image.new('RGB',(640,h+30));tile.paste(im.resize((640,h)),(0,30))
        ImageDraw.Draw(tile).text((8,8),name,fill='white');thumbs.append(tile)
    sheet=Image.new('RGB',(1280,thumbs[0].height*2))
    for i,im in enumerate(thumbs):sheet.paste(im,((i%2)*640,(i//2)*im.height))
    sheet.save(out/'managed_comparison.jpg',quality=94)
    return dict(engineering_checks_passed=all(checks.values()),checks=checks,orbit_errors=orbit_errors,
        gpu_render_executed=True,visual_quality_accepted=False,
        appearance_repair_scope='edit-delta SH rotation; no geometry cleanup/retraining or invented pixels',
        physical_weather=False,formal_evaluation_admitted=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu-audit-only',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);out=args.output.resolve()
    torch.set_num_threads(2)
    spec=source();audit=cpu_audit(spec)
    (out/'cpu_source_audit.json').write_text(json.dumps(audit,indent=2));assert audit['passed']
    files=[Path(__file__),BRIDGE/'framework_backend/native_edit.py',BRIDGE/'framework_backend/native_radiance.py',
        HERE/'edit_api.py',HERE/'scene_edit_session.py',HERE/'run_phase26.sh',HERE/'managed_vehicle_command.json',
        BRIDGE/'framework_backend/managed_assets.py',
        BRIDGE/'framework_backend/backend.py',
        BRIDGE.parent/'framework_repo/models/nodes/rigid.py',
        BRIDGE.parent/'framework_repo/models/trainers/base.py',
        fw.ROOT/'waymo23-native-import/asset_radiance.py']
    (out/'implementation.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
    if args.cpu_audit_only:
        print(json.dumps(audit,indent=2));return
    report=dict(gpu_render_executed=False,visual_quality_accepted=False)
    try:
        report=render_audit(spec,out)
        if not report['engineering_checks_passed']:raise RuntimeError('appearance engineering checks failed; inspect report.json')
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}';raise
    finally:(out/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
