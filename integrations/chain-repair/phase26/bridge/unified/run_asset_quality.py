"""Same renderer, original/supported source-view poses and existing car reference.

The supported-view pose is an explicit diagnostic scene change, never a
silent production heading correction or a claim of recovered unseen surfaces.
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

import argparse,copy,hashlib,json
pass  # Path is provided by the portable boundary above
import numpy as np
from PIL import Image,ImageDraw
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation
import torch
from run_render_quality import (HERE,BRIDGE,MANIFEST,source,UnifiedSceneSession,
    FrameworkSceneBackend,LayerCapture,save_rgb)
from edit_api import read_command


def camera_azimuth(asset_pose,camera_position):
    """Managed x-longitudinal/y-down/z-lateral -> source x-forward/y-left/z-up."""
    local=(np.asarray(camera_position)-asset_pose[:3,3])@asset_pose[:3,:3]
    return float(np.arctan2(local[2],local[0]))


def supported_view_pose(asset_pose,camera_position,target_degrees):
    pose=np.asarray(asset_pose,dtype=float).copy()
    target=np.deg2rad(target_degrees)
    initial=camera_azimuth(pose,camera_position)-target
    initial=np.arctan2(np.sin(initial),np.cos(initial))
    def candidate(yaw):
        value=pose.copy();value[:3,:3]=Rotation.from_euler('z',yaw).as_matrix()@pose[:3,:3]
        return value
    def cost(yaw):
        diff=camera_azimuth(candidate(yaw),camera_position)-target
        return 1-np.cos(diff)
    fit=minimize_scalar(cost,bounds=(initial-.5,initial+.5),method='bounded',options={'xatol':1e-11})
    if not fit.success or cost(fit.x)>1e-8:raise RuntimeError('could not align diagnostic viewing direction')
    return candidate(fit.x),float(np.rad2deg(fit.x))


def commands(spec):
    inp=Path(spec.source['inputs']);start=spec.source['start_timestep']
    ego=np.linalg.inv(np.loadtxt(inp/'ego_pose'/f'{start:03d}.txt'))@np.loadtxt(inp/'ego_pose'/f'{start+80:03d}.txt')
    camera=(ego@np.loadtxt(inp/'extrinsics/0.txt'))[:3,3]
    base=read_command(_hugsim_expand(json.loads((HERE/'managed_vehicle_command.json').read_text())))
    supported=copy.deepcopy(base)
    pose,yaw=supported_view_pose(base['asset_to_world'],camera,-57.18553928095579)
    supported['asset_to_world']=pose.tolist()
    reference=_hugsim_expand(json.loads((BRIDGE/'requests/closed552_insert_realcar.json').read_text()))
    reference.update(schema='scene_edit_v1',object_id='managed:preview_car',expected_revision=0)
    reference=read_command(reference)
    reference_yaw=copy.deepcopy(reference);r=np.asarray(reference_yaw['asset_to_world'])
    r[:3,:3]=Rotation.from_euler('z',15,degrees=True).as_matrix()@r[:3,:3]
    reference_yaw['asset_to_world']=r.tolist()
    return ego,[('source_original',base),('source_supported_view',supported),('reference_realcar',reference),('reference_realcar_yaw15',reference_yaw)],dict(
        original_view_degrees=float(np.rad2deg(camera_azimuth(np.asarray(base['asset_to_world']),camera))),
        supported_view_degrees=float(np.rad2deg(camera_azimuth(pose,camera))),diagnostic_heading_change_degrees=yaw,
        source_view_target_degrees=-57.18553928095579,
        target_source='render_quality_followups/20261103T075512Z/analysis_20261103/actor_source_audit.json; source instance2/track25',
        heading_change_is_diagnostic_only=True,reference_pose='existing calibrated realcar insertion request; ground calibration retained')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--check-only',action='store_true');args=p.parse_args()
    spec=source();ego,cases,view=commands(spec)
    from asset_calibration import load
    preflight={}
    for name,command in cases:
        asset=Path(command['asset']);load(Path(command['asset_calibration']),asset)
        preflight[name]=dict(asset=str(asset),gs_bytes=(asset/'gs.pth').stat().st_size,
            calibration=command['asset_calibration'],pose=command['asset_to_world'])
    args.output.mkdir(parents=True,exist_ok=False);out=args.output.resolve()
    paths=[Path(__file__),HERE/'managed_vehicle_command.json',HERE/'run_render_quality.py',
        BRIDGE/'framework_backend/managed_assets.py',BRIDGE/'framework_backend/backend.py',
        BRIDGE/'requests/closed552_insert_realcar.json']
    report=dict(complete=False,view=view,preflight=preflight,gpu_render_executed=False,
        implementation_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
        visual_quality_accepted=False,formal_evaluation_admitted=False,
        scope='explicit diagnostic view/asset controls; does not replace original source vehicles')
    (out/'commands.json').write_text(json.dumps(dict(cases=cases,view=view),indent=2))
    if args.check_only:
        report['cpu_preflight_passed']=True;(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(view,indent=2));return
    try:
        torch.set_num_threads(2)
        with UnifiedSceneSession(FrameworkSceneBackend) as session:
            session.load_scene(MANIFEST);capture=LayerCapture(session.backend);capture.target_kind='managed'
            try:
                for name,command in cases:
                    session.reset();session.edit(command);capture.rows.clear()
                    folder=out/name;folder.mkdir();packet=session.render(ego,8.)
                    for cam,c in packet['cameras'].items():Image.fromarray(c['rgb']).save(folder/f'{cam}.png')
                    row=capture.rows[0]
                    for key in ('target','target_gray','target_alpha','without_target'):
                        save_rgb(folder/f'{key}.png',row[key])
                    (folder/'journal.json').write_text(json.dumps(session.edit_journal(),indent=2))
                    print('Asset quality comparison saved:',name,flush=True)
            finally:capture.close()
        thumbs=[]
        for name,_ in cases:
            im=Image.open(out/name/'CAM_FRONT.png').convert('RGB');h=round(im.height*640/im.width)
            tile=Image.new('RGB',(640,h+30));tile.paste(im.resize((640,h)),(0,30));ImageDraw.Draw(tile).text((8,8),name,fill='white');thumbs.append(tile)
        sheet=Image.new('RGB',(1280,thumbs[0].height*2))
        for i,tile in enumerate(thumbs):sheet.paste(tile,((i%2)*640,(i//2)*tile.height))
        sheet.save(out/'asset_view_comparison.jpg',quality=95)
        report.update(complete=True,gpu_render_executed=True)
    except Exception as exc:report['error']=f'{type(exc).__name__}: {exc}';raise
    finally:(out/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
