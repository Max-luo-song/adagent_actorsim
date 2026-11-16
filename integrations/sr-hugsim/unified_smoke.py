"""Exercise the SAME public API on real scenes; no perception/placement certificate."""

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

import json
pass  # Path is provided by the portable boundary above
import numpy as np
from scene_api import SceneSession,SceneSpec,BASE


def save_packet(packet,folder):
    from PIL import Image
    folder.mkdir(parents=True,exist_ok=False)
    arrays={};meta={key:value for key,value in packet.items() if key not in ('cameras','ego_to_world')}
    meta['ego_to_world']=packet['ego_to_world'].tolist();meta['cameras']={}
    for name,view in packet['cameras'].items():
        Image.fromarray(view['rgb']).save(folder/(name+'.png'))
        for field in ('rgb','depth','K','camera_to_world','camera_to_ego'):
            arrays[name+'__'+field]=view[field]
        meta['cameras'][name]=dict(source_camera_id=view['source_camera_id'],rgb_shape=list(view['rgb'].shape))
    np.savez_compressed(folder/'observation.npz',**arrays)
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2))


def render_difference(a,b):
    return {name:dict(changed_rgb_pixels=int(np.count_nonzero(np.any(a['cameras'][name]['rgb']!=b['cameras'][name]['rgb'],axis=-1))),
            rgb_max_uint8_error=int(np.max(np.abs(a['cameras'][name]['rgb'].astype(np.int16)-b['cameras'][name]['rgb'].astype(np.int16)))))
            for name in a['camera_order']}


def run(request_path,output):
    from asset_placement import terrain_aligned_pose
    request=_hugsim_expand(json.loads(Path(request_path).read_text()))
    if set(request)!={'schema','cases','asset','asset_calibration'} or request['schema']!='unified_scene_smoke_v1' or not 1<=len(request['cases'])<=2:
        raise ValueError('Invalid bounded unified-interface probe')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    calibration=_hugsim_expand(json.loads(Path(request['asset_calibration']).read_text()))
    report=dict(schema='unified_scene_smoke_report_v1',training=False,n1_validation_suspended=True,
        n1_certified=False,placement_quality_verified=False,perception_connected=False,
        full_sr_trainer=False,formal_evaluation_admitted=False,scenes=[],
        backend='hugsim_geometry_sr_pbr_v1',request=str(Path(request_path).resolve()))
    previous_token=None
    with SceneSession() as session:
        for case in request['cases']:
            if set(case)!={'scene_manifest','sensor_audit'}:raise ValueError('Invalid case fields')
            spec=SceneSpec.read(case['scene_manifest']);audit=_hugsim_expand(json.loads(Path(case['sensor_audit']).read_text()))
            if Path(audit['source_report']).resolve()!=spec.source_report or audit['checkpoint']!=spec.source['checkpoint']:
                raise ValueError('Sensor audit belongs to another scene')
            description=session.load_scene(case['scene_manifest'])
            assert not session.objects and session.generation!=previous_token
            folder=output/str(spec.source['scene']);folder.mkdir()
            frame=int(audit['automatic_start_frame']);time=frame/spec.fps
            inputs=Path(spec.source['inputs']);start=spec.source['start_timestep']
            ego=np.linalg.inv(np.loadtxt(inputs/'ego_pose'/f'{start:03d}.txt'))@np.loadtxt(inputs/'ego_pose'/f'{start+frame:03d}.txt')
            baseline=session.render(ego,time);save_packet(baseline,folder/'baseline')
            pose=terrain_aligned_pose(audit['proposed_anchor_source_xyz'],audit['heading_source_radians'],
                audit['ground_fit'],calibration['support']['bottom_y'])
            insert=dict(operation='insert',object_id='preview_car',mode='diagnostic_preview',
                asset=request['asset'],asset_calibration=request['asset_calibration'],asset_to_world=pose.tolist())
            (folder/'insert_request.json').write_text(json.dumps(insert,indent=2))
            session.edit(insert);inserted=session.render(ego,time);save_packet(inserted,folder/'inserted')
            change=render_difference(baseline,inserted)
            if not any(item['changed_rgb_pixels']>0 for item in change.values()):
                raise RuntimeError('Insertion has no visible effect in any configured camera')
            moved_pose=pose.copy();moved_pose[:3,3]+=.5*pose[:3,2]
            session.edit(dict(operation='set_pose',object_id='preview_car',mode='diagnostic_preview',asset_to_world=moved_pose.tolist()))
            moved=session.render(ego,time);save_packet(moved,folder/'moved')
            move_change=render_difference(inserted,moved)
            if not any(item['changed_rgb_pixels']>0 for item in move_change.values()):
                raise RuntimeError('Pose update has no visible effect')
            session.edit(dict(operation='remove',object_id='preview_car',mode='diagnostic_preview'))
            removed=session.render(ego,time);save_packet(removed,folder/'removed')
            removal=render_difference(baseline,removed)
            if any(item['rgb_max_uint8_error']>1 for item in removal.values()):
                raise RuntimeError('Removing the managed object failed to restore the baseline')
            from PIL import Image,ImageDraw
            for name in spec.camera_order:
                tiles=[packet['cameras'][name]['rgb'] for packet in (baseline,inserted,moved,removed)]
                canvas=Image.fromarray(np.concatenate(tiles,axis=1));canvas.save(folder/(name+'_baseline_insert_move_remove.png'))
            session.edit(insert);before_reset=session.generation;reset=session.reset()
            if reset['managed_objects'] or reset['timestamp_s']!=0 or reset['history_token']==before_reset:
                raise RuntimeError('Reset failed to clear objects, time or history token')
            stepped=session.step(dict(acc_mps2=1.,steer_rate_radps=0.));save_packet(stepped,folder/'stepped')
            if not np.isclose(stepped['timestamp_s'],spec.vehicle.dt) or np.linalg.norm(stepped['ego_to_world'][:3,3]-reset['ego_to_world'][:3,3])<1e-6:
                raise RuntimeError('Action did not advance ego pose and time')
            # Leave an object before switching; next load must remove it.
            session.edit(insert);previous_token=session.generation
            report['scenes'].append(dict(scene=spec.source['scene'],description=description,
                source_frame=frame,visible_insertion=True,insertion_difference=change,
                pose_update_difference=move_change,removal_difference=removal,
                reset_cleared_objects=True,reset_invalidated_history=True,step_advanced_state=True,
                edit_pose_source='sensor_ground_plane_and_metadata_anchor',
                collision_and_multiview_quality_verified=False,artifacts=str(folder),
                model_cache_token=stepped['model_cache_token']))
            (output/'report.json').write_text(json.dumps(report,indent=2))
            print('Unified API scene passed:',spec.source['scene'],flush=True)
    report.update(status='unified_interface_smoke_passed',scene_switch_cleared_managed_objects=True,
        note='Interface/state/render verification only; no edit-quality, detection or avoidance success claim')
    (output/'report.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(args.request,args.output)
