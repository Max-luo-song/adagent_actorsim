"""Render exactly frames80..82 at recorded poses; masks valid only at these poses.

This is an offline framework-render input test, not executed closed-loop poses.
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
import hashlib
import json
import sys
pass  # Path is provided by the portable boundary above
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent
BRIDGE=HERE.parent
JOB=HERE.parents[3]
ROOT=JOB.parents[3]
sys.path[:0]=[str(BRIDGE/'framework_backend'),str(ROOT/'integrations/sr-hugsim')]
from scene_api import SceneSpec
from backend import FrameworkSceneBackend, FRAMEWORK_DATA_ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    spec=SceneSpec.read(JOB/'chain_repair/phase4/dynamic552_frame80/scene_manifest.json')
    inp=Path(spec.source['inputs']);raw_start=spec.source['start_timestep']
    origin=np.loadtxt(inp/'ego_pose'/f'{raw_start:03d}.txt')
    backend=FrameworkSceneBackend(spec)
    records=[]
    try:
        for frame in (80,81,82):
            raw_frame=raw_start+frame
            pose=np.linalg.inv(origin) @ np.loadtxt(inp/'ego_pose'/f'{raw_frame:03d}.txt')
            render=backend.render(pose,frame/spec.fps)
            # Explicit frame-major, then camera-id order; not implicit dict order.
            for cam_id,name in ((0,'CAM_FRONT'),(1,'CAM_FRONT_LEFT'),(2,'CAM_FRONT_RIGHT')):
                camera=render['cameras'][name]
                image=np.asarray(camera['rgb'])
                mask_src=Path(FRAMEWORK_DATA_ROOT)/str(spec.source['scene'])/'dynamic_masks/all'/f'{raw_frame:03d}_{cam_id}.png'
                source_image=Path(FRAMEWORK_DATA_ROOT)/str(spec.source['scene'])/'images'/f'{raw_frame:03d}_{cam_id}.jpg'
                with Image.open(mask_src) as im, Image.open(source_image) as recorded:
                    if im.size!=recorded.size: raise ValueError('source mask/recorded image size mismatch')
                    mask=im.convert('L').resize((image.shape[1],image.shape[0]),Image.Resampling.NEAREST)
                stem=f'{frame:03d}_{cam_id}'
                rgb_name=stem+'_framework.png';mask_name=stem+'_mask.png'
                Image.fromarray(image).save(args.output/rgb_name);mask.save(args.output/mask_name)
                depth_name=stem+'_depth.npy';np.save(args.output/depth_name,camera['depth'])
                records.append(dict(frame=frame,timestamp_s=frame/spec.fps,camera=name,camera_id=cam_id,
                    rgb=rgb_name,mask=mask_name,depth=depth_name,ego_to_world=pose.tolist(),
                    K=np.asarray(camera['K']).tolist(),camera_to_world=np.asarray(camera['camera_to_world']).tolist(),
                    mask_source=str(mask_src),mask_sha256=hashlib.sha256((args.output/mask_name).read_bytes()).hexdigest(),
                    mask_source_sha256=hashlib.sha256(mask_src.read_bytes()).hexdigest(),
                    rgb_sha256=hashlib.sha256((args.output/rgb_name).read_bytes()).hexdigest()))
        report=dict(schema='framework_weather_window_inputs_v1',records=records,
            camera_order=['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT'],frames=[80,81,82],
            source_checkpoint=str(spec.source['checkpoint']),source_manifest=str(JOB/'chain_repair/phase4/dynamic552_frame80/scene_manifest.json'),
            observation_source='FrameworkSceneBackend.render at explicitly supplied recorded poses',
            pose_source='recorded poses, not executed control feedback',scene_edits=[],
            mask_source='recorded dynamic masks at same camera/time/pose, resized nearest; not renderer-derived',
            mask_valid_for_novel_or_edited_poses=False,formal_evaluation_admitted=False)
        (args.output/'manifest.json').write_text(json.dumps(report,indent=2))
    finally: backend.close()

if __name__=='__main__':main()
