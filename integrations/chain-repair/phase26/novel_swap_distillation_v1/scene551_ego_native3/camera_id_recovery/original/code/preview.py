"""Render paired views with exact temporary native actor translation edits."""

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

import contextlib,importlib.util
pass  # Path is provided by the portable boundary above
import numpy as np
from PIL import Image,ImageDraw
from contracts import read,write,sha

class TranslationSwap:
    def __init__(self,backend,program):
        self.backend=backend;self.model=backend.trainer.models['RigidNodes'];self.torch=backend.torch
        self.original=self.model.instances_trans[:,3].detach().clone()
        np.testing.assert_array_equal(self.original.cpu().numpy(),program['original_car_trans'])
        self.edited=self.original.new_tensor(program['car_trans'])
        if backend.env.models['RigidNodes']['instances_trans'] is not self.model.instances_trans:raise RuntimeError('Renderer and object-state tensors split')
        self.tests=set(map(int,backend.dataset.test_timesteps))
    @contextlib.contextmanager
    def at(self,frame,edited):
        with self.torch.no_grad():
            self.model.instances_trans[:,3].copy_(self.edited if edited else self.original)
            if edited and frame in self.tests and 1<frame<len(self.original)-1:
                # Match the native held-out root interpolator without moving other actors.
                self.model.instances_trans[frame-1,3].copy_(self.edited[frame])
                self.model.instances_trans[frame+1,3].copy_(self.edited[frame])
        try:yield
        finally:
            with self.torch.no_grad():self.model.instances_trans[:,3].copy_(self.original)

def run(out,plan):
    import torch,subprocess,imageio_ffmpeg
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no new-view renders executed')
    from workflow import BASE,asset
    spec=importlib.util.spec_from_file_location('_swap_renderer_provider',BASE/'regeneration_v5/workflow.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if module.family_pins()!=plan['provider_code']:raise ValueError('Conditioned renderer implementation changed')
    module.CONTEXT=plan['conditioning']
    task=plan['source_task'];renderer=module.ConditionedRenderer(task['scene_manifest'],task['renderer_bundle'])
    with np.load(out/'swap_program.npz') as z:program={k:z[k].copy() for k in z.files}
    stage=out/'preview';stage.mkdir();(stage/'frames').mkdir();records=[];cleanup=dict(closed=False,actor_restored=False)
    try:
        swap=TranslationSwap(renderer.backend,program)
        actor_hash_before={name:sha_tensor(m.instances_trans) for name,m in renderer.trainer.models.items() if hasattr(m,'instances_trans')}
        first_images={}
        for i,frame in enumerate(plan['frames']):
            canvas=Image.new('RGB',(1440,1060),'#151b22');draw=ImageDraw.Draw(canvas)
            draw.text((12,8),f'Scene551 | source frame={frame} t={frame/10:.1f}s | camera height from ego rig; vehicle height preserved',fill='white')
            draw.text((12,29),'A: ORIGINAL EGO + ORIGINAL NATIVE3                    B: EGO AT NATIVE3 PATH + NATIVE3 AT ORIGINAL EGO PATH',fill='white')
            draw.text((12,49),'Scene setup swapped from first frame; diagnostic new-view rendering, not a crossing maneuver or closed-loop driving',fill='white')
            gap=read(out/'geometry_report.json')['minimum_ego_target_footprint_gap_m']
            draw.text((12,70),f'Minimum ego/target footprint gap over clip: {gap:.2f} m',fill='#87e4bc')
            for row,cam in enumerate([1,0,2]):
                for col,edited in enumerate([False,True]):
                    pose=program['ego_poses'][i] if edited else program['original_ego'][i]
                    data=renderer.frame(pose,frame,cam)
                    with swap.at(frame,edited),torch.no_grad():rgb=renderer.rgb(data)['rgb'].clamp(0,1).cpu().numpy()
                    image=(rgb*255).round().astype(np.uint8)
                    path=stage/f'{frame:03d}_cam{cam}_{"swapped" if edited else "original"}.png';Image.fromarray(image).save(path)
                    if i==0:first_images[(cam,edited)]=image
                    im=Image.fromarray(image);im.thumbnail((720,290));canvas.paste(im,(col*720,130+row*300))
                    draw.text((col*720+8,108+row*300),f'camera {cam} | {"SWAPPED scene, revision 1" if edited else "original scene, revision 0"}',fill='white')
                    records.append(dict(source_frame=frame,camera_id=cam,variant='swapped' if edited else 'original',edit_revision=int(edited),ego_to_world=pose.tolist(),image=asset(path)))
            canvas.save(stage/'frames'/f'{i:04d}.png');write(stage/'progress.json',dict(completed_frames=i+1,total_frames=len(plan['frames'])))
            print(f'Swap new view {i+1}/{len(plan["frames"])} frame={frame}',flush=True)
        # Exact original state after every context and a repeated original first frame.
        unchanged={name:sha_tensor(renderer.trainer.models[name].instances_trans)==value for name,value in actor_hash_before.items()}
        restored_pixels=[]
        for cam in [0,1,2]:
            data=renderer.frame(program['original_ego'][0],plan['frames'][0],cam)
            with torch.no_grad():image=(renderer.rgb(data)['rgb'].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
            restored_pixels.append(np.array_equal(image,first_images[(cam,False)]))
        cleanup['actor_restored']=all(unchanged.values());cleanup['initial_pixels_restored']=all(restored_pixels)
        if not cleanup['actor_restored'] or not cleanup['initial_pixels_restored']:raise RuntimeError('Swap did not restore original scene')
        video=stage/'original_vs_position_swapped_3CAM.mp4';ff=imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ff,'-v','error','-n','-framerate','10','-i',str(stage/'frames/%04d.png'),'-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(video)],check=True)
        subprocess.run([ff,'-v','error','-threads','1','-i',str(video),'-f','null','-'],check=True)
        result=dict(schema='position_swap_new_view_result_v1',completed=True,gpu_executed=True,frames=len(plan['frames']),
            object_id='native:RigidNodes:3',both_ego_and_vehicle_positions_changed=True,
            original_scene_restored=True,other_actor_transforms_restored=unchanged,
            geometry_report=asset(out/'geometry_report.json'),video=asset(video),full_decode_passed=True,
            records=records,training_performed=False,closed_loop_driving=False,default_model_changed=False)
    finally:
        renderer.close();cleanup['closed']=True;write(stage/'cleanup.json',cleanup)
    write(stage/'report.json',result)
    return {k:v for k,v in result.items() if k!='records'}

def sha_tensor(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
