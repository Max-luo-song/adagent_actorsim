#!/usr/bin/env python3
"""Offline JSON trajectory -> render pack -> DiFix pseudo-GT -> appearance checkpoint."""

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
import importlib.util
import json
import os
pass  # Path is provided by the portable boundary above
import shutil
import sys
import numpy as np
from PIL import Image, ImageDraw
from contracts import HERE, ROOT, P26, sha, read, write, pose, split_frame


def implementation():
    return {str(p): sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.sh')}


def asset(path):
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    return dict(path=str(p), sha256=sha(p))


def verify_asset(value):
    if sha(value['path']) != value['sha256']:
        raise ValueError('Input changed: ' + value['path'])
    return Path(value['path'])


def image(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB')).copy()


def save_image(path, value):
    Image.fromarray(value).save(path)
    return asset(path)


def prepare(task_path, output):
    task = read(task_path)
    required = {'schema', 'scene_manifest', 'trajectory', 'renderer_bundle', 'teacher'}
    if not required <= task.keys() or task.keys() - required - {'frame_stride', 'epochs', 'learning_rate'}:
        raise ValueError('Missing/unknown task fields; edits/weather are not supported by v1')
    if task['schema'] != 'novel_view_distillation_v1':
        raise ValueError('Unsupported task schema')
    stride, epochs, lr = task.get('frame_stride', 3), task.get('epochs', 3), task.get('learning_rate', 1e-4)
    if type(stride) is not int or not 1 <= stride <= 10 or type(epochs) is not int or not 1 <= epochs <= 30:
        raise ValueError('Invalid sample stride/epochs')
    if isinstance(lr, bool) or not np.isfinite(lr) or not 0 < lr <= 1e-3:
        raise ValueError('Invalid learning rate')
    from framework import bootstrap
    bootstrap(task['renderer_bundle'])
    from scene_api import SceneSpec
    from ego_trajectory import compile_trajectory
    spec = SceneSpec.read(task['scene_manifest'])
    if spec.source.get('camera_ids') != [0, 1, 2]:
        raise ValueError('First adapter expects three Waymo cameras')
    trajectory = compile_trajectory(read(task['trajectory']), spec)
    if len(trajectory['samples']) < 10:
        raise ValueError('At least ten frames are needed for train/hold-out separation')
    teacher = task['teacher']
    if set(teacher) != {'src', 'weights', 'python'}:
        raise ValueError('Teacher requires local src, weights, python')
    for path in [Path(teacher['src'])/'pipeline_difix.py', Path(teacher['weights'])/'model_index.json', Path(teacher['python'])]:
        if not path.is_file():
            raise FileNotFoundError(path)
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError('Refusing to overwrite an existing job: ' + str(out))
    inputs = {name: asset(value) for name, value in dict(task=task_path,
        scene_manifest=spec.manifest, source_report=spec.source_report,
        checkpoint=spec.source['checkpoint'], config=spec.source['source_config'],
        trajectory=task['trajectory'], renderer_manifest=Path(task['renderer_bundle'])/'snapshot_manifest.json',
        teacher_code=Path(teacher['src'])/'pipeline_difix.py', teacher_config=Path(teacher['weights'])/'model_index.json').items()}
    # Pin all local teacher model files; a changed teacher must create a new job.
    teacher_pins = {str(p.resolve()): sha(p) for p in Path(teacher['weights']).rglob('*') if p.is_file()}
    if not any(p.endswith(('.bin', '.safetensors')) for p in teacher_pins):
        raise ValueError('Teacher weights missing')
    splits = [split_frame(s['source_frame'], stride) for s in trajectory['samples']]
    if not {'train', 'eval'} <= set(splits):
        raise ValueError('No train or held-out frames')
    out.mkdir(parents=True)
    write(out/'compiled_trajectory.json', trajectory)
    write(out/'job.json', dict(schema='novel_view_distillation_job_v1', task=task, inputs=inputs,
        teacher_pins=teacher_pins, implementation=implementation(), trajectory_sha256=sha(out/'compiled_trajectory.json'),
        frame_stride=stride, epochs=epochs, learning_rate=lr, edits_applied=False,
        pseudo_gt_is_truth=False, online_training=False, controller_modified=False))
    return dict(prepared=True, output=str(out), samples_by_split={s:splits.count(s)*3 for s in set(splits)}, gpu_executed=False)


def job_at(output):
    out = Path(output).resolve()
    job = read(out/'job.json')
    if job['implementation'] != implementation():
        raise ValueError('Pipeline code changed; prepare a new job')
    for v in job['inputs'].values():
        verify_asset(v)
    if sha(out/'compiled_trajectory.json') != job['trajectory_sha256']:
        raise ValueError('Compiled trajectory changed')
    return out, job


def validate_rows(rows):
    if not rows or not {'train','eval'} <= {r['split'] for r in rows}:
        raise ValueError('Training and held-out rows required')
    ids = set()
    split_by_frame = {}
    provenance = set()
    for row in rows:
        identity = (row['source_frame'], row['camera_id'], row['kind'])
        if identity in ids or row['split'] not in ('train','eval'):
            raise ValueError('Duplicate row/invalid split')
        ids.add(identity)
        if split_by_frame.setdefault(row['source_frame'], row['split']) != row['split']:
            raise ValueError('Camera/time train-test leakage')
        provenance.add((row['scene'], row['checkpoint_sha256'], row['edit_revision']))
        if row['edit_revision'] != 0:
            raise ValueError('Edited observations require a separate mask/replay adapter')
        pose(row['ego_to_world']); pose(row['camera_to_world'])
        k = np.asarray(row['K'])
        if k.shape != (3,3) or not np.isfinite(k).all() or k[0,0] <= 0 or k[1,1] <= 0:
            raise ValueError('Invalid camera intrinsics')
        for field in ['before', 'reference', 'mask']:
            verify_asset(row[field])
        if image(row['before']['path']).shape != image(row['reference']['path']).shape:
            raise ValueError('Reference image shape mismatch')
    if len(provenance) != 1:
        raise ValueError('Mixed scene/checkpoint/revision')


def render_pack(output):
    import torch
    from framework import Renderer
    out, job = job_at(output)
    stage = out/'render'
    stage.mkdir(exist_ok=False)
    renderer = Renderer(job['inputs']['scene_manifest']['path'], job['task']['renderer_bundle'])
    rows = []
    try:
        samples = read(out/'compiled_trajectory.json')['samples']
        for sample in samples:
            frame = sample['source_frame']; split = split_frame(frame, job['frame_stride'])
            if split == 'unused':
                continue
            for cam in [0,1,2]:
                reference, raw_path, original_camera_pose = renderer.original(frame, cam)
                camera = renderer.backend.cameras_data[cam]
                original_ego = original_camera_pose @ np.linalg.inv(np.asarray(camera.cam_to_ego))
                for kind, ego in [('novel', sample['ego_to_world']), ('original', original_ego)]:
                    key = f'{frame:04d}_{cam}_{kind}'
                    with torch.no_grad():
                        data = renderer.frame(ego, frame, cam)
                        result = renderer.rgb(data, layers=True)
                        before = (result['rgb'].clamp(0,1).cpu().numpy()*255).round().astype('uint8')
                        mask = renderer.valid_mask(result, data).cpu().numpy()
                    row = dict(id=key, source_frame=frame, timestamp_s=sample['timestamp_s'],
                        scene=renderer.spec.source['scene'], checkpoint_sha256=job['inputs']['checkpoint']['sha256'],
                        edit_revision=0, camera_id=cam, kind=kind, split=split,
                        ego_to_world=np.asarray(ego).tolist(), camera_to_world=data['cam_infos']['camera_to_world'].cpu().tolist(),
                        K=data['cam_infos']['intrinsics'].cpu().tolist(),
                        before=save_image(stage/f'{key}_before.png', before),
                        reference=save_image(stage/f'{key}_ref.png', reference),
                        raw_reference=asset(raw_path),
                        mask=save_image(stage/f'{key}_mask.png', mask.astype('uint8')*255))
                    rows.append(row)
                print(f'Rendered source frame {frame}, camera {cam}', flush=True)
        validate_rows(rows)
        write(stage/'manifest.json', dict(completed=True, rows=rows, revision=0, weather='original',
            job=asset(out/'job.json'), trajectory=asset(out/'compiled_trajectory.json'), gpu_executed=True))
    finally:
        renderer.close()
    return dict(rendered_rows=len(rows), manifest=str(stage/'manifest.json'))


def filter_target(before, fixed, mask, maximum_change=.35):
    if before.shape != fixed.shape or before.shape[:2] != mask.shape:
        raise ValueError('Teacher changed image resolution')
    delta = np.mean(np.abs(before.astype(float)-fixed.astype(float)), axis=-1)/255.
    valid = mask & (delta <= maximum_change)
    coverage = float(valid.mean())
    return valid, dict(valid_fraction=coverage, mean_rgb_change=float(delta.mean()),
                       accepted=bool(coverage >= .05), semantics_certified=False)


def pseudo_gt(output):
    import torch
    out, job = job_at(output)
    teacher = job['task']['teacher']
    for path, expected in job['teacher_pins'].items():
        if sha(path) != expected:
            raise ValueError('Teacher weights changed: '+path)
    stage = out/'pseudo_gt'; stage.mkdir(exist_ok=False)
    rows = read(out/'render/manifest.json')['rows']; validate_rows(rows)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for DiFix pseudo-GT')
    from teacher import load
    pipe = load(teacher)
    refined = []
    for index, row in enumerate(rows):
        r = dict(row)
        before = image(row['before']['path']); reference = image(row['reference']['path'])
        mask = image(row['mask']['path'])[...,0] > 0
        if row['kind'] == 'novel':
            generator = torch.Generator(device='cuda').manual_seed(20261108 + row['source_frame']*3 + row['camera_id'])
            with torch.inference_mode():
                fixed = pipe('remove degradation', image=Image.fromarray(before), ref_image=Image.fromarray(reference),
                    height=before.shape[0], width=before.shape[1], num_inference_steps=1, timesteps=[199],
                    guidance_scale=0., generator=generator).images[0]
            target = np.asarray(fixed.convert('RGB'))
            valid, quality = filter_target(before, target, mask)
            r.update(target_kind='difix_pseudo_gt', **quality)
        else:
            target = reference; valid = mask
            r.update(target_kind='real_original_gt', accepted=bool(valid.mean() >= .05), valid_fraction=float(valid.mean()))
        r.update(target=save_image(stage/f'{r["id"]}_target.png', target),
                 valid_mask=save_image(stage/f'{r["id"]}_valid.png', valid.astype('uint8')*255))
        refined.append(r)
        print(f'Pseudo-GT {index+1}/{len(rows)}: {r["id"]}, accepted={r["accepted"]}', flush=True)
    for kind in ['novel','original']:
        for split in ['train','eval']:
            if not any(r['accepted'] and r['kind']==kind and r['split']==split for r in refined):
                raise ValueError('No accepted samples for '+kind+'/'+split)
    write(stage/'manifest.json', dict(completed=True, render_manifest=asset(out/'render/manifest.json'),
        rows=refined, teacher=teacher, teacher_weights_verified=True,
        identity_fallback=False, pseudo_gt_is_truth=False, gpu_executed=True))
    return dict(rows=len(refined), accepted=sum(r['accepted'] for r in refined))


def target_rows(out):
    manifest = read(out/'pseudo_gt/manifest.json')
    verify_asset(manifest['render_manifest'])
    rendered = read(manifest['render_manifest']['path'])
    if rendered.get('completed') is not True:
        raise ValueError('Incomplete render pack')
    for key in ['job','trajectory']:
        verify_asset(rendered[key])
    if Path(rendered['job']['path']) != out/'job.json':
        raise ValueError('Render pack belongs to another job')
    if manifest.get('completed') is not True or manifest.get('identity_fallback') is not False:
        raise ValueError('Actual DiFix pseudo-GT manifest required')
    rows = manifest['rows']; validate_rows(rows)
    for r in rows:
        for f in ['target','valid_mask']:
            verify_asset(r[f])
        if image(r['before']['path']).shape != image(r['target']['path']).shape:
            raise ValueError('Target shape mismatch')
    return rows


def distill(output):
    import torch
    from framework import Renderer
    from contracts import appearance_parameters, masked_l1, merge_checkpoint, check_checkpoint_invariants
    out, job = job_at(output)
    rows = target_rows(out)
    stage = out/'distill'; stage.mkdir(exist_ok=False)
    renderer = Renderer(job['inputs']['scene_manifest']['path'], job['task']['renderer_bundle'])
    losses = []
    try:
        selected = appearance_parameters(renderer.trainer)
        optimizer = torch.optim.Adam(list(selected.values()), lr=job['learning_rate'])
        groups = {kind: [r for r in rows if r['split']=='train' and r['accepted'] and r['kind']==kind] for kind in ['novel','original']}
        if not all(groups.values()):
            raise ValueError('Both novel pseudo-GT and original real GT replay required')
        rng = np.random.default_rng(20261108)
        for epoch in range(job['epochs']):
            # One novel pass, two real-GT replay passes. Held-out rows are never optimized.
            for kind in ['novel','original','original']:
                for i in rng.permutation(len(groups[kind])):
                    row = groups[kind][i]
                    data = renderer.frame(row['ego_to_world'], row['source_frame'], row['camera_id'])
                    pred = renderer.rgb(data)['rgb'].clamp(0,1)
                    target = torch.tensor(image(row['target']['path']), device=pred.device, dtype=pred.dtype)/255.
                    mask = torch.tensor(image(row['valid_mask']['path'])[...,0]>0, device=pred.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = masked_l1(pred, target, mask)
                    if not loss.requires_grad:
                        raise RuntimeError('Renderer detached appearance gradients')
                    loss.backward()
                    grads = [p.grad for p in selected.values() if p.grad is not None]
                    if not grads or any(not torch.isfinite(g).all() for g in grads):
                        raise RuntimeError('Missing/nonfinite appearance gradients')
                    optimizer.step()
                    losses.append(dict(epoch=epoch, sample=row['id'], kind=kind, split='train', loss=float(loss.detach())))
                write(stage/'progress.json', dict(epoch=epoch, kind=kind, steps=len(losses), last_loss=losses[-1]))
                print(f'Distill epoch {epoch+1}/{job["epochs"]}: {kind}, steps={len(losses)}', flush=True)
        base = torch.load(job['inputs']['checkpoint']['path'], map_location='cpu', weights_only=False)
        merged = merge_checkpoint(base, selected)
        candidate = stage/'checkpoint_distilled.pth'
        torch.save(merged, candidate)
        original = torch.load(job['inputs']['checkpoint']['path'], map_location='cpu', weights_only=False)
        changed = check_checkpoint_invariants(original, torch.load(candidate, map_location='cpu', weights_only=False))
        shutil.copy2(job['inputs']['config']['path'], stage/'config.yaml')
        source = read(job['inputs']['source_report']['path'])
        source.update(checkpoint=str(candidate), source_config=str(stage/'config.yaml'))
        write(stage/'source_report.json', source)
        scene = read(job['inputs']['scene_manifest']['path'])
        scene['source_report'] = str(stage/'source_report.json')
        # Resolve external relative configs before moving the scene manifest.
        if not Path(scene['vehicle_config']).is_absolute():
            scene['vehicle_config'] = str((Path(job['inputs']['scene_manifest']['path']).parent/scene['vehicle_config']).resolve())
        write(stage/'scene_manifest.json', scene)
        write(stage/'receipt.json', dict(completed=True, candidate=asset(candidate),
            pseudo_manifest=asset(out/'pseudo_gt/manifest.json'), changed_weights=changed,
            geometry_and_dynamic_checkpoint_state_preserved=True, losses=losses,
            trained_sample_ids=sorted({r['sample'] for r in losses}), evaluation_admitted=False,
            closed_loop_driving=False, online_training=False, gpu_executed=True))
    finally:
        renderer.close()
    return dict(checkpoint=str(candidate), changed_weights=changed, steps=len(losses))


def evaluate(output):
    import torch
    from framework import Renderer
    out, job = job_at(output)
    rows = target_rows(out); receipt = read(out/'distill/receipt.json')
    verify_asset(receipt['candidate']); verify_asset(receipt['pseudo_manifest'])
    stage = out/'evaluation'; stage.mkdir(exist_ok=False)
    renderer = Renderer(out/'distill/scene_manifest.json', job['task']['renderer_bundle'])
    metrics = []; panels = []
    try:
        for row in rows:
            if not row['accepted'] or row['split'] != 'eval':
                continue
            if row['id'] in receipt['trained_sample_ids']:
                raise ValueError('Held-out sample appeared in optimizer receipt')
            with torch.no_grad():
                data = renderer.frame(row['ego_to_world'], row['source_frame'], row['camera_id'])
                after = renderer.rgb(data)['rgb'].clamp(0,1).cpu().numpy()
            target = image(row['target']['path']).astype(float)/255.
            before = image(row['before']['path']).astype(float)/255.
            mask = image(row['valid_mask']['path'])[...,0] > 0
            def measure(rgb):
                error = rgb[mask]-target[mask]
                return dict(l1=float(np.abs(error).mean()), psnr=float(-10*np.log10(max(float((error**2).mean()),1e-12))))
            metrics.append(dict(id=row['id'], kind=row['kind'], source_frame=row['source_frame'],
                target_kind=row['target_kind'], before=measure(before), after=measure(after)))
            saved = save_image(stage/f'{row["id"]}_after.png', (after*255).round().astype('uint8'))
            if row['kind']=='novel':
                panels.append((row, saved['path']))
        averages = {kind: {when: {metric: float(np.mean([r[when][metric] for r in metrics if r['kind']==kind]))
                     for metric in ['l1','psnr']} for when in ['before','after']} for kind in ['novel','original']}
        original_drop = averages['original']['before']['psnr']-averages['original']['after']['psnr']
        improved = averages['novel']['after']['l1'] < averages['novel']['before']['l1']
        # Automatic screen only; pseudo-GT similarity is not an independent fidelity certificate.
        passed = bool(improved and original_drop <= .5)
        write(stage/'report.json', dict(completed=True, metrics=metrics, averages=averages,
            novel_teacher_similarity_improved=improved, original_psnr_drop_db=original_drop,
            automatic_screen_passed=passed, maximum_original_psnr_drop_db=.5,
            visual_quality_certified=False, temporal_consistency_certified=False,
            closed_loop_validation_required=True, formal_evaluation_admitted=False,
            checkpoint=receipt['candidate'], scene_manifest=str(out/'distill/scene_manifest.json')))
        make_comparison(panels, stage, read(out/'compiled_trajectory.json')['fps']/5.)
    finally:
        renderer.close()
    return dict(report=str(stage/'report.json'), automatic_screen_passed=passed, closed_loop_validation_required=True)


def make_comparison(panels, stage, fps):
    # One sheet per held-out source frame: columns three cameras, rows before/teacher/after.
    import subprocess
    import imageio_ffmpeg
    frames = stage/'comparison_frames'; frames.mkdir()
    grouped = {}
    for row, after in panels:
        grouped.setdefault(row['source_frame'], {})[row['camera_id']] = (row, after)
    if not grouped:
        raise ValueError('No held-out comparison frames')
    for index, (source_frame, cams) in enumerate(sorted(grouped.items())):
        sheet = Image.new('RGB', (1440, 3*300+100), '#151515'); draw = ImageDraw.Draw(sheet)
        draw.text((8,8), f'Held-out frame {source_frame} | BEFORE / DiFix PSEUDO-GT / AFTER | visual fidelity unverified', fill='white')
        for col, cam in enumerate([1,0,2]):
            row, after = cams[cam]
            for line, field in enumerate([row['before']['path'],row['target']['path'],after]):
                im = Image.open(field).convert('RGB'); im.thumbnail((480,280))
                sheet.paste(im,(col*480,40+line*300)); im.close()
        sheet.save(frames/f'{index:05d}.png')
    # Held-out cadence is five source frames; keep source timing in this comparison.
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-n','-framerate',str(fps),'-i',str(frames/'%05d.png'),
        '-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(stage/'before_teacher_after.mp4')],check=True)


def dependencies(task_path):
    import subprocess
    task = read(task_path)
    teacher = task['teacher']
    probe = 'import sys,json;sys.path.insert(0,sys.argv[1]);from teacher import components;components(json.loads(sys.argv[2]));import torch;print("teacher_import_ok",torch.cuda.is_available())'
    result = subprocess.run([teacher['python'],'-B','-c',probe,str(HERE),json.dumps(teacher)],capture_output=True,text=True)
    import torch
    return dict(renderer_cuda_available=torch.cuda.is_available(), teacher_import_ok=result.returncode==0,
        teacher_probe_stdout=result.stdout[-1000:], teacher_probe_stderr=result.stderr[-2000:],
        teacher_weights_present=(Path(teacher['weights'])/'model_index.json').is_file(), gpu_executed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check','prepare','render','pseudo-gt','distill','evaluate'])
    parser.add_argument('--task'); parser.add_argument('--output')
    args = parser.parse_args()
    if args.stage in ('check','prepare') and not args.task:
        parser.error('--task required')
    if args.stage != 'check' and not args.output:
        parser.error('--output required')
    actions = {'check':lambda:dependencies(args.task), 'prepare':lambda:prepare(args.task,args.output),
        'render':lambda:render_pack(args.output), 'pseudo-gt':lambda:pseudo_gt(args.output),
        'distill':lambda:distill(args.output), 'evaluate':lambda:evaluate(args.output)}
    result = actions[args.stage]()
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.stage=='check' and not result['teacher_import_ok']:
        raise SystemExit(2)
    if args.stage=='evaluate' and not result['automatic_screen_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
