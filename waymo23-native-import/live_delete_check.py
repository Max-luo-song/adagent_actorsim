"""Pre-episode rigid deletion persisted across reset/step; no source file writes."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from live_env import make_env


def apply_delete(env, instance, edit_id=None, preview_only=False):
    state = env.models['RigidNodes']
    if not 0 <= instance < state['instances_fv'].shape[1]:
        raise ValueError('invalid rigid instance ID')
    from edit_trust_adapter import require_edit_mutation, swept_instance_bounds
    require_edit_mutation(env, "erase", dict(old_bounds=swept_instance_bounds(state, instance)), edit_id, preview_only)
    state['instances_fv'][:, instance] = False
    env.cached_frame = None
    env.frame_cache = None


def target_points(env, instance, frame):
    ids = env.models['RigidNodes']['points_ids'].flatten().long()
    return sum(int((ids[p['idx']] == instance).sum().item())
               for p in env.geometry(frame) if p['name'] == 'RigidNodes')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=int, default=10)
    parser.add_argument('--steps', type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.steps <= 30:
        raise ValueError('steps must be 1..30')
    base = Path(os.environ['BASE'])
    checkpoint = ('/inspire/hdd/global_user/guoluosong-253108120129/adscenegraph/'
                  'Omniscenegraph/tmp/outputs/waymo/waymo_23/'
                  '20260418_lidar+cam0_1_2adstudiov_pbr_3view/checkpoint_final.pth')
    out = base/'outputs'/('waymo23_live_delete_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    env = make_env(base/'code/HUGSIM', checkpoint, base/'waymo23-inputs')
    action = {'acc': 1., 'steer_rate': .15}
    try:
        fv = env.models['RigidNodes']['instances_fv']
        original = fv.clone()
        if not 0 <= args.instance < fv.shape[1]:
            raise ValueError('invalid rigid instance ID')
        before, states, counts = [], [], []
        obs, info = env.reset()
        for step in range(args.steps+1):
            before.append(obs['rgb']['CAM_FRONT'].copy())
            states.append(info['ego_pos'])
            counts.append(target_points(env, args.instance, info['source_frame']))
            if step < args.steps:
                obs, _, _, _, info = env.step(action)
        if not any(counts):
            raise RuntimeError('target has no baseline geometry in tested frames')
        apply_delete(env, args.instance, preview_only=True)
        obs, info = env.reset()
        pairs, changes = [], []
        for step in range(args.steps+1):
            if not np.allclose(states[step], info['ego_pos'], atol=1e-8, rtol=0):
                raise RuntimeError('before/after ego states differ')
            if target_points(env, args.instance, info['source_frame']) != 0:
                raise RuntimeError('deleted geometry returned')
            after = obs['rgb']['CAM_FRONT'].copy()
            if step == 0:
                edited_reset = after.copy()
            changed = int((np.abs(before[step].astype(np.int16)-after.astype(np.int16)).max(2)>5).sum())
            changes.append(changed)
            left, right = Image.fromarray(before[step]), Image.fromarray(after)
            canvas = Image.new('RGB', (left.width*2, left.height+24), 'white')
            canvas.paste(left, (0, 24)); canvas.paste(right, (left.width, 24))
            draw = ImageDraw.Draw(canvas)
            draw.text((5, 5), f'Before | step {step}', fill='black')
            draw.text((left.width+5, 5), f'After: remove rigid {args.instance}', fill='black')
            pairs.append(canvas)
            if step in {0, args.steps//2, args.steps}:
                canvas.save(out/f'comparison_{step:03d}.png')
            print(f'step {step}: target points {counts[step]} -> 0; changed pixels {changed}', flush=True)
            if step < args.steps:
                obs, _, _, _, info = env.step(action)
        obs, _ = env.reset()
        if not np.array_equal(edited_reset, obs['rgb']['CAM_FRONT']):
            raise RuntimeError('edited reset image is not repeatable')
        other = [i for i in range(fv.shape[1]) if i != args.instance]
        if not torch.equal(fv[:, other], original[:, other]):
            raise RuntimeError('other instance visibility changed')
        pairs[0].save(out/'edit_comparison.gif', save_all=True, append_images=pairs[1:], duration=100, loop=0)
        fv.copy_(original)
        obs, _ = env.reset()
        restored = np.array_equal(before[0], obs['rgb']['CAM_FRONT'])
        report = dict(instance=args.instance, checkpoint_loads=env.checkpoint_loads,
                      frames_checked=len(pairs), before_target_points=counts,
                      after_target_points=[0]*len(pairs), changed_pixels=changes,
                      identical_ego_states=True, edit_persists_after_reset=True,
                      other_instance_visibility_unchanged=True, restoration_matches_baseline=restored,
                      visual_effect_detected=max(changes)>=20,
                      edit_stage='before_episode', original_checkpoint_written=False,
                      collision_and_scoring_validated=False)
        (out/'edit_report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2)); print('OUTPUT:', out)
        if not restored or env.checkpoint_loads != 1 or max(changes)<20:
            raise RuntimeError('restoration/load-count/visible-effect check failed; see report')
    finally:
        env.close()


if __name__ == '__main__':
    main()
