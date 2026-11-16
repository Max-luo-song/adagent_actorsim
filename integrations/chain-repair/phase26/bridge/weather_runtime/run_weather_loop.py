"""Short executed-pose weather integration using the existing framework harness."""

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
pass  # Path is provided by the portable boundary above
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent
sys.path.insert(0, str(BRIDGE / 'framework_backend'))
import run_arm_fwbackend as fw
sys.path.insert(0, str(BRIDGE / 'unified'))
from scene_edit_session import UnifiedSceneSession
from edit_api import read_command, native_request, from_native_request
from runtime_backend import RuntimeWeatherBackend
from input_receipts import InputReceiptRecorder
from mask_backend import MaskFrameworkBackend
from native_edit_probe import probe_native_edit
from native_rotation_probe import probe_native_rotation
from managed_asset_probe import probe_managed_asset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=5,
                   help='Executed controls at 10Hz, frame80 onward; default is the existing 5-step gate')
    edit_options = p.add_mutually_exclusive_group()
    edit_options.add_argument('--native-edit', type=Path)
    edit_options.add_argument('--edit-command', type=Path)
    p.add_argument('--articulation-report', type=Path)
    p.add_argument('--verify-edit-lifecycle', action='store_true')
    args = p.parse_args()
    if not 1 <= args.steps <= 117:
        p.error('--steps must be between 1 and 117 for scene552 frame80')
    return run_episode(args)


def run_episode(args,*,inference_service=None,scene_host=None,scene_manifest=None):
    """The normal driving path, optionally borrowing shared scene/model owners."""
    if not 1 <= args.steps <= 117:raise ValueError('steps must be 1..117')
    command = (read_command(_hugsim_expand(json.loads(args.edit_command.read_text()))) if args.edit_command else
               from_native_request(_hugsim_expand(json.loads(args.native_edit.read_text()))) if args.native_edit else None)
    if command and command['operation'] not in ('translate', 'rotate', 'insert'):
        raise ValueError('this loop starts from a fresh scene; use translate, rigid rotate or insert')
    edit = native_request(command) if command and command['operation'] in ('translate', 'rotate') else None
    managed = command if command and command['operation'] == 'insert' else None
    articulation = None
    if edit:
        if args.articulation_report is None:
            raise ValueError('native edit requires independent articulation GPU evidence')
        articulation = _hugsim_expand(json.loads(args.articulation_report.read_text()))
        if (articulation.get('schema') != 'dynamic_articulation_acceptance_v1'
                or articulation.get('passed') is not True
                or set(articulation.get('groups', {})) != {'SMPLNodes', 'DeformableNodes'}
                or not all(g.get('passed') is True for g in articulation['groups'].values())):
            raise ValueError('independent articulation gate failed')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    request = _hugsim_expand(json.loads((BRIDGE / 'framework_backend/requests/fastprobe_fwbackend.json').read_text()))
    if scene_manifest is not None:request['scene_manifest']=str(Path(scene_manifest).resolve())
    request.update(stop_on_stationary=False, complete_on_pass=False, steps=args.steps)
    request_path = out / 'weather_request.json'
    request_path.write_text(json.dumps(request, indent=2))
    sources = list(HERE.glob('*.py')) + [
        BRIDGE / 'weather_sequence/causal_buffer.py',
        BRIDGE / 'framework_backend/backend.py',
        BRIDGE / 'framework_backend/native_edit.py',
        BRIDGE / 'framework_backend/native_radiance.py',
        fw.ROOT / 'waymo23-native-import/asset_radiance.py',
        BRIDGE / 'framework_backend/run_arm_fwbackend.py',
        BRIDGE / 'unified/run_phase26.sh', BRIDGE / 'unified/edit_api.py',
        BRIDGE / 'unified/scene_edit_session.py']
    if inference_service is not None:
        sources += [BRIDGE/'unified/resident_inference.py',BRIDGE/'unified/shared_weather_scene.py',
                    BRIDGE/'fast_control/driver_state_audit.py']
    sources += [BRIDGE / 'framework_backend/managed_assets.py', BRIDGE / 'framework_backend/managed_evaluation.py']
    if managed:
        sources.append(args.edit_command)
    if edit:
        sources += [args.edit_command or args.native_edit, args.articulation_report,
                    BRIDGE / 'smpl_dynamic/validate_articulation.py']
    (out / 'implementation.json').write_text(json.dumps({str(f): hashlib.sha256(
        f.read_bytes()).hexdigest() for f in sources}, indent=2))
    records, packets = [], []
    edit_evidence = []
    journals = []
    lifecycle = []
    managed_probes = []
    registry_holder = []

    class WeatherSession:
        def __init__(self):
            self._session=scene_host.session if scene_host is not None else UnifiedSceneSession(
                backend_factory=lambda spec: RuntimeWeatherBackend(spec,out/'weather'))

        def __getattr__(self,name):return getattr(self._session,name)
        def __enter__(self):return self
        def __exit__(self,*args):self.close()

        def load_scene(self, manifest):
            if scene_host is None:self._session.load_scene(manifest)
            else:
                _,transition=scene_host.acquire(manifest,out/'weather')
                (out/'shared_scene_transition.json').write_text(json.dumps(transition,indent=2))
            (out / 'unified_objects_before.json').write_text(json.dumps(self.list_objects(), indent=2))
            (out / 'unified_capabilities.json').write_text(json.dumps(self.edit_capabilities(), indent=2))
            if edit:
                probe = probe_native_rotation if edit['operation'] == 'rotate' else probe_native_edit
                edit_evidence.append(probe(self.backend, edit,
                    out / ('native_rotation_probe' if edit['operation'] == 'rotate' else 'native_edit_probe'), MaskFrameworkBackend.render,
                    apply_edit=lambda legacy: self.edit(command)))
            if managed:
                managed_probes.append(probe_managed_asset(self, command,
                    out / 'managed_asset_probe', MaskFrameworkBackend.render))
                self.edit(command)
                registry_holder[:] = [self.backend.managed_assets]
                asset = self.backend.managed_assets.assets[command['object_id'].split(':',1)[1]]
                (out / 'managed_asset_sources.json').write_text(json.dumps(dict(
                    object_id=command['object_id'], asset=asset.path,
                    asset_to_world=asset.pose.tolist(), points=len(asset.local_means),
                    calibration=asset.calibration,
                    rendering='degree3 SH; shared full-scene Gaussian raster',
                    collision_proxy='same pose and Gaussian-support calibration; not physical mesh',
                    formal_evaluation_admitted=False), indent=2))
            return self.describe()

        def render(self,ego_pose,timestamp):
            packet = self._session.render(ego_pose,timestamp)
            packets.append(dict(timestamp=float(timestamp),
                edit_revision=packet['edit_revision'], edited=packet['edit_trust']['edited'],
                edit_interface=packet['edit_interface'],
                rgb_sha256={name: hashlib.sha256(c['rgb'].tobytes()).hexdigest()
                            for name, c in packet['cameras'].items()}))
            return packet

        def close(self):
            if self.backend is not None:
                records[:] = self.backend.weather_records
                if self._editor is not None:
                    journals[:] = self.edit_journal()
                    if scene_host is not None or (args.verify_edit_lifecycle and command and self.editor.journal):
                        previous_generation = self.generation
                        originals = dict(self.editor.originals)
                        original_q = dict(self.editor.original_quaternions)
                        original_sh = dict(self.editor.original_radiance)
                        self.reset()
                        lifecycle.append(dict(
                            native_positions_restored=all(self.backend.torch.equal(
                                self.backend.trainer.models[group].instances_trans[:, index].detach(), value)
                                for (group, index), value in originals.items()),
                            native_rotations_restored=all(self.backend.torch.equal(
                                self.backend.trainer.models[group].instances_quats[:, index].detach(), value)
                                for (group, index), value in original_q.items()),
                            native_radiance_restored=all(self.backend.torch.equal(
                                self.backend.trainer.models[group]._features_rest[state[0]].detach(), state[1])
                                for (group, index), state in original_sh.items()),
                            revision_cleared=self.revision == 0,
                            new_generation=self.generation != previous_generation,
                            edit_journal_cleared=not self.edit_journal(),
                            managed_assets_cleared=not self.backend.managed_assets.assets and not self.objects,
                            no_observation_after_reset=self.backend._last is None,
                            weather_history_reset=self.backend.last_weather_reset_verified))
            if scene_host is None:self._session.close()
            else:scene_host.release()

    import paired_closed_loop_arm as harness
    original = harness.SceneSession
    original_pickle = harness.atomic_pickle
    record_input = InputReceiptRecorder(original_pickle, out)
    receipts = record_input.receipts
    harness.SceneSession = WeatherSession
    harness.atomic_pickle = record_input
    import native_actor_evaluation as actor_module
    original_actor = actor_module.NativeActorEvaluation
    if managed:
        from managed_evaluation import ManagedActorEvaluation
        actor_module.NativeActorEvaluation = lambda groups: ManagedActorEvaluation(original_actor(groups), registry_holder[0])
    try:
        if inference_service is None:fw.run(str(request_path), str(out / 'closed_loop'))
        else:fw.run(str(request_path), str(out / 'closed_loop'),inference_service=inference_service)
    finally:
        harness.SceneSession = original
        harness.atomic_pickle = original_pickle
        actor_module.NativeActorEvaluation = original_actor
        (out / 'weather_packet_audit.json').write_text(json.dumps(packets, indent=2))
        (out / 'unified_edit_journal.json').write_text(json.dumps(journals, indent=2))
        if args.verify_edit_lifecycle:
            (out / 'unified_reset_audit.json').write_text(json.dumps(lifecycle, indent=2))
        if edit:
            (out / 'native_edit_evidence.json').write_text(json.dumps(edit_evidence, indent=2))
    report = _hugsim_expand(json.loads((out / 'closed_loop/run_report.json').read_text()))
    observations = report.get('observations', [])
    checks = dict(
        completed=report.get('status') == 'completed',
        **{('five_executed_controls' if args.steps == 5 else 'requested_controls_executed'):
           len(report.get('trace', [])) == args.steps,
           ('six_weather_observations' if args.steps == 5 else 'requested_weather_observations'):
           len(records) == len(observations) == len(packets) == args.steps + 1},
        control_feedback=report.get('control_actions_executed') is True,
        no_recorded_future=report.get('recorded_future_ego_used_for_closed_motion') is False,
        no_formal_admission=report.get('formal_evaluation_admitted') is False,
        executed_pose_matches_weather=bool(records) and len(records) == len(observations) and all(
            np.allclose(r['ego_to_world'], o['closed_ego_to_world'], atol=1e-6, rtol=0)
            and abs(r['timestamp'] - o['timestamp']) < 1e-6 for r, o in zip(records, observations)),
        historical_only=bool(records) and all(not r['audit']['future_frames_used'] and
            max(r['audit']['window_timestamps']) <= r['timestamp'] for r in records),
        cameras_aligned=bool(records) and all(np.allclose(
            np.asarray(r['ego_to_world']) @ np.asarray(c['camera_to_ego']),
            c['camera_to_world'], atol=1e-6, rtol=0) for r in records for c in r['cameras'].values()),
        dynamic_rgb_preserved=bool(records) and all(c['actor_rgb_preserved']
            for r in records for c in r['cameras'].values()),
        geometry_depth_unchanged=bool(records) and all(c['depth_unchanged']
            for r in records for c in r['cameras'].values()),
        weather_changes_visible=bool(records) and all(any(c['changed_pixels'] > 0
            for c in r['cameras'].values()) for r in records),
        packet_contains_weather=len(records) == len(packets) and bool(records) and all(
            all(c['rgb_sha256'] == p['rgb_sha256'][name] for name, c in r['cameras'].items())
            for r, p in zip(records, packets)))
    # The harness deletes observation.pkl after steps1+; record each successfully
    # serialized payload before deletion, and require successful worker replies.
    worker_inputs = []
    for i, record in enumerate(records[:args.steps]):
        receipt = receipts.get(f'step_{i:03d}', {})
        done = _hugsim_expand(json.loads((out / f'closed_loop/uniad_worker/done_{i:03d}.json').read_text()))
        worker_inputs.append(done.get('status') == 'ok' and
            abs(receipt.get('timestamp', -1) - record['timestamp']) < 1e-6 and
            all(receipt.get('rgb_sha256', {}).get(name) == c['rgb_sha256']
                for name, c in record['cameras'].items()))
    checks['uniad_inputs_are_weather_rgb'] = len(worker_inputs) == args.steps and all(worker_inputs)
    if edit:
        checks.update(independent_body_motion=articulation['passed'],
            unified_edit_command_used=len(journals) == 1 and journals[0]['command'] == command
                and all(p['edit_interface'] == 'scene_edit_v1' for p in packets),
            native_edit_visible_and_proxy_synchronized=bool(edit_evidence) and edit_evidence[0]['passed'],
            native_edit_revision_consistent=len(packets) == args.steps + 1 and all(
                p['edit_revision'] == 1 and p['edited'] for p in packets),
            weather_history_after_edit=bool(records) and records[0]['audit']['window_frames'] == [80,80,80]
                and len({r['audit']['epoch'] for r in records}) == 1)
    if managed:
        name = command['object_id'].split(':', 1)[1]
        checks.update(
            unified_asset_insert_used=len(journals) == 1 and journals[0]['command'] == command,
            insert_move_remove_smoke=bool(managed_probes) and managed_probes[0]['passed'],
            managed_gaussians_collected=len(records) == args.steps + 1 and all(
                r['managed_objects'] == [name] and r['managed_gaussians_in_last_collection'] > 0 for r in records),
            managed_collision_proxy_checked=len(observations) == args.steps + 1 and all(any(
                row['object_id'] == command['object_id'] for row in o['all_actor_reference'].get('managed_reference', []))
                for o in observations),
            managed_revision_consistent=len(packets) == args.steps + 1 and all(p['edit_revision'] == 1 and p['edited'] for p in packets))
    if args.verify_edit_lifecycle:
        checks['unified_reset_restores_scene_and_history'] = len(lifecycle) == 1 and all(lifecycle[0].values())
    if inference_service is not None:
        ready=report.get('worker_ready',{})
        checks['resident_model_used']=report.get('inference_service')=='resident_across_driving_episodes' and ready.get('uniad_model_loads')==1
        checks['fresh_driver_state']=all(report.get('driver_state_initial',{}).get('checks',{}).values()) and bool(report.get('driver_state_initial'))
    acceptance = dict(checks=checks, passed=all(checks.values()),
        formal_evaluation_admitted=False, diagnostic_uncertified=True,
        safety_equivalence_to_mpc=False,
        scope=('managed rigid asset insert' if managed else
               f"native {edit['group']}:{edit['instance']} {edit['operation']}" if edit else 'unedited') + f'; frame80; {args.steps} iLQR control steps; rainy RGB style',
        native_edit=edit,
        unified_edit_command=command,
        temporal_quality_certified=False, physical_weather_geometry=False)
    (out / 'acceptance.json').write_text(json.dumps(acceptance, indent=2))
    (out / 'integration_summary.json').write_text(json.dumps(dict(
        control='iLQR tracking passthrough', native_edit=edit, managed_insert=managed,
        requested_controls=args.steps, requested_simulation_seconds=args.steps * .1,
        pipeline='framework render -> aligned dynamic mask -> causal weather -> frozen UniAD -> iLQR -> Motion.advance',
        articulation_report=str(args.articulation_report) if edit else None,
        formal_evaluation_admitted=False, diagnostic_uncertified=True), indent=2))
    if not acceptance['passed']:
        raise SystemExit('weather integration acceptance failed; inspect acceptance.json')
    return acceptance


if __name__ == '__main__':
    main()
