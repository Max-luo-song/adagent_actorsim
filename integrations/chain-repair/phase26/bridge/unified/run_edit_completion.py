"""New native edit coverage and real same-session lifecycle GPU validation."""

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
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent
sys.path.insert(0, str(BRIDGE / 'weather_runtime'))


def run_stage(command, log_path):
    """Retain full logs and show startup progress / the real error at terminal."""
    with log_path.open('x') as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith('Weather startup:'):
                    print(line.rstrip(), flush=True)
            returncode = process.wait()
    if returncode:
        tail = '\n'.join(log_path.read_text().splitlines()[-35:])
        raise RuntimeError(f'{log_path.stem} failed (exit {returncode}); log: {log_path}\n{tail}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--body-run', type=Path, default=HERE / 'edited_weather_runs/20261102T031621Z')
    p.add_argument('--switch-manifest', type=Path)
    p.add_argument('--lifecycle-only', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.lifecycle_only:
        from scene_edit_session import UnifiedSceneSession
        from runtime_backend import RuntimeWeatherBackend
        from mask_backend import MaskFrameworkBackend
        from lifecycle_probe import probe_lifecycle
        manifest = _hugsim_expand(json.loads((BRIDGE / 'framework_backend/requests/fastprobe_fwbackend.json').read_text()))['scene_manifest']
        commands = [_hugsim_expand(json.loads((HERE / f'native_{kind}_command.json').read_text()))
                    for kind in ('rigid', 'deformable')]
        count = []
        def factory(spec):
            path = args.output / f'weather_load_{len(count)}'
            count.append(path)
            return RuntimeWeatherBackend(spec, path)
        with UnifiedSceneSession(factory) as session:
            probe_lifecycle(session, manifest, commands, args.output / 'probe',
                            MaskFrameworkBackend.render, args.switch_manifest)
        return
    from reuse_articulation import verify
    reuse = verify(args.body_run)  # do not start an expensive run with invalid body evidence
    if args.switch_manifest:
        from scene_edit_session import saf
        target = saf.SceneSpec.read(args.switch_manifest)
        default_manifest = _hugsim_expand(json.loads((BRIDGE / 'framework_backend/requests/fastprobe_fwbackend.json').read_text()))['scene_manifest']
        current = saf.SceneSpec.read(default_manifest)
        if not any(current.source.get(k) != target.source.get(k)
                   for k in ('scene', 'checkpoint', 'inputs')):
            raise ValueError('--switch-manifest must identify a different scene source')
        if target.frames <= 80 or target.fps != 10.:
            raise ValueError('switch probe requires a 10Hz scene with frame80 available')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = list(HERE.glob('*.py')) + list(HERE.glob('native_*_command.json'))
    sources += list((BRIDGE / 'weather_runtime').glob('*.py'))
    sources += [HERE / 'run_phase26.sh']
    (out / 'implementation.json').write_text(json.dumps({str(f): hashlib.sha256(
        f.read_bytes()).hexdigest() for f in sources}, indent=2))
    (out / 'articulation_reuse.json').write_text(json.dumps(reuse, indent=2))
    # Each stage has its own process to release CUDA and avoid module collisions.
    # The parent shell holds the service lock throughout all three stages.
    results = {}
    try:
        for kind in ('rigid', 'deformable'):
            print(f'{kind}: starting new GPU edit/weather validation; log: {out / (kind + ".log")}', flush=True)
            cmd = [sys.executable, '-u', '-B', str(BRIDGE / 'weather_runtime/run_weather_loop.py'),
                   '--output', str(out / kind), '--edit-command', str(HERE / f'native_{kind}_command.json'),
                   '--verify-edit-lifecycle', '--articulation-report', str(args.body_run / 'articulation/report.json')]
            run_stage(cmd, out / f'{kind}.log')
            data = _hugsim_expand(json.loads((out / kind / 'acceptance.json').read_text()))
            if data.get('passed') is not True:
                raise RuntimeError(f'{kind} acceptance failed')
            results[kind] = data
            print(f'{kind}: GPU edit/weather acceptance passed', flush=True)
        cmd = [sys.executable, '-u', '-B', str(Path(__file__).resolve()), '--lifecycle-only', '--output', str(out / 'lifecycle')]
        if args.switch_manifest:
            cmd += ['--switch-manifest', str(args.switch_manifest.resolve())]
        print(f'lifecycle: starting same-session reset/reload validation; log: {out / "lifecycle.log"}', flush=True)
        run_stage(cmd, out / 'lifecycle.log')
        results['lifecycle'] = _hugsim_expand(json.loads((out / 'lifecycle/probe/report.json').read_text()))
        histories = [_hugsim_expand(json.loads((out / kind / 'closed_loop/run_report.json').read_text()))['trace']
                     for kind in ('rigid', 'deformable')]
        fresh = (histories[0][0]['temporal']['scene_token'] != histories[1][0]['temporal']['scene_token']
                 and all([t['temporal']['history_bev_present'] for t in h] == [False, True, True, True, True]
                         for h in histories))
        results['separate_episode_uniad_histories'] = fresh
        results['passed'] = fresh and results['lifecycle']['passed']
        if not results['passed']:
            raise RuntimeError('combined native/lifecycle acceptance failed')
    except Exception as exc:
        results.update(passed=False, error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        results.update(formal_evaluation_admitted=False, safety_certified=False,
                       uniad_persistent_worker_reset_verified=False)
        (out / 'acceptance.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
