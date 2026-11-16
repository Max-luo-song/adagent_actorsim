"""Isolated experiment paths and immutable input contracts."""

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

pass  # Path is provided by the portable boundary above
import hashlib,json,os,sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CR=ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair'
P26=CR/'phase26'
SOURCE=CR/'phase46/cutin551_white_edge_v1'
BUNDLE=CR/'phase33/scene_data551_v1/runs/scene551_20261106T050218Z'
DEFAULT=CR/'phase26/light_cutin551_v1'
SCORER=ROOT/'code/HUGSIM/sim/utils/score_calculator.py'
WORKER=CR/'phase46/framework_driver_v1/worker.py'

def read(p):return _hugsim_expand(json.loads(Path(p).read_text()))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def asset(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def verify(a):
    if sha(a['path'])!=a['sha256']:raise ValueError('Changed frozen asset: '+a['path'])
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False,ensure_ascii=False)+'\n');tmp.replace(p)
def configure():
    # Select the reviewed renderer/session snapshot without calling run_driver.configure().
    paths=[ROOT/'code/HUGSIM',ROOT/'waymo23-native-import',ROOT/'integrations/sr-hugsim',
        SOURCE/'framework_snapshot',SOURCE/'framework_snapshot/framework_backend',SOURCE/'framework_snapshot/unified',SOURCE]
    for p in paths:sys.path.insert(0,str(p))
    fix=P26/'bridge/env_fix';build=fix/'nvdiffrast_build_src';libs=sorted((build/'build').glob('lib.linux-*'))
    for p in [build,*libs,fix/'nvdiffrast_meta']:sys.path.insert(0,str(p))
    os.environ.update(SMPL_NEUTRAL_NPZ=str(P26/'bridge/smpl_tools/resources/smpl_neutral/SMPL_NEUTRAL.npz'),
        CUDA_HOME='/usr/local/cuda-12.4',TORCH_EXTENSIONS_DIR=str(fix/'torch_ext_isolated'))
    os.environ['PATH']='/usr/local/cuda-12.4/bin:'+os.environ.get('PATH','')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/hugsim_light551_matplotlib')

def code_pins():
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))
    paths += [ROOT/'code/HUGSIM/sim/ilqr/lqr.py',ROOT/'code/HUGSIM/sim/ilqr/lqr_solver.py',
        ROOT/'code/HUGSIM/sim/utils/sim_utils.py',ROOT/'code/HUGSIM/sim/hugsim_env/envs/hug_sim.py',SCORER,
        ROOT/'waymo23-native-import/live_env.py',ROOT/'waymo23-native-import/trajectory_control.py',
        ROOT/'waymo23-native-import/configs/vehicle.json',P26/'bridge/agent_json/ego_execution.py',
        ROOT/'code/HUGSIM/configs/sim/nuscenes_camera.yaml',ROOT/'waymo23-native-import/build_source_sequence.py',
        ROOT/'integrations/sr-hugsim/uniad_bridge.py',SOURCE/'native_car_edit.py',SOURCE/'native_scores.py',WORKER,
        CR/'phase46/framework_driver_v1/protocol_branch_worker.py']
    # Bind the complete existing neural worker + renderer release, not its old MPC behavior.
    manifest=read(SOURCE/'manifest.json')
    wanted=('code/UniAD_SIM/','tools/pixi-uniad-legacy/','framework_snapshot/','phase27/branch_state.py')
    for name,digest in manifest.items():
        if any(x in name for x in wanted):
            p=ROOT/name
            if p.is_file():
                if sha(p)!=digest:raise ValueError('Existing frozen model/renderer dependency changed: '+name)
                paths.append(p)
    return {str(p.resolve()):sha(p) for p in dict.fromkeys(paths)}
