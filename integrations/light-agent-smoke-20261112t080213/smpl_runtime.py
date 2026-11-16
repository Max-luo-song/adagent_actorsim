"""Use verified neutral SMPL data independently of node-local pickle caches."""

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

import contextlib,hashlib,importlib,importlib.util,json,sys,tempfile
pass  # Path is provided by the portable boundary above
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
P26=ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26'
FRAMEWORK=P26/'framework_repo'
ASSETS={_hugsim_expand('${HUGSIM_ROOT}/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/bridge/smpl_tools/resources/smpl_neutral/SMPL_NEUTRAL.npz'): 'dfedafa89af21b224b29163722a22327de0760f89f114ca4336f57f2c6f00024', _hugsim_expand('${HUGSIM_ROOT}/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/bridge/smpl_tools/resources/smpl_neutral/conversion_provenance.json'): '1381073b1fad5089082ef5d30426c5390ad70bed7ee05137659573fda764fd4f', _hugsim_expand('${HUGSIM_ROOT}/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/bridge/smpl_tools/convert_smpl_pkl_to_npz.py'): 'bc1532652a5cfd556c364047ca3f264a42c99014adf84b464568f4ab10be5338', _hugsim_expand('${HUGSIM_ROOT}/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/framework_repo/models/human_body.py'): '526542e722e9d1f99421d05c031fe432bc4f3942aee4e73bdaff71ac5e1bec35', _hugsim_expand('${HUGSIM_ROOT}/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/framework_repo/third_party/smplx/smplx/body_models.py'): '4a6cbe3d8c75a02a27eb334bb5fa442a08981a513893c9824023325f3dd93a62'}

def verified_data():
    for name,digest in ASSETS.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('SMPL asset changed: '+name)
    tools=P26/'bridge/smpl_tools'
    provenance=_hugsim_expand(json.loads((tools/'resources/smpl_neutral/conversion_provenance.json').read_text()))
    if provenance['verification']['vertices_shape']!=[1,6890,3] or provenance['verification']['faces']!=13776:raise ValueError('Unverified SMPL conversion')
    spec=importlib.util.spec_from_file_location('_canonical_smpl_unpack',tools/'convert_smpl_pkl_to_npz.py');converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)
    with np.load(provenance['output']['path'],allow_pickle=False) as packed:data=converter.unpack_from_npz(packed)
    for key,shape in {'v_template':(6890,3),'weights':(6890,24),'f':(13776,3),'J_regressor':(24,6890)}.items():
        if data[key].shape!=shape:raise ValueError('Bad SMPL shape: '+key)
    if not np.isfinite(data['v_template']).all():raise ValueError('Nonfinite SMPL template')
    return data

def body_module():
    sys.path.insert(0,str(FRAMEWORK))
    module=importlib.import_module('models.human_body')
    if Path(module.__file__).resolve()!=FRAMEWORK/'models/human_body.py':raise ValueError('Unexpected SMPL module binding: '+str(module.__file__))
    return module

@contextlib.contextmanager
def canonical_layer():
    data=verified_data();module=body_module()
    from third_party.smplx.smplx.utils import Struct
    original=module.SMPLLayer
    def layer(*args,**kwargs):
        if args:raise ValueError('Canonical SMPL construction requires named arguments')
        if kwargs.get('gender','neutral')!='neutral':raise ValueError('Only verified neutral SMPL registered')
        return original(**dict(kwargs,model_path=None,data_struct=Struct(**data)))
    module.SMPLLayer=layer
    try:yield
    finally:module.SMPLLayer=original

def verified_backend(base):
    class CanonicalSMPLBackend(base):
        def __init__(self,*args,**kwargs):
            with canonical_layer():super().__init__(*args,**kwargs)
    return CanonicalSMPLBackend

def preflight():
    import torch
    # Existing invalid pickle proves that canonical construction bypasses the
    # cache-dependent unpickling branch, rather than relying on a missing file.
    with tempfile.TemporaryDirectory(prefix='smpl_preflight_') as folder:
        model=Path(folder)/'SMPL_NEUTRAL.pkl';model.write_bytes(b'preflight invalid pickle; must never be read')
        with canonical_layer():
            template=body_module().SMPLTemplate(str(model),1,torch.zeros(1,10),'da_pose',use_voxel_deformer=False)
            vertices,faces=template.get_init_vf()
    if tuple(vertices.shape)!=(1,6890,3) or tuple(faces.shape)!=(13776,3) or not torch.isfinite(vertices).all():raise ValueError('SMPL constructor preflight failed')
    return dict(passed=True,actual_framework_template_constructed=True,existing_pickle_bypassed=True,chumpy_required=False,vertices=list(vertices.shape),faces=list(faces.shape),assets=ASSETS,full_scene_GPU_loaded=False)
