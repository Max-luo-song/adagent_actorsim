"""Use the verified neutral SMPL NPZ even when a cached pickle happens to exist."""

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

import contextlib,importlib,importlib.util
pass  # Path is provided by the portable boundary above
import numpy as np
from contracts import P26,read

def assets():
    from workflow import asset,verify_asset
    tools=P26/'bridge/smpl_tools';provenance=tools/'resources/smpl_neutral/conversion_provenance.json'
    record=read(provenance);verify_asset(record['output'])
    if record['verification']['vertices_shape']!=[1,6890,3] or record['verification']['faces']!=13776:raise ValueError('Unverified SMPL conversion')
    return dict(npz=record['output'],provenance=asset(provenance),converter=asset(tools/'convert_smpl_pkl_to_npz.py'))

def data_struct(resources):
    from workflow import verify_asset
    for value in resources.values():verify_asset(value)
    spec=importlib.util.spec_from_file_location('_verified_smpl_unpack',resources['converter']['path']);converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)
    with np.load(resources['npz']['path'],allow_pickle=False) as packed:data=converter.unpack_from_npz(packed)
    expected={'v_template':(6890,3),'weights':(6890,24),'f':(13776,3),'J_regressor':(24,6890)}
    for key,shape in expected.items():
        if data[key].shape!=shape:raise ValueError('SMPL geometry shape mismatch: '+key)
    if not np.isfinite(data['v_template']).all() or np.min(data['f'])<0 or np.max(data['f'])>=6890:raise ValueError('Invalid neutral SMPL geometry')
    return data

@contextlib.contextmanager
def canonical_layer(resources,bundle):
    """Scoped constructor override; frozen framework files stay untouched."""
    module=importlib.import_module('models.human_body')
    expected=Path(bundle).resolve()/'framework_repo/models/human_body.py'
    if Path(module.__file__).resolve()!=expected:raise ValueError('Unexpected SMPL implementation binding')
    from third_party.smplx.smplx.utils import Struct
    original=module.SMPLLayer;data=data_struct(resources)
    def layer(*args,**kwargs):
        if args:raise ValueError('Canonical SMPL loader requires named arguments')
        if kwargs.get('gender','neutral')!='neutral':raise ValueError('Only verified neutral SMPL is registered')
        kwargs=dict(kwargs,model_path=None,data_struct=Struct(**data))
        return original(**kwargs)
    module.SMPLLayer=layer
    try:yield
    finally:module.SMPLLayer=original
