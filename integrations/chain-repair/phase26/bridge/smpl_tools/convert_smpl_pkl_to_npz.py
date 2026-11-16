"""Controlled one-time conversion: official SMPL pkl -> smplx-format npz.

The official SMPL pkl pickles chumpy.Ch objects (thin ndarray wrappers).
chumpy is unavailable offline (the aaai/ReconDrive/chumpy git dir is gutted:
no objects/HEAD). This converter registers a DESERIALIZATION-ONLY real
implementation of chumpy.Ch (an ndarray subclass carrying the actual data),
converts everything to plain numpy/scipy arrays, and writes a smplx npz.

Honesty guard: verify_npz() loads the output with the vendored smplx
(framework_repo/third_party/smplx) with NO chumpy registered, and runs a
forward pass. The artifact must work in the real chain without the shim.

Original pkl is read-only; provenance recorded.
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

import hashlib
import json
import pickle
import sys
import types
pass  # Path is provided by the portable boundary above

import numpy as np

SRC = Path(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/4D-Humans/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'))
HERE = Path(__file__).resolve().parent
OUTDIR = HERE / 'resources/smpl_neutral'
OUT = OUTDIR / 'SMPL_NEUTRAL.npz'
EXPECTED_VERTS = 6890
EXPECTED_FACES = 13776


class _ChCarrier:
    """Deserialization-only stand-in for chumpy.Ch.

    Verified against the actual pkl: chumpy.Ch pickles as a state dict whose
    'x' entry IS the array (e.g. shapedirs -> (6890,3,300)); the other keys
    (_dirty_vars/_itr/_depends_on_deps) are chumpy-internal bookkeeping with
    no data. This carrier keeps the real array; nothing else is needed.
    """
    def __setstate__(self, state):
        if not isinstance(state, dict) or 'x' not in state:
            raise ValueError('unexpected chumpy.Ch state: ' + repr(type(state)))
        self.x = np.asarray(state['x'])


def install_deserialization_chumpy():
    chumpy = types.ModuleType('chumpy')
    ch_mod = types.ModuleType('chumpy.ch')
    for mod in (chumpy, ch_mod):
        mod.Ch = _ChCarrier
    sys.modules['chumpy'] = chumpy
    sys.modules['chumpy.ch'] = ch_mod


def sha256_of(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def to_plain(obj):
    import scipy.sparse as sp
    if isinstance(obj, _ChCarrier):
        return obj.x
    if isinstance(obj, np.ndarray):
        return np.asarray(obj)
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_plain(v) for v in obj)
    return obj


def pack_for_npz(plain):
    """np.savez cannot hold scipy sparse or python strings natively; encode
    sparse matrices as component triplets and strings as str arrays."""
    import scipy.sparse as sp
    out = {}
    for k, v in plain.items():
        if sp.issparse(v):
            csc = v.tocsc()
            out[k + '__sparse_data'] = csc.data
            out[k + '__sparse_indices'] = csc.indices
            out[k + '__sparse_indptr'] = csc.indptr
            out[k + '__sparse_shape'] = np.asarray(csc.shape)
        elif isinstance(v, str):
            out[k] = np.asarray(v)
        else:
            out[k] = np.asarray(v)
    return out


def unpack_from_npz(packed):
    import scipy.sparse as sp
    data = {}
    sparse_parts = {}
    for k in packed.files:
        if '__sparse_' in k:
            base, part = k.split('__sparse_')
            sparse_parts.setdefault(base, {})[part] = packed[k]
        else:
            data[k] = packed[k]
    for base, parts in sparse_parts.items():
        data[base] = sp.csc_matrix(
            (parts['data'], parts['indices'], parts['indptr']),
            shape=tuple(parts['shape']))
    return data


def verify_npz(npz_path):
    """Load with vendored smplx, NO chumpy in sys.modules (shim removed).

    The vendored SMPL class hardcodes the .pkl filename for directory model
    paths, but accepts a prebuilt data_struct. We therefore verify the npz by
    loading it and passing data_struct=Struct(**npz) -- the same object the
    pkl path would produce, without chumpy anywhere.
    """
    sys.modules.pop('chumpy', None)
    sys.modules.pop('chumpy.ch', None)
    repo = HERE.parents[1] / 'framework_repo'
    sys.path.insert(0, str(repo / 'third_party/smplx'))
    import torch
    from smplx import SMPLLayer
    from smplx.utils import Struct
    packed = np.load(npz_path)
    data = unpack_from_npz(packed)
    layer = SMPLLayer(model_path=None, data_struct=Struct(**data))
    out = layer(betas=torch.zeros(1, 10))
    v = tuple(out.vertices.shape)
    assert v[1] == EXPECTED_VERTS, v
    n_faces = layer.faces.shape[0] if hasattr(layer.faces, 'shape') else len(layer.faces)
    assert int(n_faces) == EXPECTED_FACES, n_faces
    return dict(vertices_shape=list(v), faces=int(n_faces),
                joints=int(out.joints.shape[1]),
                verification_path='npz -> smplx.utils.Struct -> SMPLLayer(data_struct=...)')


def main():
    if OUTDIR.exists():
        raise SystemExit(f'refusing to overwrite: {OUTDIR}')
    install_deserialization_chumpy()
    with SRC.open('rb') as f:
        data = pickle.load(f, encoding='latin1')
    print('pkl keys:', sorted(data.keys()))
    plain = {k: to_plain(v) for k, v in data.items()}
    OUTDIR.mkdir(parents=True)
    np.savez(OUT, **pack_for_npz(plain))
    report = verify_npz(OUT)
    prov = dict(schema='smpl_controlled_conversion_v1', diagnostic_uncertified=True,
                source=dict(path=str(SRC), sha256=sha256_of(SRC)),
                output=dict(path=str(OUT), sha256=sha256_of(OUT)),
                converter='bridge/smpl_tools/convert_smpl_pkl_to_npz.py',
                verification=report,
                chumpy_note='deserialization-only real ndarray subclass; removed before verification',
                original_preserved=True)
    (OUTDIR / 'conversion_provenance.json').write_text(json.dumps(prov, indent=2))
    print(json.dumps(prov['verification'], indent=2))
    print('wrote', OUT, OUT.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
