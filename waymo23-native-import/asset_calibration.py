"""Asset-only Gaussian-support calibration. Never fit to model predictions or scenes."""
import hashlib
import json
from pathlib import Path
import numpy as np
from asset_geometry import SCHEMA,declared_contract

CALIBRATION_SCHEMA='gaussian_center_support_calibration_v1'
RULE=dict(opacity_threshold=.5,center_quantiles=[.005,.995],rotate_to_pca=False,
          scale_geometry=False,dimensions_source='opaque_gaussian_centers_not_physical_truth')


def fingerprint(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return dict(bytes=Path(path).stat().st_size,sha256=h.hexdigest())


def fit_support(xyz,opacity):
    xyz=np.asarray(xyz,dtype=float);opacity=np.asarray(opacity,dtype=float).reshape(-1)
    if (xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)!=len(opacity)
            or not np.isfinite(xyz).all() or not np.isfinite(opacity).all()
            or np.any((opacity<0)|(opacity>1))):raise ValueError('Invalid Gaussian center/opacity arrays')
    solid=xyz[opacity>RULE['opacity_threshold']]
    if len(solid)<100:raise ValueError('Insufficient opaque support')
    bounds=np.quantile(solid,RULE['center_quantiles'],axis=0);extent=bounds[1]-bounds[0]
    if np.any(extent<=0):raise ValueError('Degenerate asset support')
    center=(bounds[0]+bounds[1])/2
    return dict(bounds_raw=bounds.tolist(),raw_recenter_xz=center[[0,2]].tolist(),
                wlh=extent[[2,0,1]].tolist(),bottom_y=float(bounds[1,1]),opaque_points=len(solid))


def create(asset,output):
    import torch
    asset=Path(asset).resolve();output=Path(output)
    if output.exists():raise FileExistsError('Calibration is immutable; use a new versioned path')
    params,_=torch.load(asset/'gs.pth',map_location='cpu',weights_only=False)
    support=fit_support(params[1].float().numpy(),torch.sigmoid(params[7]).numpy())
    manifest=dict(schema=CALIBRATION_SCHEMA,geometry_schema=SCHEMA,asset=str(asset),rule=RULE,
                  source_files={name:fingerprint(asset/name) for name in ('gs.pth','wlh.json')},
                  support=support,geometry_contract=declared_contract(support['wlh'],support['bottom_y']),
                  independent_physical_calibration=False,n1_certified=False,
                  prediction_data_used=False,scene_data_used=False)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(manifest,indent=2))
    return manifest


def load(path,asset):
    manifest=json.loads(Path(path).read_text());asset=Path(asset).resolve()
    if manifest.get('schema')!=CALIBRATION_SCHEMA or manifest.get('geometry_schema')!=SCHEMA or manifest.get('rule')!=RULE:
        raise ValueError('Unsupported calibration schema or shared rule')
    if Path(manifest['asset']).resolve()!=asset:raise ValueError('Calibration asset identity mismatch')
    for name in ('gs.pth','wlh.json'):
        if fingerprint(asset/name)!=manifest['source_files'][name]:raise ValueError('Calibration source changed: '+name)
    expected=manifest['support'];bounds=np.asarray(expected['bounds_raw'],dtype=float)
    if bounds.shape!=(2,3) or not np.isfinite(bounds).all() or np.any(bounds[1]<=bounds[0]):raise ValueError('Invalid calibrated bounds')
    extent=bounds[1]-bounds[0]
    if (not np.allclose(expected['wlh'],extent[[2,0,1]],rtol=0,atol=1e-10)
            or not np.allclose(expected['raw_recenter_xz'],bounds.mean(0)[[0,2]],rtol=0,atol=1e-10)
            or not np.isclose(expected['bottom_y'],bounds[1,1],rtol=0,atol=1e-10)):
        raise ValueError('Calibration geometry fields disagree')
    if (manifest.get('independent_physical_calibration') is not False or manifest.get('n1_certified') is not False
            or manifest.get('prediction_data_used') is not False or manifest.get('scene_data_used') is not False):
        raise ValueError('Support calibration must not claim physical certification or prediction fitting')
    if manifest['geometry_contract']!=declared_contract(expected['wlh'],expected['bottom_y']):raise ValueError('Calibration contract mismatch')
    return manifest


def rebase_report(report,manifest):
    """Diagnostic-only new reference; preserve the historical rendered geometry pose."""
    if Path(report['asset']).resolve()!=Path(manifest['asset']).resolve():raise ValueError('Wrong asset calibration')
    support=manifest['support'];new=dict(report)
    delta=np.asarray(support['raw_recenter_xz'])-np.asarray(report['local_recenter_xz'])
    translation=np.eye(4);translation[[0,2],3]=delta
    new['placements']=[dict(p,**{key:(np.asarray(p[key])@translation).tolist() for key in ('asset_to_source','asset_to_native')}) for p in report['placements']]
    new.update(asset_wlh=support['wlh'],local_bottom_y=support['bottom_y'],local_recenter_xz=support['raw_recenter_xz'],
               geometry_contract=manifest['geometry_contract'],reference_kind='gaussian_center_support_proxy',
               n1_edit_certified=False,evaluation_admitted=False,appearance_independently_validated=False)
    return new


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--asset',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();m=create(a.asset,a.output);print(json.dumps(m['support'],indent=2))
