"""Extend existing POST-HOC actor checks with the exact managed render registry."""
import numpy as np
from scipy.spatial import ConvexHull


class ManagedActorEvaluation:
    def __init__(self, native, registry):
        self.native, self.registry = native, registry
        self.groups = dict(native.groups, ManagedAssets='shared managed-render registry')

    def check(self, pose, frame, vehicle):
        from closed172_common import polygon_check
        result = self.native.check(pose, frame, vehicle)
        pose = np.asarray(pose)
        local = np.array([[a*vehicle.half_length_m,b*vehicle.half_width_m,z,1.]
                          for a in (-1,1) for b in (-1,1) for z in (0.,vehicle.body_height_m)])
        world = (pose @ local.T).T[:, :3]
        ego = world[ConvexHull(world[:, :2]).vertices, :2]
        rows = []
        for name, asset in self.registry.assets.items():
            corners = asset.corners()
            if corners[:,2].max() < world[:,2].min() or corners[:,2].min() > world[:,2].max():
                continue
            poly = corners[ConvexHull(corners[:,:2]).vertices, :2]
            rows.append(dict(group='ManagedAssets', object_id='managed:'+name,
                             **polygon_check(ego, poly)))
        native_nearest = result['nearest']
        candidates = rows + ([native_nearest] if native_nearest.get('gap_m') is not None else [])
        nearest = min(candidates, key=lambda x:x['gap_m']) if candidates else native_nearest
        overlaps = result['overlaps'] + [r for r in rows if r['overlap']]
        return dict(result, nearest=nearest, overlap=bool(overlaps), overlaps=overlaps,
                    managed_reference=rows, managed_geometry='same pose/calibration as rendered Gaussians')
