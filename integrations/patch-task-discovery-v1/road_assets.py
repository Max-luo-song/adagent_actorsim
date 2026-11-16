"""Export measured road-support cells with holes; never invent an authorized lane map."""

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
import json
import numpy as np
from shapely.geometry import box, mapping
from shapely.ops import unary_union

def export(bundle):
    from service import write,asset
    bundle=Path(bundle);meta=_hugsim_expand(json.loads((bundle/'source_manifest.json').read_text()))
    points=np.load(bundle/'road_sr_world.npy',mmap_mode='r');step=.5
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Invalid observed road geometry')
    cells=np.unique(np.floor(points[:,:2]/step).astype(np.int64),axis=0)
    rectangles=[]
    for y in np.unique(cells[:,1]):
        xs=np.sort(cells[cells[:,1]==y,0]);starts=np.r_[0,np.flatnonzero(np.diff(xs)>1)+1];ends=np.r_[starts[1:]-1,len(xs)-1]
        rectangles.extend(box(xs[a]*step,y*step,(xs[b]+1)*step,(y+1)*step) for a,b in zip(starts,ends))
    poly=unary_union(rectangles)
    out=bundle/'map_support';out.mkdir(exist_ok=True)
    write(out/'road_support.geojson',dict(type='Feature',geometry=mapping(poly),properties=dict(scene=meta['scene'],coordinate_frame='sr_world_first_ego',cell_size_m=step,authority='RoadNodes reconstruction support, predicted road semantics',unknown_holes_preserved=True,hole_filling=False,convex_hull=False,physical_white_line_boundary_verified=False,online_static_map_allowed=False)))
    write(out/'manifest.json',dict(scene=meta['scene'],source=asset(bundle/'road_sr_world.npy'),geometry=asset(out/'road_support.geojson'),occupied_cells=len(cells),holes_preserved=True,ready_for_runtime=False,missing=['validated runtime map adapter','lane/white-boundary coverage admission'],native_evaluation_admitted=False))
