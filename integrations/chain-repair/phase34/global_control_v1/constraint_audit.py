"""Report independent sources; the legacy raw field includes their minimum."""
import numpy as np
from empty_native_temporal import TemporalOccupancy
from mpc_native_v1.native import occupancy_clearance
from bounded_visibility_geometry import polygon_clearance,projected_cell

def initial_components(occupancy,report,info,vehicle):
 off=np.asarray(info['model_lidar_frame']['virtual_to_ego'])[:2,3]
 state=np.r_[-off,0.,float(info['ego_velo']),float(info['ego_steer'])][None]
 masks=np.asarray(occupancy).reshape(1,5,200,200)
 raw=float(TemporalOccupancy(masks,vehicle).clearance(state,[0.])[0])
 depth=report.get('observed_depth_geometry',{});unknown=report.get('unknown_visibility_geometry',{})
 def component(points,radius,cell_mode):
  points=np.asarray(points if points is not None else [],dtype=float).reshape(-1,2)
  if not len(points):return None
  if cell_mode:
   extra=float(radius)-np.sqrt(3)*.25
   if extra< -1e-10:raise ValueError('Invalid voxel uncertainty radius')
   expanded=np.c_[state,[[vehicle.half_width_m,vehicle.half_length_m]]]
   return float(polygon_clearance(expanded,points,projected_cell(info['explicit_l2g_r_mat']),vehicle.clearance_margin_m+max(0.,extra)+1e-7)[0])
  return float(occupancy_clearance(state,points,vehicle,float(radius))[0])
 depth_gap=component(report.get('observed_depth_obstacles_xy'),depth.get('voxel_half_diagonal_m',np.sqrt(2)*.25),depth.get('policy')=='observed_depth_geometry_with_free_space_clearing_v1' and np.isclose(depth.get('voxel_resolution_m',-1),.5))
 unknown_gap=component(report.get('unknown_visibility_obstacles_xy'),unknown.get('point_radius_m',np.sqrt(2)*.25),unknown.get('policy')=='endpoint_supported_unknown_visibility_v2' and np.isclose(depth.get('voxel_resolution_m',-1),.5))
 fields=dict(network_raster_gap_m=raw,observed_depth_gap_m=depth_gap,unknown_boundary_gap_m=unknown_gap)
 values=[x for x in fields.values() if x is not None]
 return dict(**fields,combined_gap_m=min(values),blocking_sources=[name for name,gap in fields.items() if gap is not None and gap< -1e-5],missing_geometry_means_free_space=False,none_means_no_cells_from_this_source=True)
