"""One unchanged point-SDF kernel call for all raster time groups."""
import ctypes,numpy as np
from fast_polygon import lib,D,I,array
from mpc_native_v1.native import states5,axes
lib.occupancy_groups.argtypes=[D,D,D,D,I,I,ctypes.c_int64,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,D];lib.occupancy_groups.restype=None
class GroupedOccupancy:
 def __init__(self,groups,count):
  self.starts=np.zeros(count,dtype=np.int64);self.ends=np.zeros(count,dtype=np.int64);points=[];offset=0
  for indices,p in groups:
   p=np.asarray(p);self.starts[indices]=offset;offset+=len(p);self.ends[indices]=offset;points.append(p)
  self.points=array(np.concatenate(points) if points else np.empty((0,2)))
 def clearance(self,states,vehicle,radius):
  s=states5(states);right,forward=axes(s[:,2]);out=np.empty(len(s));lib.occupancy_groups(s,self.points,right,forward,self.starts,self.ends,len(s),vehicle.half_width_m,vehicle.half_length_m,vehicle.clearance_margin_m,radius,out);return out
