"""Authorized static-map road cost and exact full-body execution boundary."""

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

import importlib.util
pass  # Path is provided by the portable boundary above
import numpy as np
from scipy.ndimage import distance_transform_edt,map_coordinates
from shapely.geometry import Polygon,Point
from matplotlib.path import Path as PolygonPath
from common import ROOT,read
from planning_guard import sweep_allowance

spec=importlib.util.spec_from_file_location('frozen_camera_road_reference',ROOT/'integrations/light-road-boundary-v3/road_reference.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)

class StaticGeometry:
    def __init__(self,mapping,vehicle):
        self.data=mapping;self.v=vehicle;self.poly=Polygon(mapping['vertices'])
        if not self.poly.is_valid or self.poly.area<=0 or mapping.get('online_static_map_allowed') is not True:raise ValueError('Invalid/unapproved static map')
        self.uncertainty=float(mapping['uncertainty_m']);self.bounds=self.poly.bounds;self.info=None
        self.step=.25;x0,y0,x1,y1=self.bounds
        self.xs=np.arange(x0-.5,x1+.51,self.step);self.ys=np.arange(y0-.5,y1+.51,self.step)
        X,Y=np.meshgrid(self.xs,self.ys);inside=PolygonPath(mapping['vertices']).contains_points(np.c_[X.ravel(),Y.ravel()]).reshape(X.shape)
        self.sdf=(distance_transform_edt(inside)-distance_transform_edt(~inside))*self.step

    def set_pose(self,info):self.info=info

    def world(self,model_points):
        p=np.asarray(model_points);shape=p.shape
        ground=-float(np.asarray(self.info['model_lidar_frame']['virtual_to_ego'])[2,3])
        xyz=np.c_[p.reshape(-1,2),np.full(p.size//2,ground)]
        virtual=xyz@np.asarray(self.info['explicit_l2g_r_mat']).T+self.info['explicit_l2g_t']
        return np.c_[virtual[:,1],-virtual[:,0]].reshape(shape)

    def model(self,source_xy):
        p=np.asarray(source_xy);z=float(np.asarray(self.info['explicit_l2g_t'])[2])-float(np.asarray(self.info['model_lidar_frame']['virtual_to_ego'])[2,3])
        virtual=np.c_[-p[:,1],p[:,0],np.full(len(p),z)]
        local=(virtual-self.info['explicit_l2g_t'])@np.asarray(self.info['explicit_l2g_r_mat'])
        return local[:,:2]

    def bodies(self,states):
        s=np.asarray(states);corners=np.array([[-self.v.half_width_m,-self.v.half_length_m],[self.v.half_width_m,-self.v.half_length_m],[self.v.half_width_m,self.v.half_length_m],[-self.v.half_width_m,self.v.half_length_m]])
        c=np.cos(s[:,2]);t=np.sin(s[:,2]);points=s[:,None,:2]+np.stack([c[:,None]*corners[:,0]+t[:,None]*corners[:,1],-t[:,None]*corners[:,0]+c[:,None]*corners[:,1]],axis=2)
        return self.world(points)

    def clearance(self,states):
        gaps=[]
        for points in self.bodies(states):
            body=Polygon(points)
            if self.poly.covers(body):gap=body.distance(self.poly.boundary)
            else:gap=-max(1e-6,max(Point(p).distance(self.poly) for p in points))
            gaps.append(gap-self.v.clearance_margin_m-self.uncertainty)
        return np.asarray(gaps)

    def pose_clearance(self,pose):
        corners=np.array([[-self.v.half_length_m,-self.v.half_width_m,0],[self.v.half_length_m,-self.v.half_width_m,0],[self.v.half_length_m,self.v.half_width_m,0],[-self.v.half_length_m,self.v.half_width_m,0]])
        pose=np.asarray(pose);points=(corners@pose[:3,:3].T+pose[:3,3])[:,:2];body=Polygon(points)
        gap=body.distance(self.poly.boundary) if self.poly.covers(body) else -max(1e-6,max(Point(p).distance(self.poly) for p in points))
        return float(gap-self.v.clearance_margin_m-self.uncertainty)

    def distance_cost(self,states,stride):
        s=np.asarray(states);coeff=np.array([(x,y) for x in (-1.,0.,1.) for y in (-1.,0.,1.)]);c=np.cos(s[:,2]);t=np.sin(s[:,2])
        points=s[:,None,:2]+np.stack([c[:,None]*coeff[:,0]*self.v.half_width_m+t[:,None]*coeff[:,1]*self.v.half_length_m,-t[:,None]*coeff[:,0]*self.v.half_width_m+c[:,None]*coeff[:,1]*self.v.half_length_m],axis=2)
        points=self.world(points);low=np.array([self.xs[0],self.ys[0]]);high=np.array([self.xs[-1],self.ys[-1]]);clipped=np.clip(points,low,high)
        coords=np.array([(clipped[:,:,1]-self.ys[0])/self.step,(clipped[:,:,0]-self.xs[0])/self.step])
        distance=map_coordinates(self.sdf,coords,order=1,mode='nearest',prefilter=False)-np.linalg.norm(points-clipped,axis=2)
        return float(np.mean(np.maximum(self.v.clearance_margin_m+self.uncertainty-distance,0.)**2,axis=1).sum()/stride)

class MapRoadReference(previous.RoadReference):
    def __init__(self,vehicle,mapping):
        super().__init__(vehicle);self.geometry=StaticGeometry(mapping,vehicle)

    def observed(self,obs,info,arrays):
        self.mount=np.asarray(info['model_lidar_frame']['virtual_to_ego'],float);self.geometry.set_pose(info)
        self.road=self.geometry;self.soft=self.geometry
        origin=self.geometry.world(np.array([[-self.mount[0,3],-self.mount[1,3]]]))[0]
        lo,hi=self.geometry.data['longitudinal_domain_m'];xx=np.linspace(max(lo,origin[0]-3),min(hi,origin[0]+15),12)
        edges=[]
        if xx[-1]>xx[0]:
            right=np.polyval(self.geometry.data['right_line_coefficients'],xx)
            stations=np.asarray(self.geometry.data['left_stations']);left=np.interp(xx,stations[:,0],stations[:,1])
            for side,boundary in [('right',right),('left',left)]:
                p=self.geometry.model(np.c_[xx,boundary]);a,b=np.polyfit(p[:,1],p[:,0],1)
                edges.append(dict(side=side,a=float(a),b=float(b),valid_y_range=[-50.,50.],observed_y_range=[-50.,50.],uncertainty_m=self.geometry.uncertainty))
        self.packet=dict(schema='authorized_static_map_local_boundaries_v1',status='fresh' if edges else 'unavailable',
            source='frozen_offline_scene_map',edges=edges,online_static_map_authorized=True,per_frame_image_detection=False)
        return self.packet

    def coverage(self,states):
        bodies=self.geometry.bodies(states);lo,hi=self.geometry.data['longitudinal_domain_m']
        return bool(np.all(bodies[:,:,0]>=lo)&np.all(bodies[:,:,0]<=hi))

    def certificate(self,states):
        s=np.asarray(states);gap=float(np.min(self.geometry.clearance(s)-sweep_allowance(s,self.v)));covered=self.coverage(s)
        return dict(passed=bool(covered and gap>=previous.POLICY['extra_white_line_reserve_m']),
            current_edge_coverage_complete=covered,static_map_coverage_complete=covered,
            white_edge_swept_clearance_m=gap,map_full_body_swept_clearance_m=gap,
            whole_body_and_inter_sample_motion_checked=True,physical_white_boundary_certified=False,
            scope='user-authorized static sensor-derived polygon with 0.3m uncertainty; no per-frame detector')

    def process(self,raw,arrays,obs,info,motion):
        ref,audit=super().process(raw,arrays,obs,info,motion)
        audit['static_map_local_boundaries']=audit.pop('observed_white_edge')
        audit.update(source='authorized_static_map_road_reference',ground_truth_used=True,
            truth_scope='authorized static map only; no future actors or recorded future ego execution',
            online_inputs=['authorized static map','current frozen UniAD plan','actual ego pose/speed/steering'],per_frame_line_detector_calls=0)
        return ref,audit
