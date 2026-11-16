"""CPU audit of shared asset axes/dimensions and historical scene references."""
import json
import pickle
from pathlib import Path
import sys
import numpy as np
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad'))
from asset_geometry import local_corners,source_corners,bottom_polygon,validate_pose,declared_contract
from evaluate_preview_vehicle import reference_polygon
from edit_observability import projected_extent


def audit(output):
    output=Path(output);output.mkdir(exist_ok=True)
    import torch
    asset=BASE/'data/3DRealCar/2024_06_04_13_44_39'
    params,_=torch.load(asset/'gs.pth',map_location='cpu',weights_only=False)
    xyz=params[1].float().numpy();opacity=torch.sigmoid(params[7]).numpy().reshape(-1)
    wlh=np.asarray(json.loads((asset/'wlh.json').read_text()))
    solid=xyz[opacity>.5];bounds=np.quantile(solid,[.005,.995],axis=0);extent=np.diff(bounds,axis=0)[0]
    eigenvalues,vectors=np.linalg.eigh(np.cov(solid[:,[0,2]].T));axis=vectors[:,-1]
    angle=float(np.degrees(np.arccos(np.clip(abs(axis[0]),0,1))))
    support=[]
    for threshold in (.5,.8,.95):
        points=xyz[opacity>threshold]
        for q in (.005,.01,.025,.05):
            b=np.quantile(points,[q,1-q],axis=0);e=(b[1]-b[0])[[2,0,1]]
            support.append(dict(opacity_threshold=threshold,center_quantiles=[q,1-q],points=len(points),support_wlh=e.tolist(),ratio_to_declared=(e/wlh).tolist()))
    scenes={}
    for scene,tag in (('172','10'),('552','20')):
        folder=BASE/f'outputs/gpu_test_service/jobs/edfix-{tag}-{scene}-render-v1/result'
        a=json.loads((folder/'asset_report.json').read_text());s=json.loads((folder/'sequence_report.json').read_text())
        pose=validate_pose(a['placements'][0]['asset_to_source']);native=validate_pose(a['placements'][0]['asset_to_native'])
        T=np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]])
        points=source_corners(a);local=local_corners(a['asset_wlh'],a['local_bottom_y']);recovered=(points-pose[:3,3])@pose[:3,:3]
        scene_data=dict(native_transform_max_error=float(np.max(np.abs(native-T@pose))),
                        local_roundtrip_max_error=float(np.max(np.abs(local-recovered))),
                        proxy_contact_residuals_m=a['placement_search']['proxy_geometry']['contact_residuals_m'],
                        measured_wheel_contact=False,independent_appearance_reference=False,frames=[])
        centered=solid.copy();centered[:,[0,2]]-=a['local_recenter_xz']
        low=np.array([-wlh[1]/2,a['local_bottom_y']-wlh[2],-wlh[0]/2]);high=np.array([wlh[1]/2,a['local_bottom_y'],wlh[0]/2])
        scene_data['opaque_centers_outside_declared_proxy_fraction']=float(np.mean(np.any((centered<low)|(centered>high),axis=1)))
        for frame in s['frames']:
            packet=Path(frame['packet'])
            with packet.open('rb') as stream:obs,info=pickle.load(stream)
            # Independently reconstruct the historical evaluator's bottom plane.
            lidar=np.eye(4);lidar[:3,:3]=info['explicit_l2g_r_mat'];lidar[:3,3]=info['explicit_l2g_t']
            A=np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
            bottom=bottom_polygon(a['asset_wlh'],a['local_bottom_y']);world=bottom@pose[:3,:3].T+pose[:3,3]
            explicit=(np.c_[world,np.ones(4)]@(np.linalg.inv(lidar)@A).T)[:,:2]
            old=reference_polygon(a,info)
            row=dict(frame=frame['frame'],historical_reference_max_error=float(np.max(np.abs(old-explicit))),views=[])
            with np.load(packet.parent/'render_evidence.npz') as ev:
                for name,image in obs['rgb'].items():
                    camera=ev[name+'__camera_to_world'];K=ev[name+'__K']
                    check=projected_extent(points,camera,K,(image.shape[1],image.shape[0]))
                    row['views'].append(dict(camera=name,**check))
            scene_data['frames'].append(row)
        scenes[scene]=scene_data
    result=dict(asset=str(asset),contract=declared_contract(wlh,float(bounds[1,1])),
                declared_wlh=wlh.tolist(),opaque_center_quantile_extent_wlh=extent[[2,0,1]].tolist(),
                extent_ratio_to_declared=(extent[[2,0,1]]/wlh).tolist(),pca_undirected_long_axis_error_degrees=angle,
                pca_eigenvalue_ratio=float(eigenvalues[-1]/eigenvalues[0]),support_sensitivity=support,scenes=scenes,
                physical_box_independently_verified=False,signed_front_verified=False,n1_certified=False,
                reference_ready_for_accuracy_claim=False,
                blocking_reasons=["Declared dimensions differ from opaque Gaussian support",
                                  "No independent physical dimensions or signed front reference",
                                  "Proxy bottom is not measured wheel contact",
                                  "No independent object-specific multiview depth/occlusion verification"],
                new_training=False,new_rendering=False)
    (output/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('declared_wlh','opaque_center_quantile_extent_wlh','extent_ratio_to_declared','pca_undirected_long_axis_error_degrees')},indent=2),flush=True)

if __name__=='__main__':audit(BASE/'outputs/asset_reference_audit_20261021')
