"""Plot real model map output and offline candidates; no synthetic road truth."""

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
import os,json
import numpy as np

def render(output,source,data,road,candidate):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/hugsim-road-preflight-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.colors import ListedColormap
    output=Path(output);source=Path(source);report=_hugsim_expand(json.loads((source/'run_report.json').read_text()))
    info=_hugsim_expand(json.loads((source/'uniad_worker/step_000/sensor_metadata.json').read_text()));offset=np.asarray(info['model_lidar_frame']['virtual_to_ego'])[:2,3]
    old=np.array([report['trace'][0]['state_before']]+[row['state_after'] for row in report['trace']]);old[:,:2]-=offset
    mask=np.asarray(road['mask']);b=road['bounds_xy'];g=_hugsim_expand(json.loads((source/'uniad_worker/step_000/guard_memory_report.json').read_text()))
    fig,axes=plt.subplots(1,2,figsize=(14,9),dpi=110)
    for ax in axes:
        ax.imshow(mask,origin='lower',extent=[b[0],b[2],b[1],b[3]],cmap=ListedColormap(['#e1c7b5','#d1e8d5']),alpha=.8,interpolation='nearest')
        for obstacle in g['obstacles']:
            box=np.array(obstacle['box']);x,y=box[:2]
            if obstacle.get('label')==8:ax.scatter(x,y,c='#b62e8c',s=12,alpha=.7)
            elif obstacle.get('label')==0:
                right=np.array([np.cos(box[6]),-np.sin(box[6])]);forward=np.array([np.sin(box[6]),np.cos(box[6])]);p=np.array([np.array([x,y])+a*box[3]/2*right+b_*box[4]/2*forward for a,b_ in [(-1,-1),(1,-1),(1,1),(-1,1)]])
                ax.add_patch(Polygon(p,fill=False,edgecolor='#7e633a'))
        neural=np.array(data['trajectory'])-offset;ax.plot(neural[:,0],neural[:,1],':',color='#51359d',label='Frozen neural reference')
        ax.set(xlim=(-10,10),ylim=(-4,22),aspect='equal',xlabel='Model x: right (m)',ylabel='Model y: forward (m)');ax.grid(alpha=.2)
    axes[0].plot(old[:,0],old[:,1],color='#c83e3e',linewidth=2,label='Archived v20 actually driven');axes[0].set_title('Archived drive against initial predicted road')
    selected=candidate['selected']
    if selected is not None:
        states=np.array(selected['states']);axes[1].plot(states[:,0],states[:,1],color='#1673b8',linewidth=2,label='New OFFLINE candidate; not driven')
    axes[1].set_title('New road-constrained candidate')
    for ax in axes:ax.legend(loc='upper left',fontsize=8)
    fig.suptitle('Green: model-predicted road | beige: predicted non-road | magenta: pedestrian hypotheses\nNo simulator road labels or object references were supplied to planning',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(output/'predicted_road_candidate_comparison.png');plt.close(fig)
    # Project the map back onto the original FRONT RGB using independent
    # camera calibration and the sensor-supported ground plane. This exposes
    # a transpose/sign/domain error that a plausible BEV alone could hide.
    import pickle
    from PIL import Image,ImageDraw
    from depth_surface_geometry import ground_plane
    from runtime_configuration import configured_reliability
    configured_reliability()
    from input_geometry import intrinsic_matrix
    with (source/'uniad_worker/step_000/observation.pkl').open('rb') as stream:obs,metadata=pickle.load(stream)
    plane=ground_plane(obs,metadata)
    if plane is not None:
        camera=metadata['cam_params']['CAM_FRONT'];transform=np.asarray(camera['l2c']);k=intrinsic_matrix(camera['intrinsic'])[:3,:3]
        image=Image.fromarray(obs['rgb']['CAM_FRONT']).convert('RGBA');layer=Image.new('RGBA',image.size);draw=ImageDraw.Draw(layer)
        dy=(b[3]-b[1])/mask.shape[0];dx=(b[2]-b[0])/mask.shape[1]
        for row in range(mask.shape[0]):
            y=b[1]+row*dy
            if not 0<=y<=30:continue
            for column in range(mask.shape[1]):
                x=b[0]+column*dx
                if not -15<=x<=15:continue
                xy=np.array([[x,y],[x+dx,y],[x+dx,y+dy],[x,y+dy]])
                z=np.c_[xy,np.ones(4)]@plane;xyz=np.c_[xy,z,np.ones(4)];cp=(transform@xyz.T).T[:,:3]
                if np.any(cp[:,2]<=.1):continue
                uv=cp@k.T;uv=uv[:,:2]/cp[:,2,None]
                if uv[:,0].max()<0 or uv[:,0].min()>=image.width or uv[:,1].max()<0 or uv[:,1].min()>=image.height:continue
                draw.polygon([tuple(point) for point in uv],fill=(30,200,80,55) if mask[row,column] else (230,110,40,65))
        image=Image.alpha_composite(image,layer).convert('RGB')
        image.save(output/'predicted_road_front_overlay.png')
