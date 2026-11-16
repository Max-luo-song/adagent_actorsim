"""Evidence-based presentation of recorded Gaussian depth scans; no new sensing."""

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

import argparse
import hashlib
import json
import os
pass  # Path is provided by the portable boundary above
import subprocess

os.environ.setdefault('MPLCONFIGDIR', '/tmp/hugsim-sensor-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/scene_platform_v1/sensor_validation/jobs/sensor_scan_sync_v1'
OUTPUT = ROOT/'outputs/scene_platform_demo/sensors/scan_showcase_v1'


def read(path):
    return _hugsim_expand(json.loads(Path(path).read_text()))


def asset(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify(item):
    if asset(item['path'])['sha256'] != item['sha256']:
        raise ValueError('Recorded sensor artifact changed: '+item['path'])


def load():
    result = read(JOB/'task_result.json')['sensor_validation']
    assert result['status'] == 'completed' and all(result['checks'].values())
    rows = []
    provenance = []
    for index in range(3):
        arrays = {}; receipts = {}
        for arm in ['original', 'edited', 'reset']:
            item = result['frame_receipts'][arm][index]; verify(item)
            receipt = read(item['path']); verify(receipt['points'])
            with np.load(receipt['points']['path']) as packed:
                arrays[arm] = {k:packed[k].copy() for k in packed.files}
            receipts[arm] = receipt
            provenance.extend([item, receipt['points']])
        a,b,c = (arrays[k] for k in ['original','edited','reset'])
        for k in ['sensor_to_world','beam_directions']:
            np.testing.assert_allclose(a[k], b[k], rtol=0, atol=1e-8)
        for k in ['valid','range_m','xyz_sensor']:
            np.testing.assert_array_equal(a[k], c[k])
        both = a['valid'] & b['valid']
        changed = (a['valid'] != b['valid']) | (both & (np.abs(a['range_m']-b['range_m']) > .25))
        rows.append(dict(arrays=arrays, receipts=receipts, changed=changed))
    return result, rows, provenance


def style(ax, title):
    ax.set_title(title, fontsize=12, loc='left', pad=9)
    ax.grid(alpha=.13)
    for spine in ax.spines.values():
        spine.set_alpha(.25)


def bev(ax, xyz, color=None, zoom=False):
    if color is None:
        color = np.clip(xyz[:,2], -2, 6)
    ax.scatter(xyz[:,0], xyz[:,1], c=color, s=2 if zoom else .7,
               cmap='viridis', vmin=-2, vmax=6, linewidths=0, alpha=.85)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x forward (m)'); ax.set_ylabel('y left (m)')


def build(out):
    result, rows, provenance = load()
    out = Path(out).resolve(); out.mkdir(exist_ok=False,parents=True)
    plt.style.use('dark_background')
    plt.rcParams.update({'font.size':10,'figure.facecolor':'#111923','axes.facecolor':'#172330'})
    stats = []; cards = []
    for index,row in enumerate(rows):
        a,b = row['arrays']['original'], row['arrays']['edited']; changed=row['changed']
        receipt = row['receipts']['edited']; source_frame=receipt['source_frame']
        config = read(JOB/'work/sensor_validation/edited/manifest.json')['config']
        channels, azimuths = config['channels'], config['azimuth_samples']
        assert len(a['valid']) == channels*azimuths
        fig, axes = plt.subplots(2,3,figsize=(18,10),dpi=120)
        fig.subplots_adjust(left=.055,right=.97,top=.85,bottom=.12,wspace=.25,hspace=.40)
        fig.suptitle(f'SCENE 551 | 360 DEGREE GAUSSIAN DEPTH SCAN | source frame {source_frame} / {source_frame/10:.1f} s',fontsize=17,y=.965)
        fig.text(.055,.905,f'32 rings x 720 azimuths | valid returns: original {len(a["xyz_sensor"]):,} / edited {len(b["xyz_sensor"]):,} | changed rays (>0.25 m or validity): {changed.sum()}',fontsize=12,color='#83e2ce')
        for col,(label,arr) in enumerate([('ORIGINAL',a),('EDITED',b)]):
            ax=axes[0,col];bev(ax,arr['xyz_sensor']);style(ax,label+' | top view, height colors')
            ax.set_xlim(-20,50);ax.set_ylim(-25,25)
            ax.scatter([0],[0],marker='^',s=55,c='#ffdc70',edgecolors='black',zorder=5)
        ax=axes[0,2]
        bev(ax,a['xyz_sensor'],color=np.repeat(np.array([[.35,.45,.55]]),len(a['xyz_sensor']),axis=0),zoom=True)
        changed_points=[]
        for arr,color,label in [(a,'#ffb66b','old range endpoint'),(b,'#70f0d0','edited range endpoint')]:
            xyz=arr['xyz_sensor'];keep=changed[arr['ray_index']]
            ax.scatter(xyz[keep,0],xyz[keep,1],s=22,c=color,label=label,edgecolors='none',zorder=4)
            if keep.any():changed_points.append(xyz[keep,:2])
        if changed_points:
            xy=np.concatenate(changed_points);lo=xy.min(0)-3;hi=xy.max(0)+3
            center=(lo+hi)/2;half=max(6,float((hi-lo).max()/2))
            ax.set_xlim(center[0]-half,center[0]+half);ax.set_ylim(center[1]-half,center[1]+half)
        else:
            ax.set_xlim(-20,50);ax.set_ylim(-25,25)
            ax.text(.5,.90,'Before intervention: identical returns',ha='center',transform=ax.transAxes,color='#83e2ce')
        style(ax,'CHANGED RAYS | endpoint zoom');ax.legend(fontsize=8,loc='lower right')
        for col,(label,arr) in enumerate([('ORIGINAL',a),('EDITED',b)]):
            ranges=arr['range_m'].reshape(channels,azimuths)
            valid=arr['valid'].reshape(channels,azimuths)
            image=axes[1,col].imshow(np.ma.masked_where(~valid,ranges),origin='lower',aspect='auto',extent=[-180,180,0,channels],cmap='turbo',vmin=0,vmax=60,interpolation='nearest')
            style(axes[1,col],label+' | range image');axes[1,col].set_xlabel('azimuth (deg)');axes[1,col].set_ylabel('ring index')
            fig.colorbar(image,ax=axes[1,col],pad=.02,shrink=.8,label='range (m), capped at 60')
        delta=np.zeros_like(a['range_m']);both=a['valid']&b['valid'];delta[both]=b['range_m'][both]-a['range_m'][both]
        ax=axes[1,2];im=ax.imshow(delta.reshape(channels,azimuths),origin='lower',aspect='auto',extent=[-180,180,0,channels],cmap='coolwarm',vmin=-15,vmax=15,interpolation='nearest')
        rr,cc=np.nonzero(changed.reshape(channels,azimuths));ax.scatter(cc/azimuths*360-180,rr+.5,marker='s',s=7,c='#ffdf76',linewidths=.5,facecolors='none')
        style(ax,'EDIT - ORIGINAL | yellow = changed rays');ax.set_xlabel('azimuth (deg)');ax.set_ylabel('ring index')
        fig.colorbar(im,ax=ax,pad=.02,shrink=.8,label='signed range change (m)')
        fig.text(.055,.058,'Recorded GPU scans, SAME ego pose and source time in both arms. Reset exactly reproduces original scans.',fontsize=11)
        fig.text(.055,.030,'Approximate expected Gaussian depth; not physical LiDAR or mmWave Radar. No intensity / Doppler. Recorder only, not controller input.',fontsize=10,color='#aebcca')
        path=out/f'frame_{source_frame:03d}.png';fig.savefig(path);plt.close(fig);cards.append(path)
        stats.append(dict(source_frame=source_frame,source_timestamp_s=receipt['source_timestamp_s'],original_points=len(a['xyz_sensor']),edited_points=len(b['xyz_sensor']),changed_rays=int(changed.sum()),changed_threshold_m=.25,reset_matches_original=True))
    # Three measured snapshots, explicitly held rather than interpolated as a fake sequence.
    ff=imageio_ffmpeg.get_ffmpeg_exe(); video=out/'sensor_edit_comparison.mp4'
    concat=out/'snapshots.ffconcat'
    concat.write_text('ffconcat version 1.0\n'+''.join(f"file '{p.name}'\nduration 4\n" for p in cards)+f"file '{cards[-1].name}'\n")
    subprocess.run([ff,'-v','error','-n','-f','concat','-safe','0','-i',str(concat),'-vf','fps=10','-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(video)],check=True)
    subprocess.run([ff,'-v','error','-threads','1','-i',str(video),'-f','null','-'],check=True)
    # Orbit only the visualization camera around ONE measured cloud; no simulated motion.
    b=rows[1]['arrays']['edited'];xyz=b['xyz_sensor'];mask=(np.linalg.norm(xyz[:,:2],axis=1)<45)&(xyz[:,2]>-3)&(xyz[:,2]<12)
    xyz=xyz[mask]; orbit=out/'orbit';orbit.mkdir()
    fig=plt.figure(figsize=(12.8,7.2),dpi=100);ax=fig.add_subplot(projection='3d')
    ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],s=1.2,c=xyz[:,2],cmap='viridis',vmin=-2,vmax=6,linewidths=0)
    ax.scatter([0],[0],[0],c='#ffdc70',marker='^',s=65)
    ax.set(xlim=(-30,45),ylim=(-35,35),zlim=(-3,12),xlabel='x forward (m)',ylabel='y left (m)',zlabel='height (m)')
    ax.set_box_aspect((75,70,30))
    fig.suptitle('RECORDED 360 DEGREE POINT CLOUD | scene 551 / source frame 30',fontsize=15)
    fig.text(.04,.04,'One fixed edited scan. Only display camera rotates; not a new sensing sequence or driving replay.',fontsize=10)
    fig.text(.04,.015,'Approximate Gaussian depth scan | 32 rings x 720 azimuths | no physical reflectance / Doppler',fontsize=10,color='#aebcca')
    for i in range(48):
        ax.view_init(elev=25,azim=-65+i*360/48);fig.savefig(orbit/f'{i:04d}.png')
    plt.close(fig)
    orbit_video=out/'pointcloud_3d_orbit.mp4'
    subprocess.run([ff,'-v','error','-n','-framerate','8','-i',str(orbit/'%04d.png'),'-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(orbit_video)],check=True)
    subprocess.run([ff,'-v','error','-threads','1','-i',str(orbit_video),'-f','null','-'],check=True)
    report=dict(completed=True,source_job=str(JOB),source_assets=provenance,comparisons=stats,
                snapshot_video=asset(video),orbit_video=asset(orbit_video),overview=asset(cards[1]),full_video_decode_passed=True,
                source_capture_count=9,source_extra_gaussian_renders=54,new_gpu_renders=0,training_executed=False,
                controller_input=False,physical_lidar_certified=False,mmwave_radar=False,
                snapshot_playback='3 measured snapshots; each held 4 seconds; no temporal interpolation',
                orbit_playback='single measured frame 30, display-camera rotation only')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'README.md').write_text('# 雷达模块效果展示\n\n这里展示已接入的近似高斯深度扫描模块，而非毫米波 Radar。数据复用实际 GPU 扫描。\n\n- sensor_edit_comparison.mp4：源帧10/30/50，每帧停留4秒，原始／编辑／差分，上方点云俯视与变化放大，下方32线距离图。固定自车位置，未执行驾驶。\n- pointcloud_3d_orbit.mp4：源帧30编辑后同一份点云，只有展示相机旋转。\n- frame_030.png：编辑同步效果总览。\n- report.json：源文件哈希、变化波束计数、输出和限制。\n\n距离变化超过0.25米或有效性变化计入变化波束。颜色表示高度／距离／距离差，不表示物理反射强度。背景点与变化点没有做物体语义识别。复位逐点恢复原扫描；当前点云仅录制，不进入UniAD或控制器。\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_assets'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT)
    build(parser.parse_args().output)
