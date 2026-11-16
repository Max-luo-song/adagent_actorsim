"""Native light GPU evidence: source camera images and explicit nearest zoom."""

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

import json
import subprocess
pass  # Path is provided by the portable boundary above
import numpy as np
from PIL import Image,ImageDraw
from ego_trajectory import ffmpeg_path



def lens_color_checks(values):
    def dominant(state):
        r,g,b=values[state+'_view'].get(state,[0,0,0])
        return r-g>50 and r-b>50 if state=='red' else (r-b>50 and g-b>35 if state=='yellow' else g-r>50 and g-b>20)
    return dict(calibrated_active_lens_is_clear_color=all(dominant(state) for state in ['red','yellow','green']),
        calibrated_inactive_lenses_are_dark=all(max(values[state+'_view'].get(other,[999.]*3))<110
            for state in ['red','yellow','green','off'] for other in ['red','yellow','green'] if state!=other))


def present_signals(observations,output,backend):
    by_id={o['request_id']:o for o in observations}
    names=['original_view','red_view','yellow_view','green_view','off_view']
    if not all(name in by_id and 'traffic_signals' in by_id[name] for name in names):return None
    target=by_id['red_view']['traffic_signals']['programs'][0]['object_id'];cam='CAM_FRONT'
    original=by_id['original_view'];targets=original['traffic_signals']['projected_targets']
    entry=next((t for t in targets if t['object_id']==target),None)
    roi=entry['projected_candidate_rois'].get(cam) if entry else None
    out=Path(output)/'traffic_lights';out.mkdir();frames=out/'state_frames';frames.mkdir()
    checks=dict(projected_candidate_roi_available=bool(roi and roi['intersects_image']))
    with Image.open(original['cameras'][cam]['rgb']) as source:w,h=source.size
    raw=roi['box_xyxy'] if roi else [0,0,w,h]
    box=(max(0,raw[0]),max(0,raw[1]),min(w,raw[2]),min(h,raw[3]))
    if box[2]<=box[0] or box[3]<=box[1]:box=(0,0,w,h);checks['projected_candidate_roi_available']=False
    colors={};panels=[];order=('CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT')
    for i,name in enumerate(names):
        packet=by_id[name];sheet=Image.new('RGB',(1440,720),(20,20,20));draw=ImageDraw.Draw(sheet)
        draw.text((8,8),f"Signal light state probe | {backend} | {name} | frame={packet['source_frame']} | fixed-view comparison",fill='white')
        for col,camera in enumerate(order):
            with Image.open(packet['cameras'][camera]['rgb']) as source:
                im=source.convert('RGB');im.thumbnail((480,320));sheet.paste(im,(480*col,40))
        with Image.open(packet['cameras'][cam]['rgb']) as source:
            crop=source.convert('RGB').crop(box);colors[name]=np.asarray(crop).copy();zoom=crop.resize((320,320),Image.Resampling.NEAREST)
            sheet.paste(zoom,(8,384));tile=Image.new('RGB',(256,296),(20,20,20));ImageDraw.Draw(tile).text((8,8),name,fill='white');tile.paste(crop.resize((256,256),Image.Resampling.NEAREST),(0,32));panels.append(tile)
        draw.text((350,400),target+' | front projected emitter ROI; nearest-neighbour pixel enlargement',fill='white')
        draw.text((350,430),'Calibrated 3D lens surfaces in shared raster; original scene parameters preserved.' if target.startswith('signal:') else 'Existing native candidate Gaussian RGB changed; geometry, alpha and weights preserved.',fill='white')
        draw.text((350,460),'Photographic lens registration; rendered appearance requires review. No traffic-rule controller.' if target.startswith('signal:') else 'Candidate mask / three-bulb semantics require visual review. No traffic-rule controller.',fill='white')
        draw.text((350,490),'These are separate state comparisons at the same source time; each held 2s for viewing.',fill='white')
        sheet.save(frames/f'{i:05d}.png')
    # Camera/ego transforms must be identical across static color states.
    checks['same_camera_and_ego_pose']=all(
        o['timestamp_s']==original['timestamp_s'] and np.allclose(o['ego_to_world'],original['ego_to_world'],atol=1e-9,rtol=0) and
        all(np.allclose(o['cameras'][c]['camera_to_world'],original['cameras'][c]['camera_to_world'],atol=1e-9,rtol=0) and
            np.allclose(o['cameras'][c]['K'],original['cameras'][c]['K'],atol=1e-9,rtol=0) for c in order)
        for o in [by_id[name] for name in names])
    changed={};rgb_delta={}
    for a,b in [('red_view','green_view'),('red_view','off_view'),('yellow_view','green_view')]:
        key=a+'_vs_'+b;changed[key]=int(np.any(colors[a]!=colors[b],axis=-1).sum())
        rgb_delta[key]=(colors[a].astype(float)-colors[b].astype(float)).mean(axis=(0,1)).tolist()
    checks['colors_change_inside_projected_roi']=all(v>0 for v in changed.values())
    lens_values={}
    if target.startswith('signal:'):
        lenses=entry.get('projected_lenses',{}).get(cam,[])
        for name in names:
            with Image.open(by_id[name]['cameras'][cam]['rgb']) as source:rgb=np.asarray(source).astype(float)
            yy,xx=np.indices(rgb.shape[:2]);values={}
            for lens in lenses:
                u,v=lens['pixel'];radius=max(1.,lens['radius_px']*.4);mask=(xx-u)**2+(yy-v)**2<=radius**2
                values[lens['state']]=rgb[mask].mean(0).tolist() if mask.any() else [999.]*3
            lens_values[name]=values
        checks.update(lens_color_checks(lens_values))
    contact=Image.new('RGB',(256*len(panels),296))
    for i,panel in enumerate(panels):contact.paste(panel,(256*i,0))
    contact.save(out/'signal_closeups.png')
    video=out/'signal_color_comparison.mp4';ffmpeg=ffmpeg_path()
    subprocess.run([str(ffmpeg),'-v','error','-threads','1','-filter_threads','1','-filter_complex_threads','1','-n','-framerate','0.5','-i',str(frames/'%05d.png'),'-r','10','-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-crf','20',str(video)],check=True)
    subprocess.run([str(ffmpeg),'-v','error','-threads','2','-i',str(video),'-f','null','-'],check=True)
    phase=[]
    for name,state in [('phase_red','red'),('phase_yellow','yellow'),('phase_green','green')]:
        if name in by_id:
            signal=next((p for p in by_id[name]['traffic_signals']['programs'] if p['object_id']==target),None)
            phase.append(dict(request_id=name,actual=signal['state'] if signal else None,expected=state,timestamp_s=by_id[name]['timestamp_s']))
    checks['declared_program_states_match']=len(phase)==3 and all(r['actual']==r['expected'] for r in phase)
    report=dict(passed=all(checks.values()),checks=checks,target=target,roi_xyxy=list(box),roi_changed_pixels=changed,
        roi_mean_rgb_deltas=rgb_delta,lens_inner_mean_rgb=lens_values,phase_samples=phase,video=str(video),contact_sheet=str(out/'signal_closeups.png'),
        execution_backend=backend,visual_selection_certified=False,three_bulb_semantics_verified=False,signal_rule_control=False)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report
