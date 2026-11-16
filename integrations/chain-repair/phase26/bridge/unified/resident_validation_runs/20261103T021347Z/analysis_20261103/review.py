"""Independent CPU artifact review; does not render or start inference."""

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

import hashlib
import json
pass  # Path is provided by the portable boundary above
import pickle

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).resolve().parent
RUN=OUT.parent
ROOT=next(p for p in RUN.parents if (p/'code/HUGSIM').is_dir())


def read(path):return _hugsim_expand(json.loads(path.read_text()))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(array):return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    original={str(p.relative_to(RUN)):sha(p) for p in RUN.rglob('*')
              if p.is_file() and OUT not in p.parents and p.suffix in ('.json','.png','.pkl','.npz')}
    acceptance=read(RUN/'acceptance.json')
    episodes=read(RUN/'episodes.json')
    records=read(RUN/'inference_records.json')
    ready=read(RUN/'worker_ready.json')
    checks=dict(original_11_passed=len(acceptance['checks'])==11 and acceptance['passed'] and all(acceptance['checks'].values()),
        five_episodes_ten_records=len(episodes)==5 and len(records)==10,
        real_scene_order=[e['scene'] for e in episodes]==[552,552,551,552,552],
        different_scene_models=episodes[0]['checkpoint']!=episodes[2]['checkpoint'] and
            episodes[0]['gaussian_sample_sha256']!=episodes[2]['gaussian_sample_sha256'],
        return_model_restored=episodes[0]['gaussian_sample_sha256']==episodes[3]['gaussian_sample_sha256'],
        transition_assertions=all(all(e['transition'].values()) for e in episodes[:4]),
        same_backend_reset=episodes[1]['transition']['same_backend'] and episodes[1]['transition']['weather_reset_ack'])
    packet_checks=[];weather_checks=[];calibrations=[]
    for index,record in enumerate(records):
        ep=episodes[index//2];frame=ep['frames'][index%2]
        request=read(RUN/f'uniad_worker/request_{index:03d}.json')
        done=read(RUN/f'uniad_worker/done_{index:03d}.json')
        temporal=read(RUN/f'uniad_worker/step_{index:03d}/temporal_state.json')
        state=read(RUN/f'uniad_worker/step_{index:03d}/session_lifecycle.json')
        # These are our own persisted diagnostic packets, not external pickle data.
        with Path(frame['packet']).open('rb') as stream:obs,info=pickle.load(stream)
        hashes={cam:digest(image) for cam,image in obs['rgb'].items()}
        packet_checks.append(dict(index=index,episode=ep['name'],
            packet_path=request['packet']==frame['packet'],
            rgb_hashes=hashes==frame['rgb_sha256'],
            timestamp=info['timestamp']==frame['timestamp']==temporal['timestamp']==done['timestamp'],
            token=info['scene_token']==ep['generation']==temporal['scene_token'],
            done=done['status']=='ok' and done==record['done'],
            state_files=temporal==record['temporal'] and state==record['session'],
            identity=state['identity']==ready['resident_identity'],
            first_frame_reset=(state['reset']==(index%2==0)) and (index%2==1 or state['reset_state_cleared']),
            temporal_history=temporal['history_bev_present']==(index%2==1),
            seed_protocol=(request.get('diagnostic_seed')==20261021 if index%2==0 else 'diagnostic_seed' not in request)))
        if index%2==0:calibrations.append(info['cam_params'])
        if index<8:
            eindex=index//2;frame_index=index%2
            load_index=0 if eindex<2 else eindex-1
            weather_index=index if eindex<2 else frame_index
            weather=read(RUN/f'weather_load_{load_index}/report.json')['records'][weather_index]
            for cam,image in obs['rgb'].items():
                folder=RUN/f'weather_load_{load_index}/step_{weather_index:03d}'
                with Image.open(folder/f'{cam}_weather.png') as im:png=np.array(im.convert('RGB'))
                with Image.open(folder/f'{cam}_clear.png') as im:clear=np.array(im.convert('RGB'))
                with Image.open(folder/f'{cam}_mask.png') as im:mask=np.array(im)>0
                weather_checks.append(dict(index=index,camera=cam,
                    packet_png_equal=bool(np.array_equal(image,png)),
                    report_hash=digest(png)==weather['cameras'][cam]['rgb_sha256'],
                    clear_hash=digest(clear)==weather['cameras'][cam]['clear_sha256'],
                    dynamic_pixels_preserved=bool(np.array_equal(clear[mask],png[mask])),
                    history_window=weather['audit']['window_frames']==([80,80,80] if index%2==0 else [80,80,81])))
    checks['all_30_serialized_rgb_hashes']=len(packet_checks)==10 and all(c['rgb_hashes'] for c in packet_checks)
    checks['requests_done_and_state_files']=all(all(v for k,v in c.items() if k not in ('index','episode')) for c in packet_checks)
    checks['all_24_real_weather_views']=len(weather_checks)==24 and all(all(v for k,v in c.items() if k not in ('index','camera')) for c in weather_checks)
    checks['single_worker_model_identity']=all(r['session']['identity']==ready['resident_identity'] for r in records) and ready['uniad_model_loads']==1
    checks['reset_count_sequence']=[r['session']['reset_count'] for r in records]==[1,1,2,2,3,3,4,4,5,5]
    checks['prior_state_existed_before_reset']=all(r['session']['before']['bev_present'] and r['session']['before']['tracking_present'] for r in records[2::2])
    checks['reset_clears_counters_pose_and_evidence']=all(
        not r['session']['after']['bev_present'] and not r['session']['after']['tracking_present']
        and not r['session']['after']['tracking_pose_present'] and not r['session']['after']['metadata_bev_present']
        and r['session']['after']['track_next_id']==r['session']['after']['reliability_next_id']==r['session']['after']['reliability_tracks']==0
        for r in records[::2])
    calibration_json=[json.dumps(c,sort_keys=True,default=lambda value:np.asarray(value).tolist()) for c in calibrations]
    checks['scene_calibration_changes_and_returns']=calibration_json[0]!=calibration_json[2] and calibration_json[0]==calibration_json[3]
    comparisons=[]
    for first,last in ((0,8),(1,9)):
        row={}
        with np.load(RUN/f'uniad_worker/step_{first:03d}/perception.npz') as a,np.load(RUN/f'uniad_worker/step_{last:03d}/perception.npz') as b:
            for key in ('trajectory','occupancy','candidate_boxes','candidate_scores','candidate_labels'):
                assert a[key].shape==b[key].shape
                row[key]=float(np.max(np.abs(a[key].astype(float)-b[key].astype(float)))) if a[key].size else 0.
        comparisons.append(row)
    checks['replay_output_differences_recomputed']=comparisons==acceptance['replay_output_differences']
    checks['replay_outputs_zero_difference']=all(v==0 for row in comparisons for v in row.values())
    # The last episode intentionally replays inputs; verify exact RGB and depth.
    replay_matches=[]
    for first,last in ((0,8),(1,9)):
        paths=[episodes[i//2]['frames'][i%2]['packet'] for i in (first,last)]
        with Path(paths[0]).open('rb') as stream:a,_=pickle.load(stream)
        with Path(paths[1]).open('rb') as stream:b,_=pickle.load(stream)
        replay_matches.append(all(np.array_equal(a[field][cam],b[field][cam]) for field in ('rgb','depth') for cam in a[field]))
    checks['replay_inputs_exact']=all(replay_matches)
    implementation=read(RUN/'implementation.json')
    source_checks={path:Path(path).is_file() and sha(Path(path))==expected for path,expected in implementation.items()}
    checks['current_source_fingerprints_match']=all(source_checks.values())
    checks['no_driving_or_formal_claim']=acceptance['closed_loop_driving'] is False and acceptance['formal_evaluation_admitted'] is False
    checks['original_evidence_unchanged']=all(sha(RUN/path)==h for path,h in original.items())
    summary=dict(passed=all(checks.values()),checks=checks,worker_identity=ready['resident_identity'],
        packet_checks=packet_checks,weather_view_checks=weather_checks,replay_output_differences=comparisons,
        source_checks=source_checks,original_evidence_sha256=original,
        closed_loop_driving=False,formal_evaluation_admitted=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    font=ImageFont.truetype(str(ROOT/'code/HUGSIM/.pixi/envs/default/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'),24)
    sheet=Image.new('RGB',(1920,600),'#10141b');draw=ImageDraw.Draw(sheet)
    for col,ep in enumerate((episodes[0],episodes[2])):
        path=Path(ep['frames'][0]['packet']).parent/'CAM_FRONT.png'
        with Image.open(path) as image:sheet.paste(image.convert('RGB').resize((960,540)),(col*960,60))
        draw.text((col*960+16,16),f'Scene {ep["scene"]} | actual weather observation | t=8.0s',font=font,fill='white')
    sheet.save(OUT/'scene552_vs_551.png')
    print(json.dumps(dict(passed=summary['passed'],checks=checks,replay_output_differences=comparisons),indent=2))
    if not summary['passed']:raise SystemExit(1)


if __name__=='__main__':main()
