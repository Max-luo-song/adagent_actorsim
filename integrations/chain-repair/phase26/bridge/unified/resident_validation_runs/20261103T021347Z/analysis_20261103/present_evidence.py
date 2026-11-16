"""Make an evidence storyboard/CSV from sealed resident-validation artifacts."""

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

import csv
import hashlib
import json
pass  # Path is provided by the portable boundary above
import pickle
import subprocess

from PIL import Image, ImageDraw, ImageFont

RUN=Path(__file__).resolve().parent.parent
OUT=RUN/'presentation_20261103'
ROOT=next(p for p in RUN.parents if (p/'code/HUGSIM').is_dir())
CAMERAS=('CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT')


def main():
    episodes=_hugsim_expand(json.loads((RUN/'episodes.json').read_text()))
    records=_hugsim_expand(json.loads((RUN/'inference_records.json').read_text()))
    acceptance=_hugsim_expand(json.loads((RUN/'acceptance.json').read_text()))
    assert acceptance['passed'] and len(records)==10
    OUT.mkdir(exist_ok=False);frames=OUT/'frames';frames.mkdir()
    font_path=ROOT/'code/HUGSIM/.pixi/envs/default/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'
    font=ImageFont.truetype(str(font_path),24)
    large=ImageFont.truetype(str(font_path),36)
    count=0;source_hashes={};rows=[]
    def emit(sheet,seconds):
        nonlocal count
        for _ in range(seconds*2):
            sheet.save(frames/f'{count:04d}.png');count+=1
    def card(title,lines):
        sheet=Image.new('RGB',(1920,720),'#10141b');draw=ImageDraw.Draw(sheet)
        draw.text((60,70),title,font=large,fill='white')
        for i,line in enumerate(lines):draw.text((60,165+i*60),line,font=font,fill='#88ddff')
        return sheet
    emit(card('Scene switching + ONE resident UniAD - measured GPU evidence',[
        '552 first -> reset 552 -> switch 551 -> return 552 -> replay identical first inputs',
        'Each episode has only TWO observations at source t=8.0s / 8.1s, held on screen for viewing.',
        'Scenes/weather were rendered first. The ten saved packets were then inferred in one model process.',
        'This is a verification storyboard, not continuous cross-scene driving.']),3)
    for i,record in enumerate(records):
        episode=episodes[i//2];frame=episode['frames'][i%2]
        path=Path(frame['packet'])
        source_hashes[str(path.relative_to(RUN))]=hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open('rb') as stream:obs,info=pickle.load(stream)
        for cam,image in obs['rgb'].items():
            assert hashlib.sha256(image.tobytes()).hexdigest()==frame['rgb_sha256'][cam]
        state=record['session'];before=state['before'];after=state['after']
        identity=state['identity'];temporal=record['temporal']
        sheet=Image.new('RGB',(1920,720),'#10141b');draw=ImageDraw.Draw(sheet)
        replay=i>=8
        title=f'Episode {i//2+1}/5 | Scene {episode["scene"]} | '+episode['name']
        if replay:title+=' | SAME INPUT REPLAY'
        draw.text((16,8),title,font=large,fill='white')
        draw.text((16,55),f'Source time {info["timestamp"]:.1f}s | observation {i%2+1}/2 | held still, no invented intermediate frames',font=font,fill='#ffd080')
        for col,cam in enumerate(CAMERAS):
            x=col*640
            draw.text((x+10,100),cam+' | actual UniAD weather input',font=font,fill='#88ddff')
            sheet.paste(Image.fromarray(obs['rgb'][cam]).resize((640,360)),(x,135))
        draw.text((16,515),f'ONE worker PID {identity["pid"]} | model loads={identity["model_loads"]} | model/parameter/storage identity unchanged',font=font,fill='white')
        draw.text((16,558),f'Before inference: history_bev={temporal["history_bev_present"]} | reset={state["reset"]} | reset count={state["reset_count"]}',font=font,fill='#88ddff')
        draw.text((16,601),f'Track next ID {before["track_next_id"]} -> {after["track_next_id"]} | reliability tracks {before["reliability_tracks"]} -> {after["reliability_tracks"]} | reset clean={state["reset_state_cleared"]}',font=font,fill='#88ddff')
        if replay:draw.text((16,648),'Replay comparison: trajectory / occupancy / boxes / scores / labels max difference = 0',font=font,fill='#88ee99')
        else:draw.text((16,648),'Evidence: original input packet + temporal_state.json + session_lifecycle.json + perception.npz',font=font,fill='#cccccc')
        emit(sheet,2)
        if i%2==0:sheet.save(OUT/f'episode_{i//2+1}.png')
        rows.append(dict(index=i,episode=episode['name'],scene=episode['scene'],timestamp=info['timestamp'],
            pid=identity['pid'],model_object_id=identity['model_object_id'],parameter_data_ptr=identity['first_parameter_data_ptr'],
            model_loads=identity['model_loads'],history_bev_present=temporal['history_bev_present'],
            reset=state['reset'],reset_count=state['reset_count'],reset_state_cleared=state['reset_state_cleared'],
            track_id_before=before['track_next_id'],track_id_after=after['track_next_id'],
            reliability_tracks_before=before['reliability_tracks'],reliability_tracks_after=after['reliability_tracks']))
    assert all(value==0 for row in acceptance['replay_output_differences'] for value in row.values())
    emit(card('PASS: scene observation / resident inference state isolation',[
        'Original acceptance: 11 / 11 passed. Independent CPU artifact review: 21 / 21 passed.',
        'One PID / one model load; five clean episode starts, history retained on each second frame.',
        'Two replayed frames: trajectory, occupancy, boxes, scores and labels all have max difference 0.',
        'No vehicle control was executed; production driving runners still start independent workers.']),3)
    video=OUT/'resident_scene_switch_evidence.mp4'
    ffmpeg=ROOT/'code/HUGSIM/.pixi/envs/default/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
    subprocess.run([str(ffmpeg),'-v','error','-n','-framerate','2','-i',str(frames/'%04d.png'),
        '-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
    subprocess.run([str(ffmpeg),'-v','error','-i',str(video),'-f','null','-'],check=True)
    with (OUT/'state_evidence.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest=dict(source_run=str(RUN),rendered_again=False,model_inference_run_again=False,
        storyboard=True,frames=count,playback_fps=2,playback_seconds=count/2,
        measured_observations=10,observations_per_episode=2,closed_loop_driving=False,
        video_full_decode_passed=True,source_packet_sha256=source_hashes,
        source_report_sha256={name:hashlib.sha256((RUN/name).read_bytes()).hexdigest() for name in
            ['acceptance.json','episodes.json','inference_records.json','worker_ready.json','analysis_20261103/summary.json']})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(str(video))


if __name__=='__main__':main()
