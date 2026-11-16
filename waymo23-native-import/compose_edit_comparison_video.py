"""Compose same-pose edit previews; labels explicitly exclude closed-loop claims."""
import argparse,json,pickle,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
BASE=Path(__file__).resolve().parents[1]
FFMPEG=BASE/'code/HUGSIM/.pixi/envs/default/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
CAMERAS=('CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT')


def compose(output):
    output=Path(output).resolve();plan=json.loads((output/'plan.json').read_text());jobs={j['role']:BASE/'outputs/gpu_test_service/jobs'/j['id'] for j in plan['jobs']}
    for root in jobs.values():
        if json.loads((root/'status.json').read_text())['status']!='passed':raise ValueError('Render did not pass')
    sequences={role:json.loads((root/'result/sequence_report.json').read_text()) for role,root in jobs.items()}
    a,b=sequences['baseline']['frames'],sequences['edited']['frames']
    if len(a)!=len(b) or len(a)!=plan['frames']:raise ValueError('Frame counts differ')
    sizes={'comparison_front':(1920,688),'comparison_three_views':(1440,720),'baseline_front':(960,688),'edited_front':(960,688)}
    writers={};logs={};audit=[];began=time.monotonic()
    try:
        for name,(w,h) in sizes.items():
            logs[name]=(output/(name+'_encoding.log')).open('w')
            writers[name]=subprocess.Popen([str(FFMPEG),'-y','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate',str(plan['fps']),'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output/(name+'.mp4'))],stdin=subprocess.PIPE,stderr=logs[name])
        for left,right in zip(a,b):
            if left['frame']!=right['frame'] or not np.isclose(left['timestamp'],right['timestamp']):raise ValueError('Frame/time mismatch')
            loaded={}
            for role,record in [('baseline',left),('edited',right)]:
                with Path(record['packet']).open('rb') as f:loaded[role]=pickle.load(f)
            controls,info_a=loaded['baseline'];edited,info_b=loaded['edited']
            for field in ('explicit_l2g_r_mat','explicit_l2g_t'):
                if not np.allclose(info_a[field],info_b[field],rtol=0,atol=1e-10):raise ValueError('Pose mismatch')
            for camera in CAMERAS:
                ca,cb=info_a['cam_params'][camera],info_b['cam_params'][camera]
                if ca['intrinsic']!=cb['intrinsic'] or not np.allclose(ca['l2c'],cb['l2c'],rtol=0,atol=1e-10):raise ValueError('Camera calibration mismatch')
            frame=left['frame'];stamp=left['timestamp'];header=f'OPEN-LOOP PREVIEW | NOT CLOSED LOOP | N1 NOT CERTIFIED | frame={frame} t={stamp:.1f}s'
            front=Image.new('RGB',sizes['comparison_front'],'#17202a');draw=ImageDraw.Draw(front);draw.text((12,5),header,fill='white');draw.text((12,25),'BASELINE: no inserted car',fill='white');draw.text((972,25),'EDITED: calibrated car',fill='white')
            front.paste(Image.fromarray(controls['rgb']['CAM_FRONT']),(0,48));front.paste(Image.fromarray(edited['rgb']['CAM_FRONT']),(960,48))
            grid=Image.new('RGB',sizes['comparison_three_views'],'#17202a');d=ImageDraw.Draw(grid)
            for role,y,label in [('baseline',0,'BASELINE: no inserted car'),('edited',360,'EDITED: calibrated car')]:
                obs=loaded[role][0];d.text((10,y+3),header,fill='white')
                for col,camera in enumerate(CAMERAS):
                    d.text((col*480+10,y+22),label+' | '+camera,fill='white');grid.paste(Image.fromarray(obs['rgb'][camera]).resize((480,320)),(col*480,y+40))
            frames={'comparison_front':front,'comparison_three_views':grid}
            for role in ('baseline','edited'):
                im=Image.new('RGB',sizes[role+'_front'],'#17202a');di=ImageDraw.Draw(im);di.text((10,5),header,fill='white');di.text((10,25),role.upper()+' | SAME RECORDED CAMERA POSES',fill='white');im.paste(Image.fromarray(loaded[role][0]['rgb']['CAM_FRONT']),(0,48));frames[role+'_front']=im
            for name,im in frames.items():writers[name].stdin.write(np.asarray(im).tobytes())
            audit.append(dict(frame=frame,timestamp=stamp,pose_and_cameras_equal=True))
            if frame==a[0]['frame']:
                front.save(output/'comparison_front_first_frame.jpg');grid.save(output/'comparison_three_views_first_frame.jpg')
        for name,p in writers.items():
            p.stdin.close()
            if p.wait(timeout=120)!=0:raise RuntimeError('Encoding failed: '+name)
    finally:
        for p in writers.values():
            if p.poll() is None:p.terminate();p.wait(timeout=15)
        for log in logs.values():log.close()
    files=[]
    for name in sizes:
        path=output/(name+'.mp4');checked=subprocess.run([str(FFMPEG),'-v','error','-i',str(path),'-progress','pipe:1','-nostats','-f','null','-'],capture_output=True,text=True,timeout=90)
        decoded=[int(line.split('=')[1]) for line in checked.stdout.splitlines() if line.startswith('frame=')]
        if checked.returncode or not decoded or decoded[-1]!=len(a):raise RuntimeError('Video decode/frame verification failed: '+name)
        files.append(dict(path=str(path),bytes=path.stat().st_size,decoded_frames=decoded[-1]))
    report=dict(status='complete',closed_loop=False,recorded_camera_poses=True,original_sensor_video=False,n1_certified=False,frames=len(a),fps=plan['fps'],duration_seconds=len(a)/plan['fps'],control_actions_executed=False,files=files,frame_checks=audit,composition_seconds=time.monotonic()-began)
    (output/'video_report.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--wait-for-renders',action='store_true');args=p.parse_args();out=Path(args.output)
    if args.wait_for_renders:
        plan=json.loads((out/'plan.json').read_text());deadline=time.monotonic()+3000
        while True:
            states=[]
            for j in plan['jobs']:
                path=BASE/'outputs/gpu_test_service/jobs'/j['id']/'status.json'
                states.append(json.loads(path.read_text())['status'] if path.exists() else 'queued')
            if any(s in ('failed','error') for s in states):raise RuntimeError('Render failed; video not mislabelled as complete')
            if all(s=='passed' for s in states):break
            if time.monotonic()>deadline:raise TimeoutError('Render waiting budget ended')
            time.sleep(10)
    result=compose(out);print(json.dumps(dict(status=result['status'],files=result['files']),indent=2),flush=True)
