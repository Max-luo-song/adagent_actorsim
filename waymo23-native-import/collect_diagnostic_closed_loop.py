"""Collect diagnostic closed-loop results and encode explicitly labelled videos."""
import json,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'outputs/diagnostic_closed_loop_20261021'
ROOT=BASE/'outputs/gpu_test_service/jobs/diagloop-172-pair-20261021-v1'
FFMPEG=BASE/'code/HUGSIM/.pixi/envs/default/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'


def video(kind):
    gifs={role:Image.open(ROOT/'result'/role/(kind+'.gif')) for role in ('baseline','edited')}
    w,h=gifs['baseline'].size
    if gifs['edited'].size!=(w,h):raise ValueError('Video dimensions differ')
    vertical=kind=='three_views';size=(w,2*h+64) if vertical else (2*w,h+64)
    counts={role:im.n_frames for role,im in gifs.items()};count=max(counts.values());target=OUT/(kind+'_closed_loop_comparison.mp4')
    with (OUT/(kind+'_encoding.log')).open('w') as log:
        p=subprocess.Popen([str(FFMPEG),'-y','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{size[0]}x{size[1]}','-framerate','10','-i','pipe:0','-an','-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE,stderr=log)
        try:
            for i in range(count):
                canvas=Image.new('RGB',size,'#17202a');d=ImageDraw.Draw(canvas)
                d.text((8,4),'DIAGNOSTIC CLOSED LOOP | N1 SUSPENDED | NOT A CERTIFIED EVALUATION',fill='white')
                d.text((8,20),'Independent control feedback: camera poses may differ between runs.',fill='white')
                for j,role in enumerate(('baseline','edited')):
                    im=gifs[role];im.seek(min(i,im.n_frames-1));x,y=(0,64+j*h) if vertical else (j*w,64)
                    canvas.paste(im.convert('RGB'),(x,y))
                    label=('BASELINE / TOP' if j==0 else 'EDITED / BOTTOM') if vertical else role.upper()
                    if i>=im.n_frames:label+=' END / LAST FRAME HELD'
                    d.text((x+8,y+4),label,fill='yellow')
                p.stdin.write(np.asarray(canvas).tobytes())
            p.stdin.close()
            if p.wait(timeout=180):raise RuntimeError('Video encoding failed')
        finally:
            if p.poll() is None:p.terminate();p.wait(timeout=15)
    checked=subprocess.run([str(FFMPEG),'-v','error','-i',str(target),'-progress','pipe:1','-nostats','-f','null','-'],capture_output=True,text=True,timeout=90)
    frames=[int(l.split('=')[1]) for l in checked.stdout.splitlines() if l.startswith('frame=')]
    if checked.returncode or not frames or frames[-1]!=count:raise RuntimeError('Video decode/frame check failed')
    return dict(path=str(target),decoded_frames=count,original_frame_counts=counts,bytes=target.stat().st_size)


def finish():
    report=json.loads((ROOT/'result/experiment_report.json').read_text());rows={}
    for role in ('baseline','edited'):
        folder=ROOT/'result'/role;summary=json.loads((folder/'closed_loop_report.json').read_text());trace=json.loads((folder/'trace.json').read_text())
        if not summary.get('diagnostic_closed_loop') or not summary.get('n1_validation_suspended'):raise ValueError('Missing diagnostic scope')
        rows[role]=dict(status=summary['status'],steps=len(trace),emergency_braking_steps=summary['emergency_braking_steps'],
                        first_endpoint_xy=trace[0]['neural_plan'][-1] if trace else None,last_endpoint_xy=trace[-1]['neural_plan'][-1] if trace else None,
                        final_speed_mps=trace[-1]['speed_after'] if trace else None,
                        reference_overlap_detected=any(r.get('inserted_reference_overlap') for r in trace),
                        temporal_audit_frames=len(list(folder.glob('step_*/temporal_state.json'))))
    outputs=[]
    for kind in ('front','three_views'):
        if all((ROOT/'result'/r/(kind+'.gif')).exists() for r in rows):outputs.append(video(kind))
    result=dict(status='complete',diagnostic_only=True,n1_validation_suspended=True,n1_certified=False,
                checkpoint_unchanged=report.get('checkpoint_unchanged'),implementation_unchanged=report.get('implementation_unchanged'),
                perception_accuracy_proven=False,reliable_avoidance_proven=False,runs=rows,videos=outputs)
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    lines=['# 172 诊断闭环对照结果','','N1按用户要求暂停；保留MPC约束、预测占据和紧急制动。不是正式评测或泛化成功证明。','']
    for role,row in rows.items():lines += [f"- {role}: {row['steps']}步，紧急制动{row['emergency_braking_steps']}步，末速{row['final_speed_mps']}m/s，状态{row['status']}。"]
    lines += ['','视频：']+[f"- [{Path(v['path']).name}]({Path(v['path']).name})，{v['decoded_frames']}帧。" for v in outputs]
    (OUT/'results.md').write_text('\n'.join(lines))
    note='\n\n## 172诊断闭环已结束（自动同步）\n\n结果与视频：[诊断闭环结果](outputs/diagnostic_closed_loop_20261021/results.md)。N1暂停，不能作正式评测通过证据。\n'
    for name in ('cli-notes.md','DIAGNOSTIC_CLOSED_LOOP.md'):
        path=BASE/name
        if path.exists() and '## 172诊断闭环已结束（自动同步）' not in path.read_text():path.write_text(path.read_text()+note)
    print('Diagnostic loop collected and videos verified.',flush=True)


if __name__=='__main__':
    deadline=time.monotonic()+7200
    while True:
        path=ROOT/'status.json';status=json.loads(path.read_text()) if path.exists() else dict(status='queued')
        (OUT/'monitor_status.json').write_text(json.dumps(status,indent=2))
        if status['status'] in ('passed','failed','error'):
            try:finish()
            except Exception as exc:
                (OUT/'collector_error.txt').write_text(repr(exc));raise
            break
        if time.monotonic()>=deadline:print('Collector wait budget exhausted; job remains queued/running.',flush=True);break
        time.sleep(15)
