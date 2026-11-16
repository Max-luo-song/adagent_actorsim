"""Post-hoc comparison across separate output roots; never rewrites completed A."""

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
import numpy as np
from common import read,write,asset

def verify_actor_sync(root,tracks,session,receipt):
    root=Path(root);plan=read(root/'plan.json');a=Path(plan['baseline_root'])/'A'
    with np.load(a/'actor_tracks.npz') as z:original={k:z[k].copy() for k in z.files}
    with np.load(root/'scenario/target_track.npz') as z:target={k:z[k].copy() for k in z.files}
    if set(tracks)!=set(original):raise ValueError('Actor schema changed')
    other_checks={}
    for k,value in tracks.items():
        x=value.copy();y=original[k].copy()
        if k in ('RigidNodes__instances_trans','RigidNodes__instances_quats'):
            x[:,3]=y[:,3]
        other_checks[k]=bool(np.array_equal(x,y))
    np.testing.assert_allclose(tracks['RigidNodes__instances_trans'][:,3],target['rendered_trans'],rtol=0,atol=1e-5)
    np.testing.assert_allclose(tracks['RigidNodes__instances_quats'][:,3],target['rendered_quats'],rtol=0,atol=1e-6)
    np.testing.assert_array_equal(tracks['RigidNodes__instances_trans'][:16,3],original['RigidNodes__instances_trans'][:16,3])
    m=session.backend.trainer.models['RigidNodes'];env=session.backend.env.models['RigidNodes']
    shared=all(env[k] is getattr(m,k) for k in ('instances_trans','instances_quats','instances_size','instances_fv'))
    checks=dict(other_actor_geometry_unchanged=all(other_checks.values()),single_program=list(session.vehicles.programs)==[3],
        renderer_environment_share_native_state=shared,activation_source_frame=receipt['activation_frame']==15,
        no_actor_insertion=len(tracks['RigidNodes__instances_size'])==15,
        canonical_export_matches_installed_program=True,prefix_effective_pose_unchanged=True)
    result=dict(passed=all(checks.values()),checks=checks,other_actor_array_checks=other_checks,
        installed_program=asset(root/'scenario/target_track.npz'),actual_collision_and_scoring_tracks=asset(root/'B/actor_tracks.npz'),
        scoring_uses_exported_effective_tracks=True,camera_effect_requires_actual_render_comparison=True)
    write(root/'B/actor_sync.json',result)
    if not result['passed']:raise ValueError('Native actor synchronization failed')
    return result

def array_difference(p,q,keys=None,tolerance=1e-4):
    with np.load(p) as x,np.load(q) as y:
        if set(x.files)!=set(y.files):return True
        for k in keys if keys is not None else x.files:
            if k not in x or k not in y:continue
            a,b=x[k],y[k]
            if a.shape!=b.shape:return True
            if a.dtype.kind in 'fc':
                if not np.allclose(a,b,rtol=0,atol=tolerance,equal_nan=False):return True
            elif not np.array_equal(a,b):return True
    return False

def compare(root):
    root=Path(root);plan=read(root/'plan.json');aroot=Path(plan['baseline_root'])/'A';broot=root/'B'
    reports={n:read(p/'run_report.json') for n,p in [('A',aroot),('B',broot)]}
    summaries={n:read(p/'summary.json') for n,p in [('A',aroot),('B',broot)]}
    a,b=reports['A'],reports['B'];ao={r['index']:r for r in a['observations']};bo={r['index']:r for r in b['observations']}
    at={r['index']:r for r in a['trace']};bt={r['index']:r for r in b['trace']}
    first={k:None for k in ['camera_pixels','detection','motion_prediction','raw_plan','adopted_reference','control','executed_state']}
    rows=[];prefix=[];direct=[]
    for i in sorted(set(ao)&set(bo)):
        x,y=ao[i],bo[i];dif={};same_pose=bool(np.allclose(x['pose'],y['pose'],rtol=0,atol=1e-9))
        dif['camera_pixels']=x['image_sha256']!=y['image_sha256']
        dif['executed_state']=not np.allclose(x['motion_state'],y['motion_state'],rtol=0,atol=1e-8)
        for label,key in [('raw_plan','raw_uniad'),('adopted_reference','adopted')]:
            dif[label]=i in at and i in bt and not np.allclose(at[i][key],bt[i][key],rtol=0,atol=1e-4)
        dif['control']=i in at and i in bt and not np.allclose([at[i]['action'][k] for k in sorted(at[i]['action'])],
            [bt[i]['action'][k] for k in sorted(bt[i]['action'])],rtol=0,atol=1e-5)
        for label,file,keys in [('detection','perception.npz',['boxes','scores','labels']),('motion_prediction','motion_head_raw.npz',None)]:
            p=aroot/f'uniad_worker/step_{i:03d}'/file;q=broot/f'uniad_worker/step_{i:03d}'/file
            dif[label]=array_difference(p,q,keys) if p.exists() and q.exists() else None
        for k,v in dif.items():
            if v and first[k] is None:first[k]=dict(index=i,local_time_s=i*.1,source_frame=i+10,same_actual_ego_pose=same_pose)
        if i<=5:
            gate=broot/f'steps/{i:03d}/prefix_gate.json'
            prefix.append(i if gate.exists() and not any(v for v in dif.values() if v is not None) else -1)
        if i>5 and same_pose and dif['camera_pixels']:direct.append(i)
        target=y['environment'].get('target_relative_forward_left_up_m')
        rows.append(dict(index=i,local_time_s=i*.1,source_frame=i+10,same_actual_ego_pose=same_pose,differences=dif,
            interpretation='direct intervention at equal actual ego pose' if same_pose else 'same clock, downstream viewpoints also differ',
            B_target_relative_forward_left_up_m=target,B_target_world=y['environment'].get('target_world'),
            A_speed_mps=x['motion_state'][3],B_speed_mps=y['motion_state'][3]))
    common_ok=prefix==list(range(6));sync=read(broot/'actor_sync.json')
    entered=[r['index'] for r in rows if r['B_target_relative_forward_left_up_m'] is not None and
        0<r['B_target_relative_forward_left_up_m'][0]<50 and abs(r['B_target_relative_forward_left_up_m'][1])<2.]
    gates=[r['road_gate']['certificate']['passed'] for r in b['trace']]
    map_departure=any(o['environment'].get('authorized_static_map_departure',False) for o in b['observations'])
    full=b['status']=='completed' and len(b['trace'])==plan['controls']
    anomaly=summaries['B']['any_anomaly'] or map_departure
    camera_proved=bool(direct) and common_ok and sync['passed']
    if summaries['A']['any_anomaly']:conclusion='A异常；先诊断基准，不提出切入补丁'
    elif not common_ok or not camera_proved or not entered:conclusion='差分或干预生效证据不足；结论未决'
    elif anomaly:conclusion='B出现异常；须复现并定位机制，尚不能采纳补丁'
    elif not full:conclusion='B观察不足或资源中止；结论未决'
    elif summaries['B']['stop_recovery_unresolved']:conclusion='观察窗内未见已定义异常；减速停车可能合理，恢复行驶仍未决'
    else:conclusion='B完成观察窗且未见已定义异常；不强行生成补丁'
    out=dict(schema='native551_map_v5_scene_differential',conclusion=conclusion,
        A_root=str(aroot),B_root=str(broot),first_divergences=first,common_prefix_complete=common_ok,
        common_prefix_observations=prefix,full_history_retained_at_intervention=True if common_ok else None,
        intervention_verified_in_camera_at_equal_ego_pose=camera_proved,direct_camera_difference_steps=direct,
        target_entered_actual_ego_front_corridor_steps=entered,actor_sync=asset(broot/'actor_sync.json'),
        B_full_source_time_window_completed=full,B_map_departure=map_departure,
        B_all_action_map_certificates=bool(gates) and all(gates),B_model_weights_unchanged=b.get('model_weights_unchanged'),
        summaries={n:asset(p/'summary.json') for n,p in [('A',aroot),('B',broot)]},
        native_executed_scores={n:((summaries[n].get('native_scores') or {}).get('scores') or {}).get('executed_path') for n in summaries},
        native_score_scope='Executed path uses recorded realized poses over actual 2.5s future windows. Raw/adopted scores separately evaluate proposed plans; neither temporal completion nor planning score implies route completion.',
        patch_generated=False,patch_adoption='none; evidence and repeat required',rows=rows)
    write(root/'comparison_A_B.json',out)
    return {k:v for k,v in out.items() if k not in ['rows','native_executed_scores']}

def artifacts(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw
    import imageio_ffmpeg,subprocess
    from common import verify
    root=Path(root);plan=read(root/'plan.json');stage=root/'presentation_A_B';stage.mkdir(exist_ok=True)
    arms={'A':Path(plan['baseline_root'])/'A','B':root/'B'}
    if (stage/'artifacts.json').exists():
        cached=read(stage/'artifacts.json')
        for a in [cached['video'],cached['overlay'],*cached['source_reports']]:verify(a)
        if not cached['full_decode_passed']:raise ValueError('Cached video decode unverified')
        return cached
    reports={n:read(p/'run_report.json') for n,p in arms.items()};fig,ax=plt.subplots(figsize=(12,4))
    for n,p in arms.items():
        r=reports[n];pose=np.array([o['pose'] for o in r['observations']]);ax.plot(pose[:,0,3],pose[:,1,3],label=n+' executed ego')
        with np.load(p/'actor_tracks.npz') as z:t=z['RigidNodes__instances_trans'][10:10+len(pose),3]
        ax.plot(t[:,0],t[:,1],'--',label=n+' native3 target')
    ax.set(xlabel='Source forward x (m)',ylabel='Source left y (m)',title='A original / B scripted cut-in, actual source clock10..198');ax.axis('equal');ax.legend();fig.tight_layout()
    fig.savefig(stage/'executed_trajectory_overlay.png',dpi=160);fig.savefig(stage/'executed_trajectory_overlay.svg');plt.close(fig)
    frames=stage/'frames';frames.mkdir(exist_ok=True);count=max(len(r['observations']) for r in reports.values())
    for i in range(count):
        sheet=Image.new('RGB',(1440,680),'#151515');d=ImageDraw.Draw(sheet)
        d.text((8,5),f'Local t={i*.1:.1f}s | source frame={i+10} | same time; actual ego viewpoints may differ',fill='white')
        for row,(n,p) in enumerate(arms.items()):
            d.text((8,30+row*320),n+(' ORIGINAL' if n=='A' else ' NATIVE3 CUT-IN')+' | actual closed-loop observations',fill='white')
            for col,cam in enumerate(['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT']):
                path=p/f'steps/{i:03d}'/(cam+'.png')
                if path.exists():
                    with Image.open(path) as im:im.thumbnail((480,280));sheet.paste(im,(col*480,55+row*320))
                else:d.text((col*480+10,150+row*320),'Arm ended; no fabricated observation',fill='white')
        sheet.save(frames/f'{i:04d}.png')
    video=stage/'synchronized_3CAM.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-n','-framerate','10','-i',str(frames/'%04d.png'),'-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(video)],check=True,timeout=180)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-threads','1','-i',str(video),'-f','null','-'],check=True,timeout=60)
    out=dict(video=asset(video),overlay=asset(stage/'executed_trajectory_overlay.png'),full_decode_passed=True,
        frames=count,source_reports=[asset(p/'run_report.json') for p in arms.values()])
    write(stage/'artifacts.json',out);return out
