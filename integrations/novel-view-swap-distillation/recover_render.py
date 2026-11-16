"""CPU-only recovery of complete rendered rows missing a top-level camera_id."""

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

import copy,re
pass  # Path is provided by the portable boundary above
import numpy as np
from contracts import read,write,split_frame
from guards import camera_row_header,validate_rows

def recover(output):
    from workflow import provider,configure,asset,verify_asset,pins,HERE,sha
    out=Path(output).resolve();archive=out/'camera_id_recovery/original'
    old=read(archive/'swap_contract.json');original_contract=asset(archive/'swap_contract.json')
    if (out/'camera_id_recovery/report.json').exists() and (out/'render/manifest.json').exists():
        configure(out)
        manifest=read(out/'render/manifest.json');validate_rows(manifest['rows'],old['program']['sha256'],provider().legacy)
        return dict(recovered=True,reused_existing_recovery=True,rows=len(manifest['rows']),extra_GPU_renders=0)
    if any((out/p).exists() for p in ['pseudo_gt','structure/train','structure/evaluation']):raise ValueError('Downstream stages exist; this migration is limited to the failed render finalization')
    # Old implementation and rows were archived before any fix. No blanket pin bypass.
    allowed={'workflow.py','guards.py','test_guards.py','run.sh'}
    old_code=[]
    for path,expected in old['code'].items():
        previous=archive/'code'/Path(path).name
        if sha(previous)!=expected:raise ValueError('Original code snapshot disagrees with frozen contract: '+path)
        if Path(path).name not in allowed and sha(path)!=expected:raise ValueError('Unrelated implementation changed: '+path)
        old_code.append(asset(previous))
    for key in ['source_plan','source_preview','program','effective_actor_tracks','job','plan']:verify_asset(old[key])
    module=provider();module.configure(out);legacy=module.legacy;job=read(out/'job.json')
    compiled=read(out/'compiled_trajectory.json');expected={}
    for sample in compiled['samples']:
        split=split_frame(sample['source_frame'],job['frame_stride'])
        if split=='unused':continue
        for cam in [0,1,2]:
            for kind in ['original','novel']:
                header=camera_row_header(sample['source_frame'],cam,kind,split);expected[header['id']]=header
    # Reuse an actual GPU-established camera/image-ID mapping for the same source.
    source_plan=read(old['source_plan']['path']);source_job_asset=source_plan['assets']['source_job'];verify_asset(source_job_asset)
    source_root=Path(source_job_asset['path']).parent;source_job=read(source_job_asset['path'])
    for key in ['checkpoint','config','scene_manifest','source_report']:
        if source_job['inputs'][key]!=job['inputs'][key]:raise ValueError('Conditioning source belongs to another scene/model: '+key)
    if source_job['conditioning']!=job['conditioning']:raise ValueError('Appearance policy changed')
    source_manifest=read(source_root/'render/manifest.json');verify_asset(source_manifest['job'])
    mapping_asset=source_manifest['appearance_conditioning_sources'];verify_asset(mapping_asset);mapping=read(mapping_asset['path'])
    if mapping['global_lighting_fingerprint']!=job['conditioning']['global_lighting_fingerprint']:raise ValueError('Lighting model identity differs')
    previews=read(old['source_preview']['path'])['records'];previews={(r['source_frame'],r['camera_id'],r['variant']):r for r in previews}
    records=[];backups=[];seen=set();max_difference=0.
    for path in sorted((archive/'rows').glob('*.json')):
        row=read(path);audit=row['appearance_conditioning'];key=row['id']
        if key not in expected or key in seen or path.stem!=key:raise ValueError('Unexpected/duplicate rendered row: '+str(path))
        header=expected[key];seen.add(key)
        if any(row[k]!=header[k] for k in ['id','source_frame','kind','split']):raise ValueError('Stored identity disagrees with expected task')
        cam=audit['camera_id'];frame=row['source_frame']
        if type(cam) is not int or cam!=header['camera_id'] or audit['source_frame']!=frame:raise ValueError('Actual camera audit disagrees with row ID')
        if 'camera_id' in row and row['camera_id']!=cam:raise ValueError('Existing camera_id conflicts; not recoverable')
        if audit['image_id']!=mapping['camera_image_ids'][str(cam)][frame]:raise ValueError('Wrong GPU image ID')
        if audit['policy_sha256']!=job['conditioning']['policy_asset']['sha256'] or audit['global_lighting_fingerprint']!=mapping['global_lighting_fingerprint']:raise ValueError('Mixed appearance conditioning')
        row['camera_id']=cam
        for field in ['before','reference','mask','reference_mask','raw_reference']:verify_asset(row[field])
        prior=previews[(frame,cam,'swapped' if row['kind']=='novel' else 'original')];verify_asset(prior['image'])
        if not np.allclose(row['ego_to_world'],prior['ego_to_world'],rtol=0,atol=1e-4):raise ValueError('Pose differs from accepted preview')
        rgb=legacy.image(row['before']['path']);before=legacy.image(prior['image']['path'])
        if rgb.shape!=before.shape:raise ValueError('Preview image resolution differs')
        difference=float(np.abs(rgb.astype(float)-before).mean()/255.)
        if difference>.005 or not np.isclose(difference,row['source_preview_rgb_mean_difference'],atol=1e-10):raise ValueError('Rendered pixels differ from recorded preview check')
        mask=legacy.image(row['mask']['path'])[...,0]>0
        if not np.isclose(mask.mean(),row['valid_fraction'],atol=1e-10):raise ValueError('Mask coverage differs from actual saved mask')
        max_difference=max(max_difference,difference);records.append(row);backups.append(asset(path))
    if seen!=set(expected):raise ValueError('Incomplete render; missing '+str(sorted(set(expected)-seen)[:10]))
    validate_rows(records,old['program']['sha256'],legacy)
    mapping=copy.deepcopy(mapping);mapping['recovered_mapping_source']=mapping_asset
    mapping['recovery_note']='Same pinned source checkpoint/config/policy; every recovered row cross-checked against its actual GPU appearance audit.'
    stage=out/'render';write(stage/'conditioning_sources.json',mapping)
    for row in records:write(stage/'rows'/f'{row["id"]}.json',row)
    manifest=dict(completed=True,rows=records,revision='mixed explicit original=0/swapped=1',weather='original',
        job=asset(out/'job.json'),trajectory=asset(out/'compiled_trajectory.json'),gpu_executed=True,
        edit_program=old['program'],variant_contract='original replay + edited new views, explicit per-row revision',
        appearance_conditioning_sources=asset(stage/'conditioning_sources.json'),conditioning_policy=job['conditioning']['policy_asset'],
        recovered_finalization_on_CPU=True,extra_GPU_renders_for_recovery=0)
    write(stage/'manifest.json',manifest)
    report=dict(schema='missing_camera_id_recovery_v1',recovered=True,rows=len(records),expected_rows=len(expected),
        original_contract=original_contract,original_code=old_code,original_rows=backups,
        all_image_hashes_verified=True,embedded_actual_camera_audits_verified=True,mask_coverage_verified=True,
        preview_poses_and_pixels_verified=True,maximum_preview_rgb_difference=max_difference,
        conditioning_mapping_source=mapping_asset,extra_GPU_renders=0,training_executed=False,
        renderer_fix='Emit camera_id in production row header; persist conditioning source map before validation',
        new_code=pins(),recovered_manifest=asset(stage/'manifest.json'))
    write(out/'camera_id_recovery/report.json',report)
    updated=copy.deepcopy(old);updated['code']=pins();updated['metadata_recovery']=asset(out/'camera_id_recovery/report.json')
    write(out/'swap_contract.json',updated)
    configure(out)
    return {k:report[k] for k in ['recovered','rows','expected_rows','all_image_hashes_verified','maximum_preview_rgb_difference','extra_GPU_renders','training_executed']}
