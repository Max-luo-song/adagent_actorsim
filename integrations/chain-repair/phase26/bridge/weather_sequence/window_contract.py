"""Reject camera/time/mask ordering errors before batched TV attention."""
import numpy as np
CAMERAS=['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']


def validate_manifest(data):
    records=data['records'];frames=data['frames']
    if frames!=[80,81,82] or data['camera_order']!=CAMERAS or len(records)!=9:
        raise ValueError('expected 3 frames x 3 cameras')
    for i,r in enumerate(records):
        f=frames[i//3];cam=i%3
        if (r['frame'],r['camera_id'],r['camera'])!=(f,cam,CAMERAS[cam]):
            raise ValueError('frame-major camera order mismatch')
        if not np.isfinite(r['timestamp_s']):raise ValueError('invalid time')
        pose=np.asarray(r['ego_to_world'],float)
        if pose.shape!=(4,4) or not np.isfinite(pose).all():raise ValueError('invalid ego pose')
        if i%3 and not np.allclose(pose,records[i-i%3]['ego_to_world']):
            raise ValueError('camera batch must share ego pose')
    timestamps=[records[i]['timestamp_s'] for i in (0,3,6)]
    if not np.allclose(np.diff(timestamps),.1):raise ValueError('expected contiguous 10Hz source times')
    for i in (0,3,6):
        if any(records[j]['timestamp_s']!=records[i]['timestamp_s'] for j in range(i,i+3)):
            raise ValueError('camera time mismatch')
    if data.get('scene_edits') or data.get('mask_valid_for_novel_or_edited_poses'):
        raise ValueError('recorded mask alignment is limited to unedited recorded views')
    return records
