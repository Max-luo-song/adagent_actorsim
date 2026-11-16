"""Edit-aware data checks and conservative stale-actor pixel quarantine."""
import numpy as np

def camera_row_header(frame,camera,kind,split):
    if type(frame) is not int or frame<0 or type(camera) is not int or camera not in (0,1,2):raise ValueError('Invalid source frame/camera identity')
    if kind not in ('original','novel') or split not in ('train','eval'):raise ValueError('Invalid row variant/split')
    return dict(id=f'{frame:04d}_{camera}_{kind}',source_frame=frame,camera_id=camera,kind=kind,split=split)

def outside_box_mask(shape,K,camera_to_world,center,rotation,size,padding=12):
    """Conservative 2-D ROI; near-plane clip box edges before projection."""
    h,w=shape;corners=np.array([[x,y,z] for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]])*np.asarray(size)
    world=corners@np.asarray(rotation).T+center;inv=np.linalg.inv(camera_to_world)
    cp=world@inv[:3,:3].T+inv[:3,3];near=.05;points=[p for p in cp if p[2]>=near]
    for a in range(8):
        for bit in [1,2,4]:
            b=a^bit
            if a>=b:continue
            if (cp[a,2]<near)!=(cp[b,2]<near):
                t=(near-cp[a,2])/(cp[b,2]-cp[a,2]);points.append(cp[a]+t*(cp[b]-cp[a]))
    mask=np.ones((h,w),bool)
    if not points:return mask
    p=np.array(points);uv=p@np.asarray(K).T;uv=uv[:,:2]/uv[:,2,None]
    lo=np.maximum(np.floor(uv.min(0)).astype(int)-padding,0);hi=np.minimum(np.ceil(uv.max(0)).astype(int)+padding+1,[w,h])
    if np.all(hi>lo):mask[lo[1]:hi[1],lo[0]:hi[0]]=False
    return mask

def validate_rows(rows,program_sha,helpers):
    if not rows or {r['kind'] for r in rows}!={'novel','original'} or {r['split'] for r in rows}!={'train','eval'}:raise ValueError('Both scene variants and train/eval required')
    ids=set();split={};source=set()
    for row in rows:
        key=(row['source_frame'],row['camera_id'],row['kind'])
        if key in ids:raise ValueError('Duplicate supervision row')
        ids.add(key)
        if split.setdefault(row['source_frame'],row['split'])!=row['split']:raise ValueError('Temporal train/eval leakage')
        edited=row['kind']=='novel'
        if row['edit_revision']!=int(edited) or row['scene_variant']!=('position_swapped' if edited else 'original'):raise ValueError('Wrong scene variant/edit revision')
        if row['edit_program_sha256']!=program_sha:raise ValueError('Mixed edit programs')
        if edited and not(row['reference_sanitized'] is True and row['dynamic_loss_masked'] is True):raise ValueError('Missing edited-reference guards')
        helpers.pose(row['ego_to_world']);helpers.pose(row['camera_to_world'])
        k=np.asarray(row['K'])
        if k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0:raise ValueError('Invalid camera calibration')
        for field in ['before','reference','mask','reference_mask']:helpers.verify_asset(row[field])
        if helpers.image(row['before']['path']).shape!=helpers.image(row['reference']['path']).shape:raise ValueError('Image resolution mismatch')
        source.add((row['scene'],row['checkpoint_sha256']))
    if len(source)!=1:raise ValueError('Mixed scene/model supervision')

class DeferredSceneState:
    """Hold the native edit until backward completes; restore before next row/close."""
    def __init__(self,owner):self.owner=owner;self.context=None;self.key=None
    def select(self,frame,edited):
        if self.key==(frame,edited):return
        self.close()
        self.context=self.owner.at(frame,edited);self.context.__enter__();self.key=(frame,edited)
    def close(self):
        if self.context is not None:
            context=self.context;self.context=None;self.key=None;context.__exit__(None,None,None)
