"""Shared CPU planning guard: evidence tiers, metric rectangles and recovery.

No scene IDs, asset identities, training, or simulator truth. Neural outputs stay
unchanged. Raw occupancy is aligned to prediction time; proposals are
checked at their predicted time, without rasterizing them into every frame.
"""

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

from dataclasses import dataclass
import numpy as np

POLICY = 'truncated_surface_visibility_memory_v19'
GRID_HALF_DIAGONAL_M = np.sqrt(2)*.25


class ObstacleMemory:
    """Pose-compensated confirmed history; weak proposals cannot move anchors.

    New tracks remain probationary until repeated strong evidence. Confirmed
    tracks survive occlusion; uncertain single-frame tracks expire at max_gap.
    Motion is estimated from strong world observations, never low-score drift.
    """
    def __init__(self,ground_points_world=None,ground_signature=None):
        self.tracks=[];self.time=None;self.scene=None;self.next_id=0
        import json
        pass  # Path is provided by the portable boundary above
        planning=_hugsim_expand(json.loads((Path(__file__).parent/'configs/planning.json').read_text()))
        self.motion_horizon_s=6*float(planning['control_interval_s'])

        self.ground_tree=None;self.ground=None;self.ground_signature=ground_signature
        if ground_points_world is not None:
            from scipy.spatial import cKDTree
            points=np.asarray(ground_points_world,dtype=float)
            if points.ndim!=2 or points.shape[1]!=3 or len(points)<3 or not np.isfinite(points).all():
                raise ValueError('Invalid observed ground skin')
            self.ground=points;self.ground_tree=cKDTree(points[:,:2])

    @classmethod
    def from_source(cls,source):
        import json,hashlib
        pass  # Path is provided by the portable boundary above
        entry=source.get('planning_ground_skin')
        if entry is None:return cls()
        data=Path(entry['data']);meta=_hugsim_expand(json.loads(Path(entry['metadata']).read_text()))
        signature=hashlib.sha256(data.read_bytes()).hexdigest()
        if signature!=entry['sha256'] or meta.get('scene')!=source['scene'] or meta.get('source_kind')!='sensor' or meta.get('edited_asset_included') is not False or meta.get('coordinate_system')!='first_waymo_ego_z_up_meters':
            raise ValueError('Ground-skin provenance or scene mismatch')
        with np.load(data,allow_pickle=False) as f:points=f['ground_points'].astype(float)
        # source-world XY (forward,left) -> model-world XY (right,forward).
        return cls(np.column_stack([-points[:,1],points[:,0],points[:,2]]),signature)

    def update(self,report,boxes,scores,labels,info,current_occupancy=None):
        import copy
        result=copy.deepcopy(report);cfg=report.get('config',{})
        now=float(info['timestamp']);scene=info.get('scene_token')
        static_epoch=info.get('declared_static_epoch')
        if static_epoch is not None and (not isinstance(static_epoch,str) or not static_epoch):raise ValueError('Invalid static snapshot epoch')
        result['declared_static_epoch']=static_epoch
        rotation=np.asarray(info['explicit_l2g_r_mat'],dtype=float)
        translation=np.asarray(info['explicit_l2g_t'],dtype=float)
        if rotation.shape!=(3,3) or translation.shape!=(3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all() or not np.isfinite(now):
            raise ValueError('Invalid obstacle memory pose')
        max_gap=float(cfg.get('max_gap_seconds',.75));association=float(cfg.get('association_distance_m',2.))
        tolerance=float(cfg.get('obstacle_margin_m',.5));required=int(cfg.get('confirmation_observations',3));duration=float(cfg.get('confirmation_seconds',.2))
        reset=self.time is not None and (scene!=self.scene or now<=self.time or now-self.time>max_gap)
        if reset:self.tracks=[]
        self.time=now;self.scene=scene
        boxes=np.asarray(boxes,dtype=float);scores=np.asarray(scores);labels=np.asarray(labels)
        if boxes.ndim!=2 or boxes.shape[1]!=9 or len(boxes)!=len(scores) or len(scores)!=len(labels):
            raise ValueError('Invalid obstacle memory candidates')
        valid=np.isfinite(boxes).all(1)&np.isfinite(scores)&(boxes[:,3:6]>=cfg.get('min_dimension_m',.2)).all(1)&(boxes[:,3:6]<=cfg.get('max_dimension_m',30.)).all(1)
        if static_epoch is not None:
            boxes=boxes.copy();boxes[:,7:9]=0.
            for obstacle in result.get('obstacles',[]):
                current_box=np.asarray(obstacle['box'],dtype=float).copy()
                obstacle.setdefault('neural_box',current_box.tolist())
                current_box[7:9]=0.;obstacle['box']=current_box.tolist()
                obstacle['declared_static_epoch']=static_epoch
                obstacle['geometry_motion_evidence']=dict(velocity_source='declared_static_snapshot')
        world=boxes[:,:3]@rotation.T+translation
        rejected=[]
        if self.ground_tree is not None:
            distances,indices=self.ground_tree.query(np.where(np.isfinite(world[:,:2]),world[:,:2],1e9),k=3)
            known=distances[:,-1]<=.5  # existing observed-skin support radius
            heights=np.median(self.ground[indices,2],axis=1)
            discrepancy=np.abs(world[:,2]-heights)
            bad=valid & known & (discrepancy>boxes[:,5]/2+tolerance)
            for obstacle in result.get('obstacles',[]):
                i=obstacle['candidate_index']
                if 0<=i<len(boxes) and bad[i]:
                    rejected.append(dict(candidate_index=i,raw_score=obstacle['raw_score'],
                        reason='bottom inconsistent with observed local ground',ground_error_m=float(discrepancy[i]),
                        allowed_error_m=float(boxes[i,5]/2+tolerance)))
            result['obstacles']=[o for o in result.get('obstacles',[]) if not (0<=o['candidate_index']<len(boxes) and bad[o['candidate_index']])]
        result['geometry_rejections']=rejected
        current={o['candidate_index']:o for o in result.get('obstacles',[])}
        strong={i:o for i,o in current.items() if 0<=i<len(boxes) and valid[i] and
                ((o['state']=='network_retained' and scores[i]>=report.get('detection_threshold',.4)) or
                 (o['state']=='persistent_uncertain' and o.get('observations',0)>=required and o.get('duration_seconds',0)>=duration-1e-6))}
        if static_epoch is not None:
            for track in self.tracks:track['velocity']=np.zeros(3)
        # Globally rank pairs, instead of letting old tracks greedily claim weak
        # candidates ahead of genuinely observed identities.
        pairs=[]
        for j,track in enumerate(self.tracks):
            predicted=track['world']+track['velocity']*(now-track['time']) if track['confirmed'] else track['world']
            for i in strong:
                ratio=boxes[i,3:6]/track['box'][3:6]
                distance=float(np.linalg.norm(world[i]-predicted))
                if labels[i]==track['label'] and (.5<=ratio).all() and (ratio<=2.).all() and distance<=association:
                    pairs.append((distance,j,i))
        matches={};used=set()
        for _,j,i in sorted(pairs):
            if j not in matches and i not in used:matches[j]=i;used.add(i)
        retained=[];memory_rows=[];rejected_updates=0;rejected_velocities=0
        heading_support=OccupancyHeadingSupport(current_occupancy)
        for j,track in enumerate(self.tracks):
            index=matches.get(j);accepted=False
            if index is not None:
                history=[h for h in track['history'] if now-h[0]<=max_gap]+[(now,world[index].copy())]
                velocities=(track.get('velocity_history',[])+[rotation@np.r_[boxes[index,7:9],0.]])[-required:]
                track['velocity_history']=velocities
                observed_right=rotation@np.array([np.cos(boxes[index,6]),-np.sin(boxes[index,6]),0.])
                track['right_history']=(track.get('right_history',[track['right_world']])+[observed_right])[-required:]
                track['max_wl']=np.maximum(track.get('max_wl',track['box'][3:5]),boxes[index,3:5])
                # A high score alone cannot relocate a confirmed anchor on one
                # frame. Commit only a temporally consistent world-space window.
                if len(history)>=required and now-history[0][0]>=duration-1e-6:
                    recent=history[-max(required,3):];times=np.array([h[0] for h in recent]);positions=np.array([h[1] for h in recent]);dt=times-times[-1]
                    design=np.column_stack([dt,np.ones(len(dt))]);fit=np.linalg.lstsq(design,positions,rcond=None)[0]
                    residual=float(np.linalg.norm(positions-design@fit,axis=1).max())
                    spread=float(np.linalg.norm(positions-np.median(positions,axis=0),axis=1).max())
                    if residual<=tolerance:
                        reported_velocity=np.median(velocities,axis=0)
                        # A smooth depth/association drift also has a tiny line-fit
                        # residual. Require independent detector velocity agreement
                        # in displacement units before promoting that fit to motion.
                        velocity_difference=float(np.linalg.norm(fit[0]-reported_velocity))
                        disagreement_window=velocity_difference*float(np.ptp(times))
                        disagreement=velocity_difference*max(float(np.ptp(times)),self.motion_horizon_s)
                        velocity_fit_accepted=bool(static_epoch is None and spread>tolerance and disagreement<=tolerance)
                        velocity=np.zeros(3) if static_epoch is not None else (fit[0] if velocity_fit_accepted else reported_velocity)
                        rejected_velocities+=int(spread>tolerance and not velocity_fit_accepted)
                        track['motion_evidence']=dict(
                            velocity_source='declared_static_snapshot' if static_epoch is not None else ('position_fit_with_velocity_agreement' if velocity_fit_accepted else 'reported_velocity_consensus'),
                            position_fit_velocity_world=fit[0].tolist(),
                            reported_velocity_world=reported_velocity.tolist(),
                            fit_velocity_disagreement_m=disagreement,
                            observed_window_disagreement_m=disagreement_window,
                            prediction_horizon_s=self.motion_horizon_s,
                            velocity_fit_accepted=velocity_fit_accepted,
                            position_fit_window_s=float(np.ptp(times)))
                        compensated=positions+velocity[None]*(times[-1]-times)[:,None]
                        compensated_spread=float(np.linalg.norm(compensated-np.median(compensated,axis=0),axis=1).max())
                        use_compensation=compensated_spread<spread
                        anchor=np.median(compensated if use_compensation else positions,axis=0) if spread<=tolerance else fit[1]
                        anchor_method=('motion_compensated_temporal_world_median' if use_compensation else 'temporal_world_median') if spread<=tolerance else 'temporal_world_linear_fit'
                        track.update(world=anchor,velocity=velocity,anchor_method=anchor_method,time=now,box=boxes[index].copy(),score=float(scores[index]),
                            confirmed=True,right_world=rotation@np.array([np.cos(boxes[index,6]),-np.sin(boxes[index,6]),0.]),position_uncertainty=max(residual,float(np.linalg.norm(anchor-world[index]))),radius=max(track['radius'],float(np.hypot(*boxes[index,3:5])/2)))
                        accepted=True
                    else:rejected_updates+=1
                track['history']=history;track['last_seen']=now
            age=now-track['time']
            if not track['confirmed'] and now-track['first']>max_gap and not accepted:continue
            local=rotation.T@(track['world']-translation)
            if np.linalg.norm(local[:2])>cfg.get('max_range_m',60.):continue
            retained.append(track)
            if index is not None and accepted:
                entry=current[index];entry['geometry_radius_m']=track['radius'];entry['geometry_position_uncertainty_m']=track['position_uncertainty']
                # The accepted world anchor must feed CURRENT planning too.
                # Otherwise only occluded tracks are filtered, while matched
                # single-frame jitter can invalidate an already executing path.
                box=np.asarray(entry['box'],dtype=float).copy();entry['raw_box']=box.tolist()
                box[:3]=rotation.T@(track['world']-translation)
                box[7:9]=(rotation.T@track['velocity'])[:2]
                entry['box']=box.tolist();entry['geometry_anchor']=track['anchor_method']
                entry.update(self.heading_evidence(track,required))
                entry['geometry_motion_evidence']=track.get('motion_evidence',{})
                supported=heading_support.refine(box,entry,max(tolerance,track['position_uncertainty']),track['radius'])
                if supported is not None:
                    directions=np.asarray(track['right_history'])[:,:2]
                    directions=directions/np.maximum(np.linalg.norm(directions,axis=1,keepdims=True),1e-12)
                    history_times=np.array([item[0] for item in track['history'][-len(directions):]])
                    angular_steps=np.arccos(np.clip(np.abs(np.sum(directions[1:]*directions[:-1],axis=1)),0.,1.))
                    observed_rate=float(np.max(angular_steps/np.diff(history_times))) if len(angular_steps) else 0.
                    # Reuse the existing metric error budget as an angular
                    # uncertainty floor; no class/scene-specific turn-rate cap.
                    angular_floor=2*np.arcsin(min(1.,tolerance/(2*track['radius'])))
                    track['validated_heading']=dict(time=now,right_world=track['right_world'].copy(),
                        span=supported[1]['heading_range_rad'],dimensions=supported[0][3:5].copy(),static_epoch=static_epoch,
                        rate=max(observed_rate,angular_floor/max_gap))
                continue
            box=track['box'].copy();box[:3]=local
            # Reconstruct heading from its world right axis; do not repeatedly
            # enclose an already-enlarged square or attach it to a weak query.
            right=rotation.T@track['right_world'];box[6]=np.arctan2(-right[1],right[0]);box[7:9]=(rotation.T@track['velocity'])[:2]
            stale_motion=track['confirmed'] and age>max_gap
            if stale_motion or static_epoch is not None:box[7:9]=0.
            row=dict(candidate_index=-1-track['id'],memory_id=track['id'],
                state='last_known_confirmed' if track['confirmed'] else 'provisional_uncertain',
                raw_score=track['score'],box=box.tolist(),label=track['label'],missing_seconds=age,
                geometry_motion_age_s=0. if stale_motion else age,geometry_radius_m=track['radius'],
                motion_model_expired=stale_motion,
                geometry_position_uncertainty_m=track['position_uncertainty'],
                observations=len(track['history']),duration_seconds=now-track['first'])
            row['geometry_motion_evidence']=track.get('motion_evidence',{})
            row['declared_static_epoch']=static_epoch
            # A missing detector query does not instantly erase a previously
            # independently validated orientation. Carry it only for max_gap,
            # monotonically widening from measured angular rates; never extend this lease
            # from a mere detection or a replayed memory row.
            verified=track.get('validated_heading')
            if track['confirmed'] and verified is not None:
                heading_age=now-verified['time']
                static_heading=static_epoch is not None and verified.get('static_epoch')==static_epoch
                if 0<=heading_age and (heading_age<max_gap or static_heading):
                    right=rotation.T@verified['right_world']
                    heading_yaw=float(np.arctan2(-right[1],right[0]))
                    span=float(verified['span']) if static_heading else min(np.pi/2,float(verified['span']+verified['rate']*heading_age))
                    row['geometry_heading_carry']=dict(age_s=heading_age,max_gap_s=max_gap,
                        heading_yaw=heading_yaw,heading_range_rad=span,angular_rate_evidence_radps=verified['rate'],static_epoch=static_epoch if static_heading else None,
                        dimensions_wl_m=np.maximum(verified['dimensions'],track['max_wl']).tolist(),
                        source='previous_current_occupancy_and_temporal_box_agreement',
                        uncertainty_bound_certified=False)
            result['obstacles'].append(row);memory_rows.append(row)
            # Keep the last observed obstacle; do not extrapolate stale velocity
            # indefinitely across the ego. Nearby unresolved history caps speed.
            stopping=float(info.get('ego_velo',0.))**2/(2*float(cfg.get('braking_mps2',2.)))
            if stale_motion and np.linalg.norm(local[:2])<stopping+track['radius']+float(cfg.get('ego_half_length_m',1.5))+tolerance:
                cap=float(cfg.get('caution_speed_mps',2.));old_cap=result.get('speed_limit_mps')
                result['speed_limit_mps']=min(cap,old_cap) if old_cap is not None else cap
        for index,obstacle in strong.items():
            if index in used:continue
            # Multiple detector queries near the same physical center are not
            # separate permanent memories. Current detections remain untouched.
            if any(t['label']==int(labels[index]) and np.linalg.norm(t['world']-world[index])<min(t['box'][3],boxes[index,3])/2 for t in retained):continue
            right=np.array([np.cos(boxes[index,6]),-np.sin(boxes[index,6]),0.])
            retained.append(dict(id=self.next_id,world=world[index].copy(),box=boxes[index].copy(),velocity=np.zeros(3) if static_epoch is not None else rotation@np.r_[boxes[index,7:9],0.],
                right_world=rotation@right,time=now,last_seen=now,first=now,confirmed=False,
                history=[(now,world[index].copy())],right_history=[rotation@right],max_wl=boxes[index,3:5].copy(),velocity_history=[rotation@np.r_[boxes[index,7:9],0.]],radius=float(np.hypot(*boxes[index,3:5])/2),position_uncertainty=0.,
                score=float(scores[index]),label=int(labels[index])))
            self.next_id+=1
        self.tracks=retained
        result['obstacle_memory']=dict(reset=reset,tracks=len(retained),confirmed_tracks=sum(t['confirmed'] for t in retained),constraints=memory_rows,
            rejected_geometry_updates=rejected_updates,rejected_position_velocity_fits=rejected_velocities,weak_queries_update_geometry=False,
            ground_skin_signature=self.ground_signature,ground_rejections=len(rejected),
            ground_truth_used=False,missing_confirmed_observation_means_free_space=False)
        return result


    @staticmethod
    def heading_evidence(track,required):
        directions=np.asarray(track.get('right_history',[]))
        if not track['confirmed'] or len(directions)<required:return {}
        directions=directions[:,:2]
        directions=directions/np.maximum(np.linalg.norm(directions,axis=1,keepdims=True),1e-12)
        current=np.asarray(track['right_world'])[:2]
        current=current/max(np.linalg.norm(current),1e-12)
        span=float(np.arccos(np.clip(np.abs(directions@current),0.,1.)).max())
        return dict(geometry_heading_span_rad=span,
                    geometry_dimensions_wl_m=np.asarray(track['max_wl']).tolist(),
                    geometry_heading_confirmed=True)


def rectangle_axes(yaw):
    """Local right/forward axes in model XY for native clockwise yaw."""
    yaw = np.asarray(yaw)
    return np.stack([np.cos(yaw), -np.sin(yaw)], -1), np.stack([np.sin(yaw), np.cos(yaw)], -1)


def rectangle_separation(states, boxes, times, half_width, half_length):
    """SAT separation lower bound for oriented rectangles [states, boxes].

    Positive means separated; negative means overlapping. Requiring a positive
    margin on a separating axis is conservative for diagonal corner clearance.
    Boxes follow the installed legacy LiDAR dimension/yaw convention.
    """
    states = np.asarray(states, dtype=float)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 9)
    if not len(boxes): return np.empty((len(states), 0))
    times = np.asarray(times, dtype=float)
    er, ef = rectangle_axes(states[:, 2]); br, bf = rectangle_axes(boxes[:, 6])
    axes = np.stack([np.broadcast_to(er[:, None], (len(states), len(boxes), 2)),
        np.broadcast_to(ef[:, None], (len(states), len(boxes), 2)),
        np.broadcast_to(br[None], (len(states), len(boxes), 2)),
        np.broadcast_to(bf[None], (len(states), len(boxes), 2))], 2)
    delta = boxes[None, :, :2]+times[:, None, None]*boxes[None, :, 7:9]-states[:, None, :2]
    projection = np.abs(np.sum(delta[:, :, None]*axes, -1))
    ego_radius = half_width*np.abs(np.sum(er[:, None, None]*axes, -1))+half_length*np.abs(np.sum(ef[:, None, None]*axes, -1))
    box_radius = boxes[None, :, None, 3]/2*np.abs(np.sum(br[None, :, None]*axes, -1))+boxes[None, :, None, 4]/2*np.abs(np.sum(bf[None, :, None]*axes, -1))
    return np.max(projection-ego_radius-box_radius, -1)


def rectangle_motion_envelope(states,boxes,times,half_width,half_length,ages=None,heading_ranges=None):
    """Oriented bounds enclose stationary through reported-velocity motion."""
    states=np.asarray(states);boxes=np.asarray(boxes).reshape(-1,9);times=np.asarray(times)
    er,ef=rectangle_axes(states[:,2]);br,bf=rectangle_axes(boxes[:,6])
    travel=(times[:,None]+(np.zeros(len(boxes)) if ages is None else np.asarray(ages))[None])[:,:,None]*boxes[None,:,7:9]
    n,m=len(states),len(boxes)
    theta=np.zeros(m) if heading_ranges is None else np.asarray(heading_ranges)
    plus_r,plus_f=rectangle_axes(boxes[:,6]+theta);minus_r,minus_f=rectangle_axes(boxes[:,6]-theta)
    delta=boxes[None,:,:2]+travel/2-states[:,None,:2]
    # The nearest ego point supplies the disk's exact separating direction.
    local=np.stack([np.sum(delta*er[:,None],-1),np.sum(delta*ef[:,None],-1)],-1)
    residual=local-np.clip(local,[-half_width,-half_length],[half_width,half_length])
    radial=residual[:,:,0,None]*er[:,None]+residual[:,:,1,None]*ef[:,None]
    radial_norm=np.linalg.norm(radial,axis=-1,keepdims=True)
    radial=np.where(radial_norm>1e-12,radial/np.maximum(radial_norm,1e-12),er[:,None])
    axes=np.stack([np.broadcast_to(er[:,None],(n,m,2)),np.broadcast_to(ef[:,None],(n,m,2)),
        *[np.broadcast_to(axis[None],(n,m,2)) for axis in (br,bf,plus_r,plus_f,minus_r,minus_f)],radial],2)
    projection=np.abs(np.sum(delta[:,:,None]*axes,-1))
    ego=half_width*np.abs(np.sum(er[:,None,None]*axes,-1))+half_length*np.abs(np.sum(ef[:,None,None]*axes,-1))
    corners=np.stack([a*boxes[:,3,None]/2*br+b*boxes[:,4,None]/2*bf for a,b in ((-1,-1),(-1,1),(1,-1),(1,1))],1)
    rotated=np.stack([corners[:,:,1],-corners[:,:,0]],-1)
    aa=np.einsum('nmkd,mcd->nmkc',axes,corners);bb=np.einsum('nmkd,mcd->nmkc',axes,rotated)
    peak=np.arctan2(bb,aa)
    endpoints=aa*np.cos(theta)[None,:,None,None]+np.abs(bb)*np.sin(theta)[None,:,None,None]
    support=np.where(np.abs(peak)<=theta[None,:,None,None],np.hypot(aa,bb),endpoints).max(-1)
    motion=np.abs(np.sum(travel[:,:,None]*axes,-1))/2
    return (projection-ego-support-motion).max(-1)



class OccupancyHeadingSupport:
    """Require temporal box evidence AND agreement with a whole current component.

    Metric uncertainty stays unchanged. Observed yaw variation expands the
    shape; the metric margin covers unresolved corner displacement once. Disagreement,
    missing evidence, or merged/partial occupancy retains the all-heading disk.
    This is evidence-based geometry, not a certified absolute yaw error bound.
    """
    def __init__(self,mask):
        self.components=[]
        if mask is None:return
        from scipy.ndimage import label,find_objects
        labels,count=label(np.asarray(mask)!=0)
        for index,region in enumerate(find_objects(labels),1):
            if region is None:continue
            rc=np.argwhere(labels[region]==index)+[region[0].start,region[1].start]
            self.components.append((rc[:,[1,0]]-100.)*.5+.25)

    def refine(self,box,obstacle,margin,radius):
        if (not obstacle.get('geometry_heading_confirmed') or not self.components
                or obstacle.get('state')=='last_known_confirmed'):return None
        span=float(obstacle.get('geometry_heading_span_rad',float('nan')))
        dimensions=np.asarray(obstacle.get('geometry_dimensions_wl_m',box[3:5]),dtype=float)
        if (not np.isfinite(span) or span<0 or span>=np.pi/2
                or dimensions.shape!=(2,) or not np.isfinite(dimensions).all()
                or np.any(dimensions<=0)):return None
        width,length=np.maximum(box[3:5],obstacle.get('geometry_dimensions_wl_m',box[3:5]))
        theta=min(np.pi/2,float(obstacle['geometry_heading_span_rad']))
        agreement_angle=min(np.pi/2,theta+2*np.arcsin(min(1.,margin/(2*radius))))
        half_w,half_l=width/2,length/2
        a=min(theta,np.arctan2(half_l,half_w));b=min(theta,np.arctan2(half_w,half_l))
        extents=np.array([half_w*np.cos(a)+half_l*np.sin(a),half_l*np.cos(b)+half_w*np.sin(b)])
        points=min(self.components,key=lambda p:float(np.linalg.norm(p-box[:2],axis=1).min()))
        if np.linalg.norm(points-box[:2],axis=1).min()>radius:return None
        right,forward=rectangle_axes(box[6]);local=(points-box[:2])@np.stack([right,forward],axis=1)
        # A tiny blob or a merged neighboring object cannot validate a car's yaw.
        if np.any(np.ptp(local,axis=0)<np.array([width,length])/2):return None
        if abs(length-width)>2*margin:
            _,vectors=np.linalg.eigh(np.cov(points.T))
            major=vectors[:,-1];expected=forward if length>width else right
            if np.arccos(np.clip(abs(float(major@expected)),0.,1.))>agreement_angle:return None
        if np.any(np.abs(local)>extents+margin+GRID_HALF_DIAGONAL_M):return None
        refined=box.copy();refined[3:5]=[width,length]
        return refined,dict(heading_range_rad=theta,component_cells=len(points),bounded_wl_m=(2*extents).tolist())



def carried_heading_shape(box,obstacle):
    """Finite evidence lease, not a new current observation or certified bound."""
    carry=obstacle.get('geometry_heading_carry')
    if not carry or obstacle.get('state')!='last_known_confirmed':return None
    if carry.get('source')!='previous_current_occupancy_and_temporal_box_agreement':return None
    age=float(carry.get('age_s',np.inf));gap=float(carry.get('max_gap_s',0.))
    theta=float(carry.get('heading_range_rad',np.pi/2));yaw=float(carry.get('heading_yaw',np.nan))
    dims=np.asarray(carry.get('dimensions_wl_m',[]),dtype=float)
    static_heading=carry.get('static_epoch') is not None and carry.get('static_epoch')==obstacle.get('declared_static_epoch')
    if (not np.isfinite([age,gap,theta,yaw]).all() or not (age>=0 and (age<gap or static_heading)) or not 0<=theta<np.pi/2
            or dims.shape!=(2,) or not np.isfinite(dims).all() or np.any(dims<=0)):return None
    refined=box.copy();refined[3:5]=np.maximum(box[3:5],dims);refined[6]=yaw
    w,l=refined[3:5]/2;a=min(theta,np.arctan2(l,w));b=min(theta,np.arctan2(w,l))
    extent=np.array([w*np.cos(a)+l*np.sin(a),l*np.cos(b)+w*np.sin(b)])
    return refined,dict(carried=True,static_snapshot=static_heading,heading_range_rad=theta,age_s=age,max_gap_s=gap,
        bounded_wl_m=(2*extent).tolist(),source=carry['source'],uncertainty_bound_certified=False)


def occupancy_clearance(states, points, vehicle, point_radius=GRID_HALF_DIAGONAL_M):
    """Exact ego rectangle vs grid-center circles enclosing occupied cells.

    Vehicle clearance and cell discretization are each counted once. There is
    no extra ego bounding circle or pre-dilation of the same occupied cells.
    """
    states = np.asarray(states, dtype=float)
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(points): return np.full(len(states), np.inf)
    d = points[None]-states[:, None, :2]
    c, s = np.cos(states[:, 2, None]), np.sin(states[:, 2, None])
    local = np.stack([d[:, :, 0]*c-d[:, :, 1]*s, d[:, :, 0]*s+d[:, :, 1]*c], -1)
    outside = np.abs(local)-[vehicle.half_width_m, vehicle.half_length_m]
    sdf = np.linalg.norm(np.maximum(outside, 0), axis=-1)+np.minimum(np.max(outside, axis=-1), 0)
    return sdf.min(axis=1)-vehicle.clearance_margin_m-point_radius


def envelope_separation(states, boxes, times, radii, half_width, half_length, ages=None):
    """Ego rectangle vs heading-free obstacle and uncertain motion envelope.

    Each obstacle disk encloses every heading of its observed rectangle.
    Its center segment includes both stationary and reported-velocity motion.
    Exact segment-to-rectangle distance avoids spreading motion uncertainty
    perpendicular to the obstacle's travel direction.
    These are conservative hypotheses, not calibrated error bounds.
    """
    states = np.asarray(states, dtype=float)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 9)
    if not len(boxes): return np.empty((len(states), 0))
    times = np.asarray(times, dtype=float)
    effective_times=times[:,None]+(np.zeros(len(boxes)) if ages is None else np.asarray(ages))[None]
    travel = effective_times[:,:,None]*boxes[None,:,7:9]
    delta = boxes[None,:,:2]-states[:,None,:2]
    right, forward = rectangle_axes(states[:,2])
    start = np.stack([np.sum(delta*right[:,None],-1), np.sum(delta*forward[:,None],-1)],-1)
    direction = np.stack([np.sum(travel*right[:,None],-1), np.sum(travel*forward[:,None],-1)],-1)
    extent = np.array([half_width,half_length])
    def sdf(points):
        outside = np.abs(points)-extent
        return np.linalg.norm(np.maximum(outside,0),axis=-1)+np.minimum(np.max(outside,axis=-1),0)
    # Closest pairs are an endpoint to the rectangle, a corner to the segment,
    # or a segment/rectangle intersection. This enumerates all three cases.
    distance = np.minimum(sdf(start),sdf(start+direction))
    corners = np.array([[-half_width,-half_length],[-half_width,half_length],
                        [half_width,-half_length],[half_width,half_length]])
    delta_corner = corners[None,None]-start[:,:,None]
    denominator = np.sum(direction**2,axis=-1)
    fraction = np.clip(np.sum(delta_corner*direction[:,:,None],axis=-1)/np.maximum(denominator[:,:,None],1e-20),0,1)
    corner_distance = np.linalg.norm(delta_corner-fraction[:,:,:,None]*direction[:,:,None],axis=-1).min(axis=-1)
    distance = np.minimum(distance,corner_distance)
    parallel = np.abs(direction)<1e-12
    safe_direction = np.where(parallel,1.,direction)
    a = (-extent-start)/safe_direction; b = (extent-start)/safe_direction
    enter = np.where(parallel,-np.inf,np.minimum(a,b)).max(axis=-1)
    leave = np.where(parallel,np.inf,np.maximum(a,b)).min(axis=-1)
    intersects = (np.maximum(enter,0)<=np.minimum(leave,1)) & ~np.any(parallel & (np.abs(start)>extent),axis=-1)
    distance = np.where(intersects,np.minimum(distance,0.),distance)
    return distance-np.asarray(radii)[None]


def sample_half_intervals(times):
    """Nearest-sample time cells, without extrapolating beyond the horizon."""
    times=np.asarray(times,dtype=float)
    half=np.zeros(len(times))
    if len(times)>1:
        if np.any(np.diff(times)<=0): raise ValueError('Non-increasing guard sample times')
        half[:-1]=np.maximum(half[:-1],np.diff(times)/2)
        half[1:]=np.maximum(half[1:],np.diff(times)/2)
    return half


def sweep_allowance(states, vehicle):
    """Cover translations and rotations between successive vehicle states."""
    allowance = np.zeros(len(states))
    if len(states) > 1:
        intervals = (np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1)
            +np.hypot(vehicle.half_width_m, vehicle.half_length_m)*np.abs(np.diff(states[:, 2])))/2
        allowance[:-1] = np.maximum(allowance[:-1], intervals)
        allowance[1:] = np.maximum(allowance[1:], intervals)
    return allowance


class TemporalOccupancy:
    """UniAD's current + four future masks in the current LiDAR frame.

    nuScenes keyframes use 0.5s spacing. Bracket future times conservatively;
    t=0 checks only the current mask. Beyond 2s hold the last prediction,
    explicitly an extrapolation rather than a new network prediction.
    """
    def __init__(self,masks,vehicle,observed_points=None,observed_point_radius=GRID_HALF_DIAGONAL_M,unknown_points=None,unknown_point_radius=GRID_HALF_DIAGONAL_M):
        masks=np.asarray(masks)
        if masks.shape!=(1,5,200,200) or not np.isfinite(masks).all():
            raise ValueError('Unexpected temporal occupancy shape')
        self.observed_points=np.asarray(observed_points if observed_points is not None else [],dtype=float).reshape(-1,2)
        if not np.isfinite(self.observed_points).all():raise ValueError('Invalid observed geometry')
        self.observed_point_radius=float(observed_point_radius)
        if not np.isfinite(self.observed_point_radius) or self.observed_point_radius<0:raise ValueError('Invalid observed cell bound')
        from scipy.spatial import cKDTree
        self.observed_tree=cKDTree(self.observed_points) if len(self.observed_points) else None
        self.unknown_points=np.asarray(unknown_points if unknown_points is not None else [],dtype=float).reshape(-1,2)
        self.unknown_point_radius=float(unknown_point_radius)
        if not np.isfinite(self.unknown_points).all() or not np.isfinite(self.unknown_point_radius) or self.unknown_point_radius<0:raise ValueError('Invalid unknown visibility geometry')
        self.unknown_tree=cKDTree(self.unknown_points) if len(self.unknown_points) else None
        self.vehicle=vehicle;self.points=[]
        for mask in masks[0]:
            rc=np.argwhere(mask!=0)
            self.points.append((rc[:,[1,0]].astype(float)-100.)*.5+.25)
        self.counts=[len(p) for p in self.points]
        if not any(self.counts):raise ValueError('Empty occupancy: stop diagnostic instead of assuming clear space')

    def clearance(self,states,times):
        states=np.asarray(states);times=np.asarray(times)
        if len(states)!=len(times) or not np.isfinite(times).all() or np.any(times<0):
            raise ValueError('Invalid occupancy query times')
        index=np.clip(times/.5,0.,4.);lo=np.floor(index+1e-8).astype(int);hi=np.ceil(index-1e-8).astype(int)
        # Empty bins mean no predicted occupied cells, not a fake obstacle.
        # Still constrain the physical footprint to the 100m prediction grid.
        c=np.abs(np.cos(states[:,2]));s=np.abs(np.sin(states[:,2]))
        half_x=self.vehicle.half_width_m*c+self.vehicle.half_length_m*s
        half_y=self.vehicle.half_width_m*s+self.vehicle.half_length_m*c
        result=np.minimum(50.-np.abs(states[:,0])-half_x,50.-np.abs(states[:,1])-half_y)-self.vehicle.clearance_margin_m
        for a,b in set(zip(lo.tolist(),hi.tolist())):
            indices=np.flatnonzero((lo==a)&(hi==b))
            points=self.points[a] if a==b else np.concatenate([self.points[a],self.points[b]])
            if len(points):result[indices]=np.minimum(result[indices],occupancy_clearance(states[indices],points,self.vehicle))
        if len(self.observed_points):
            # Exact pruning: rectangle SDF is between center distance +/- its
            # circumradius. Points beyond nearest_center_distance + 2R cannot win.
            distance,_=self.observed_tree.query(states[:,:2])
            neighborhoods=self.observed_tree.query_ball_point(states[:,:2],distance+2*np.hypot(self.vehicle.half_width_m,self.vehicle.half_length_m)+1e-8)
            ids=np.unique(np.concatenate(neighborhoods)).astype(int)
            result=np.minimum(result,occupancy_clearance(states,self.observed_points[ids],self.vehicle,self.observed_point_radius))
        if len(self.unknown_points):
            distance,_=self.unknown_tree.query(states[:,:2])
            neighborhoods=self.unknown_tree.query_ball_point(states[:,:2],distance+2*np.hypot(self.vehicle.half_width_m,self.vehicle.half_length_m)+1e-8)
            ids=np.unique(np.concatenate(neighborhoods)).astype(int)
            result=np.minimum(result,occupancy_clearance(states,self.unknown_points[ids],self.vehicle,self.unknown_point_radius))
        return result

    def audit(self):
        return dict(unknown_visibility_cells=len(self.unknown_points),unknown_visibility_policy='hard unknown continuation; kept separate from measured depth',observed_depth_cells=len(self.observed_points),times_s=[0.,.5,1.,1.5,2.],occupied_cells=self.counts,
            current_state_policy='current mask only',future_policy='union bracketing time bins',
            tail_policy='hold final prediction beyond 2s; extrapolation uncertified')


class PlanningGuard:
    def __init__(self, report, vehicle, braking_states, braking_times, current_occupancy=None):
        self.vehicle = vehicle
        self.report = report or {}
        cfg = self.report.get('config', {})
        self.confirm_seconds = float(cfg.get('confirmation_seconds', .2))
        self.confirm_observations = int(cfg.get('confirmation_observations', 3))
        self.max_gap_seconds = float(cfg.get('max_gap_seconds', .75))
        self.caution_speed = float(cfg.get('caution_speed_mps', 2.))
        uncertainty_margin = float(cfg.get('obstacle_margin_m', .5))
        if not np.isfinite([self.confirm_seconds, self.max_gap_seconds, self.caution_speed, uncertainty_margin]).all() or min(self.confirm_seconds, self.max_gap_seconds, self.caution_speed) <= 0 or uncertainty_margin < 0 or self.confirm_observations < 1:
            raise ValueError('Invalid shared guard parameters')
        self.rows = []; self.hard_boxes = []; self.hard_margins = []; self.soft_boxes = []; self.soft_margins = []
        self.hard_envelopes = []; self.hard_radii = []; self.hard_ages = [];self.hard_oriented=[];self.hard_heading_ranges=[]
        heading_support=OccupancyHeadingSupport(current_occupancy)
        for obstacle in self.report.get('obstacles', []):
            box = np.asarray(obstacle['box'], dtype=float)
            if box.ndim != 1 or len(box) not in (7, 9) or not np.isfinite(box).all() or np.any(box[3:6] <= 0):
                raise ValueError('Invalid proposal geometry in guard report')
            if len(box) == 7: box = np.r_[box, 0., 0.]
            score = float(obstacle['raw_score'])
            if not np.isfinite(score) or not 0 <= score <= 1: raise ValueError('Invalid proposal score')
            state = obstacle['state']
            if state not in ('network_retained', 'persistent_uncertain', 'provisional_uncertain', 'last_known_confirmed'):
                raise ValueError('Unknown proposal evidence state')
            retained = state == 'network_retained' and score >= float(self.report.get('detection_threshold', .4))
            confirmed = (state == 'persistent_uncertain'
                and int(obstacle.get('observations', 0)) >= self.confirm_observations
                and float(obstacle.get('duration_seconds', 0)) >= self.confirm_seconds-1e-6)
            remembered = state == 'last_known_confirmed'
            robust = retained or confirmed or remembered
            # A high classification score does not certify metric accuracy.
            position_error = float(obstacle.get('geometry_position_uncertainty_m', 0.))
            age=float(obstacle.get('geometry_motion_age_s',0.))
            if not np.isfinite(age) or age<0:raise ValueError('Invalid memory motion age')
            radius = float(obstacle.get('geometry_radius_m', np.hypot(*box[3:5])/2))
            if not np.isfinite([position_error,radius]).all() or position_error < 0 or radius <= 0:
                raise ValueError('Invalid geometry envelope evidence')
            margin = max(vehicle.clearance_margin_m, uncertainty_margin, position_error)
            heading=heading_support.refine(box,obstacle,margin,radius) if robust else None
            if heading is None and remembered:
                heading=carried_heading_shape(box,obstacle)
            if heading is not None:box=heading[0]
            if robust and heading is not None:
                envelope_times=np.minimum(np.asarray(braking_times)+sample_half_intervals(braking_times),braking_times[-1])
                gaps=rectangle_motion_envelope(braking_states,box[None],envelope_times,vehicle.half_width_m,vehicle.half_length_m,ages=[age],heading_ranges=[heading[1]['heading_range_rad']])[:,0]-margin
            elif robust:
                envelope_times=np.minimum(np.asarray(braking_times)+sample_half_intervals(braking_times),braking_times[-1])
                gaps = envelope_separation(braking_states, box[None], envelope_times+age, [radius],
                    vehicle.half_width_m, vehicle.half_length_m)[:,0]-margin
            else:
                gaps = rectangle_separation(braking_states, box[None], braking_times,
                    vehicle.half_width_m, vehicle.half_length_m)[:,0]-margin
            # Include relative obstacle travel as well as ego turning between samples.
            swept = sweep_allowance(braking_states, vehicle)
            if not robust: swept += np.linalg.norm(box[7:9])*sample_half_intervals(braking_times)
            urgent = bool(np.min(gaps-swept) <= 0)
            hard = retained or confirmed or urgent or remembered
            reason = 'braking_envelope_conflict' if remembered and urgent else ('last_known_confirmed' if remembered else ('network_retained' if retained else ('temporally_confirmed' if confirmed else ('braking_envelope_conflict' if urgent else 'awaiting_confirmation'))))
            row = dict(candidate_index=obstacle['candidate_index'], evidence_state=state,
                raw_score=score, hard_constraint=hard, reason=reason,
                geometry=((('static_snapshot_validated_heading_motion_envelope' if heading[1].get('static_snapshot') else 'finite_age_validated_heading_motion_envelope') if heading[1].get('carried') else 'occupancy_supported_heading_motion_envelope') if heading is not None else 'heading_free_disk_motion_envelope') if robust else 'reported_rectangle',
                heading_support=heading[1] if heading is not None else None,
                geometry_radius_m=radius if robust else None,
                effective_margin_m=margin, braking_envelope_clearance_m=float(np.min(gaps-swept)))
            self.rows.append(row)
            if hard:
                self.hard_boxes.append(box); self.hard_margins.append(margin)
                self.hard_envelopes.append(robust and heading is None);self.hard_oriented.append(robust and heading is not None); self.hard_radii.append(radius); self.hard_ages.append(age);self.hard_heading_ranges.append(heading[1]['heading_range_rad'] if heading is not None else 0.)
            else:
                self.soft_boxes.append(box); self.soft_margins.append(margin)
        self.hard_boxes = np.asarray(self.hard_boxes).reshape(-1, 9)
        self.soft_boxes = np.asarray(self.soft_boxes).reshape(-1, 9)
        self.hard_margins = np.asarray(self.hard_margins)
        self.hard_envelopes = np.asarray(self.hard_envelopes, dtype=bool)
        self.hard_radii = np.asarray(self.hard_radii)
        self.hard_ages=np.asarray(self.hard_ages);self.hard_oriented=np.asarray(self.hard_oriented,dtype=bool);self.hard_heading_ranges=np.asarray(self.hard_heading_ranges)
        self.soft_margins = np.asarray(self.soft_margins)
        provisional_path_conflicts = any(not row['hard_constraint'] and obstacle.get('path_clearance_m', float('inf')) <= 0
            for row, obstacle in zip(self.rows, self.report.get('obstacles', [])))
        reported_cap = self.report.get('speed_limit_mps')
        if reported_cap is not None and (not np.isfinite(reported_cap) or reported_cap < 0): raise ValueError('Invalid caution speed cap')
        self.speed_limit = min(self.caution_speed, float(reported_cap)) if reported_cap is not None else (self.caution_speed if provisional_path_conflicts else None)

    def clearance(self, states, times):
        gaps = rectangle_separation(states, self.hard_boxes, times,
            self.vehicle.half_width_m, self.vehicle.half_length_m)
        if not len(self.hard_boxes): return np.full(len(states), np.inf)
        mask = self.hard_envelopes
        if np.any(mask):
            envelope_times=np.minimum(np.asarray(times)+sample_half_intervals(times),times[-1])
            gaps[:,mask] = envelope_separation(states,self.hard_boxes[mask],envelope_times,self.hard_radii[mask],
                self.vehicle.half_width_m,self.vehicle.half_length_m,ages=self.hard_ages[mask])
        # Relative motion can cover an interval even when endpoints are clear.
        speed = np.linalg.norm(self.hard_boxes[:, 7:9], axis=1)
        # The nested motion capsule already covers the full temporal cell.
        # Only point-at-time provisional rectangles need isotropic travel error.
        oriented=self.hard_oriented
        if np.any(oriented):
            envelope_times=np.minimum(np.asarray(times)+sample_half_intervals(times),times[-1])
            gaps[:,oriented]=rectangle_motion_envelope(states,self.hard_boxes[oriented],envelope_times,
                self.vehicle.half_width_m,self.vehicle.half_length_m,ages=self.hard_ages[oriented],heading_ranges=self.hard_heading_ranges[oriented])
        speed=np.where(self.hard_envelopes|self.hard_oriented,0.,speed)
        temporal = sample_half_intervals(times)
        return np.min(gaps-self.hard_margins[None]-temporal[:, None]*speed[None], axis=1)

    def soft_cost(self, states, times):
        if not len(self.soft_boxes): return 0.
        gaps = rectangle_separation(states, self.soft_boxes, times,
            self.vehicle.half_width_m, self.vehicle.half_length_m)-self.soft_margins[None]
        return float(np.mean(np.maximum(-gaps, 0)**2))

    def audit(self):
        return dict(policy=POLICY, evidence_tiers=self.rows,
            hard_proposals=len(self.hard_boxes), soft_proposals=len(self.soft_boxes),
            speed_limit_mps=self.speed_limit,
            geometry='oriented ego; current-component heading agreement or finite-age widening of previously validated heading; otherwise heading-free disks; provisional rectangles; occupancy cells',
            margin_policy='max(base clearance, uncertainty floor, observed position residual); occupancy grid error counted once',
            proposal_time_policy='match vehicle and obstacle time; no proposal temporal union',
            raw_occupancy_time_policy='current mask at t=0; bracketed future masks at matching time',
            geometry_error_bound_certified=False,
            confirmed_motion_policy='enclose stationary through reported constant velocity; no oracle motion',
            temporal_motion_budget='nested capsules evaluated at cell end; travel allowance only for provisional rectangles',
            recovery=dict(confirmation_seconds=self.confirm_seconds, confirmation_observations=self.confirm_observations,
                max_gap_seconds=self.max_gap_seconds, speed_limit_mps=self.caution_speed),
            ground_truth_used=False)


@dataclass
class RecoveryController:
    """Re-evaluate every frame, with continuous feasible evidence before release."""
    state: str = 'normal'
    clear_since: float = None
    clear_count: int = 0
    previous_time: float = None

    def update(self, optimized, speed, timestamp, dt, braking):
        timestamp = float(timestamp)
        audit = optimized.get('guard', {})
        cfg = audit.get('recovery', {})
        seconds = float(cfg.get('confirmation_seconds', .2))
        observations = int(cfg.get('confirmation_observations', 3))
        gap = float(cfg.get('max_gap_seconds', .75))
        if not np.isfinite([timestamp, speed, dt, braking, seconds, gap]).all() or speed<0 or min(dt, braking, seconds, gap)<=0 or observations<1:
            raise ValueError('Invalid recovery motion context')
        if self.previous_time is not None and (timestamp<=self.previous_time or timestamp-self.previous_time>gap):
            self.state = 'stopped_reassessing'; self.clear_since=None; self.clear_count=0
        self.previous_time = timestamp
        before = self.state
        selected = optimized.get('selected')
        # A failed or urgent frame breaks recovery evidence immediately.
        if selected is None:
            self.state = 'braking' if speed>1e-8 else 'stopped_reassessing'
            self.clear_since=None; self.clear_count=0
            reason = optimized.get('reason', 'no_feasible_plan')
            action = dict(acc=-min(braking, speed/dt), steer_rate=0.)
        else:
            controls = np.asarray(selected['controls'], dtype=float)
            if controls.shape!=(6,2) or not np.isfinite(controls).all(): raise ValueError('Invalid feasible controls')
            urgent = any(row['reason']=='braking_envelope_conflict' for row in audit.get('evidence_tiers', []))
            # Urgency describes the fixed-steering emergency-stop path.
            # The optimizer checks every hard obstacle on the selected path;
            # a verified alternative must not be replaced by that conflicting stop.
            residual = selected.get('minimum_constraint', float('nan'))
            verified = (selected.get('feasible') is True
                        and np.isfinite(residual) and residual >= -1e-5)
            if urgent and not verified:
                self.state='braking' if speed>1e-8 else 'stopped_reassessing'
                self.clear_since=None; self.clear_count=0
                reason='urgent_proposal_braking_envelope'
                action=dict(acc=-min(braking, speed/dt),steer_rate=0.)
            elif self.state in ('braking','stopped_reassessing'):
                if self.clear_since is None:self.clear_since=timestamp
                self.clear_count+=1
                if self.clear_count>=observations and timestamp-self.clear_since>=seconds-1e-6:
                    self.state='recovering';reason='sustained_feasible_recovery'
                    action=dict(acc=float(controls[0,0]),steer_rate=float(controls[0,1]))
                else:
                    self.state='braking' if speed>1e-8 else 'stopped_reassessing';reason='awaiting_sustained_feasibility'
                    action=dict(acc=-min(braking,speed/dt),steer_rate=0.)
            else:
                if self.state!='recovering' or (self.clear_since is not None and timestamp-self.clear_since>=2*seconds):
                    self.state='caution' if audit.get('speed_limit_mps') is not None else 'normal'
                reason='feasible_plan'
                action=dict(acc=float(controls[0,0]),steer_rate=float(controls[0,1]))
        return action,dict(policy=POLICY,state_before=before,state_after=self.state,reason=reason,
            clear_observations=self.clear_count,clear_since=self.clear_since,
            optimization_recomputed=True)

    def speed_cap(self, report):
        """Recovery starts under the same scene-independent caution limit."""
        if self.state in ('normal','caution'):return None
        return float((report or {}).get('config',{}).get('caution_speed_mps',2.))
