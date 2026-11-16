"""Isolated memory association repair; evidence/geometry admission unchanged."""

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

import numpy as np
from planning_guard import OccupancyHeadingSupport
class ObstacleMemory:
    """Pose-compensated confirmed history; weak proposals cannot move anchors.

    New tracks remain probationary until repeated strong evidence. Confirmed
    tracks survive occlusion; uncertain single-frame tracks expire at max_gap.
    Motion is estimated from strong world observations, never low-score drift.
    """
    def __init__(self,ground_points_world=None,ground_signature=None):
        self.tracks=[];self.time=None;self.scene=None;self.next_id=0;self.static_epoch=None
        import json
        pass  # Path is provided by the portable boundary above
        root=next(p for p in Path(__file__).resolve().parents if (p/'code/HUGSIM').is_dir())
        planning=_hugsim_expand(json.loads((root/'waymo23-native-import/configs/planning.json').read_text()))
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

    def update(self,report,boxes,scores,labels,info,current_occupancy=None,observation=None):
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
        reset=self.time is not None and (scene!=self.scene or static_epoch!=self.static_epoch or now<=self.time or now-self.time>max_gap)
        if reset:self.tracks=[]
        self.time=now;self.scene=scene;self.static_epoch=static_epoch
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
            # Prediction aids association only; it does not confirm motion/heading.
            predicted=track['world']+track['velocity']*(now-track['time'])
            for i in strong:
                ratio=boxes[i,3:6]/track['box'][3:6]
                distance=float(np.linalg.norm(world[i]-predicted))
                if labels[i]==track['label'] and (.5<=ratio).all() and (ratio<=2.).all() and distance<=association:
                    pairs.append((distance,j,i))
        matches={};used=set()
        for _,j,i in sorted(pairs):
            if j not in matches and i not in used:matches[j]=i;used.add(i)
        retained=[];memory_rows=[];rejected_updates=0;rejected_velocities=0;retired=[];visibility_checks=[]
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
            if index is not None and accepted:
                track['free_count']=0
                retained.append(track)
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
            expired_geometry=bool(stale_motion and static_epoch is None)
            evidence=dict(free=False,reason='current_observation_or_static_epoch')
            if index is None and track['confirmed'] and static_epoch is None:
                from memory_visibility import box_free
                evidence=box_free(box,observation,info)
                count=track.get('free_count',0)
                if evidence['free']:
                    if not count:track['free_first']=now
                    track['free_count']=count+1
                else:track['free_count']=0
                visibility_checks.append(dict(memory_id=track['id'],age_s=age,
                    repeated_free_observations=track['free_count'],evidence=evidence))
                if track['free_count']>=required and now-track['free_first']>=duration-1e-6:
                    retired.append(dict(memory_id=track['id'],age_s=age,
                        observations=track['free_count'],span_s=now-track['free_first'],
                        reason='repeated_reliable_depth_beyond_entire_old_box',evidence=evidence))
                    continue
            else:track['free_count']=0
            retained.append(track)
            if stale_motion or static_epoch is not None:box[7:9]=0.
            row=dict(candidate_index=-1-track['id'],memory_id=track['id'],
                state='last_known_confirmed' if track['confirmed'] else 'provisional_uncertain',
                raw_score=track['score'],box=box.tolist(),label=track['label'],missing_seconds=age,
                geometry_motion_age_s=0. if stale_motion else age,geometry_radius_m=track['radius'],
                motion_model_expired=stale_motion,geometry_evidence_expired=expired_geometry,
                evidence_age_s=age,geometry_observation_timestamp_s=track['time'],
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
            if stale_motion and not expired_geometry and np.linalg.norm(local[:2])<stopping+track['radius']+float(cfg.get('ego_half_length_m',1.5))+tolerance:
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
            ground_truth_used=False,missing_confirmed_observation_means_free_space=False,
            lifecycle=dict(retired=retired,visibility_checks=visibility_checks,expired_risk_rows=sum(o.get('geometry_evidence_expired',False) for o in memory_rows),hard_lease_s=max_gap,free_space_requires_repeated_whole_box_depth=True))
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

from global_repair_v1.pipeline import validate_memory_input
class ContractObstacleMemory(ObstacleMemory):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from motion_observer import MotionObserver
        self.motion_observer=MotionObserver()
    def update(self,report,boxes,scores,labels,info,current_occupancy=None,observation=None):
        receipt=validate_memory_input(report,boxes,scores,labels,info)
        result=super().update(report,boxes,scores,labels,info,current_occupancy=current_occupancy,observation=observation)
        certificates,audit=self.motion_observer.observe(report,info)
        for obstacle in result.get('obstacles',[]):
            certificate=certificates.get(obstacle['candidate_index'])
            if certificate is not None:obstacle['geometry_time_local_motion']=certificate
        result['motion_observation_audit']=audit
        result["global_evidence_receipt"]=dict(receipt,stage="after_driver_memory")
        return result
