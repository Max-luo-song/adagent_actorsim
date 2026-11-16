"""Separate observed footprint, motion support and unobserved uncertainty."""
import numpy as np

EVIDENCE_POLICY=dict(motion_window_s=1.6,minimum_motion_span_s=.39,
    position_noise_floor_m=.6,range_noise_fraction=.02,minimum_direction_consistency=.75,
    displacement_noise_ratio=2.,maximum_residual_noise_ratio=2.,
    soft_age_margin_limit_m=1.5,soft_age_growth_mps=.5,soft_confidence_decay_s=2.,
    ground_top_support_m=.2,maximum_motion_mps=30.)

def update_motion(track):
    samples=track['samples'];now=track['last'];old_velocity=np.asarray(track.get('velocity',np.zeros(2))).copy()
    distance=float(np.linalg.norm(track['box'][:2]));noise=EVIDENCE_POLICY['position_noise_floor_m']+EVIDENCE_POLICY['range_noise_fraction']*distance
    epoch=float(track.get('motion_epoch',samples[0][0]));reset=False
    if len(samples)>=2:
        t0,p0=samples[-2];dt=now-t0
        innovation=float(np.linalg.norm(samples[-1][1]-(p0+old_velocity*dt)))
        if (track.get('motion_reliable',False) or dt>=.25) and innovation>max(2*noise,.5*track['box'][4],2.):epoch=now;reset=True
    else:innovation=0.
    track['motion_epoch']=epoch
    recent=[(t,p) for t,p in samples if t>=epoch and now-t<=EVIDENCE_POLICY['motion_window_s']]
    ts=np.array([t for t,_ in recent]);pts=np.array([p for _,p in recent]);span=ts[-1]-ts[0]
    velocity=np.zeros(2);supported=False;consistency=0.;residual90=0.;signal=0.;fitted=np.zeros(2)
    if len(recent)>=3 and span>=EVIDENCE_POLICY['minimum_motion_span_s']:
        estimates=[(pts[j]-pts[i])/(ts[j]-ts[i]) for i in range(len(ts)) for j in range(i+1,len(ts)) if ts[j]-ts[i]>=.19]
        fitted=np.median(estimates,axis=0);fitted*=min(1.,EVIDENCE_POLICY['maximum_motion_mps']/max(np.linalg.norm(fitted),1e-9))
        intercept=np.median(pts-(ts-now)[:,None]*fitted,axis=0)
        residual=np.linalg.norm(pts-(intercept+(ts-now)[:,None]*fitted),axis=1);residual90=float(np.quantile(residual,.9))
        direction=fitted/max(np.linalg.norm(fitted),1e-9)
        consistency=float(np.mean(np.diff(pts,axis=0)@direction>.05));signal=float(np.linalg.norm(fitted)*span)
        supported=bool(signal>EVIDENCE_POLICY['displacement_noise_ratio']*noise and consistency>=EVIDENCE_POLICY['minimum_direction_consistency'] and residual90<=EVIDENCE_POLICY['maximum_residual_noise_ratio']*noise)
        if supported:velocity=fitted
    anchor=np.median(pts-(ts-now)[:,None]*velocity,axis=0)
    track.update(velocity=velocity,anchor_world=anchor,motion_reliable=supported,
        stationary_compatible=not supported,position_growth_mps=0.,
        motion_audit=dict(status='supported_motion' if supported else 'unresolved_includes_stationary',
            fit_velocity_world_mps=fitted.tolist(),adopted_velocity_world_mps=velocity.tolist(),
            span_s=float(span),observations=len(recent),position_noise_m=noise,signal_m=signal,
            direction_consistency=consistency,residual90_m=residual90,innovation_m=innovation,
            motion_segment_reset=reset,physical_stationarity_certified=False))
    top=float(track['box'][2]+track['box'][5]);ground=track['declared_ground_z']
    track['height_supported']=bool(top-ground>=EVIDENCE_POLICY['ground_top_support_m'])
    track['evidence_tier']='temporal_neural_with_height_support' if track['height_supported'] else 'unverified_height'
