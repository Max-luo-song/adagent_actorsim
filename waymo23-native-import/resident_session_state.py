"""Opt-in resident UniAD episode isolation; no model weights are changed."""
import math
import os


def snapshot(core):
    reliability = core.scene_reliability
    tracks = core.test_track_instances
    return dict(adapter_scene=getattr(core, '_adapter_scene_token', None),
        adapter_timestamp=getattr(core, '_adapter_timestamp', None),
        bev_present=getattr(core, 'prev_bev', None) is not None,
        tracking_present=tracks is not None,
        tracking_count=0 if tracks is None else len(tracks),
        track_next_id=int(core.track_base.max_obj_id),
        reliability_tracks=len(reliability.tracks),
        reliability_next_id=int(reliability.next_id),
        reliability_scene=reliability.scene_token,
        reliability_timestamp=reliability.previous_time,
        tracking_pose_present=any(getattr(core, key, None) is not None
                                 for key in ('l2g_r_mat', 'l2g_t', 'timestamp')),
        metadata_bev_present=core.prev_frame_info.get('prev_bev') is not None)


def reset_episode(core):
    from run_reliable_sequence import synchronize_temporal_state
    synchronize_temporal_state(core, reset=True)
    # These are inference caches from UniADTrack / UniADE2E, separate from BEV.
    for key in ('l2g_r_mat', 'l2g_t', 'timestamp', 'scene_token'):
        if hasattr(core, key):
            setattr(core, key, None)
    core.prev_frame_info.update(prev_bev=None, scene_token=None, prev_pos=0, prev_angle=0)
    core.track_base.clear()
    if hasattr(core, 'geometry_candidate_refiner'):
        core.geometry_candidate_refiner = None


class ResidentSessionState:
    def __init__(self, core):
        self.core = core
        parameter = next(core.parameters())
        self.identity = dict(pid=os.getpid(), model_object_id=id(core),
                             first_parameter_object_id=id(parameter),
                             first_parameter_data_ptr=int(parameter.data_ptr()), model_loads=1)
        self.reset_count = 0

    def prepare(self, token, timestamp, expected_dt):
        from run_reliable_sequence import prepare_temporal_frame
        timestamp = float(timestamp)
        if not isinstance(token, str) or not token or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Invalid episode token/timestamp')
        if not math.isfinite(expected_dt) or expected_dt <= 0:
            raise ValueError('Invalid expected_dt')
        core = self.core
        before = snapshot(core)
        new_episode = (before['adapter_scene'] != token or before['adapter_timestamp'] is None)
        gap = (not new_episode and timestamp - before['adapter_timestamp'] >
               core.scene_reliability.config.max_gap_seconds)
        if new_episode or gap:
            reset_episode(core)
            self.reset_count += 1
        reset = prepare_temporal_frame(core, token, timestamp, expected_dt=expected_dt)
        after = snapshot(core)
        parameter = next(core.parameters())
        identity = dict(pid=os.getpid(), model_object_id=id(core),
                        first_parameter_object_id=id(parameter),
                        first_parameter_data_ptr=int(parameter.data_ptr()), model_loads=1)
        if identity != self.identity:
            raise RuntimeError('Resident model identity changed')
        cleared = (not after['bev_present'] and not after['tracking_present']
            and not after['metadata_bev_present'] and not after['tracking_pose_present']
            and after['track_next_id'] == after['reliability_tracks'] == after['reliability_next_id'] == 0
            and after['reliability_scene'] is None and after['reliability_timestamp'] is None)
        if reset and not cleared:
            raise RuntimeError('Resident episode reset left temporal state behind')
        return dict(identity=identity, reset=bool(reset), reset_count=self.reset_count,
                    reset_state_cleared=cleared if reset else None,
                    before=before, after=after)
