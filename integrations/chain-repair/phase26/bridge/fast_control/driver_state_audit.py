"""Capture freshly constructed per-episode control state before first feedback."""
def initial_driver_state(motion,warm,controller,obstacles,depth,road,navigation,route):
    checks=dict(warm_empty=warm is None,
        recovery_fresh=controller.state=='normal' and controller.previous_time is None
            and controller.clear_count==0 and controller.clear_since is None,
        obstacle_history_empty=not obstacles.tracks and obstacles.time is None
            and obstacles.scene is None and obstacles.next_id==0,
        depth_history_empty=not depth.cells and depth.scene is None and depth.time is None,
        visibility_history_empty=not depth.visibility.cells and not depth.visibility.hypotheses
            and depth.visibility.scene is None and depth.visibility.time is None,
        road_history_empty=not road.history,
        navigation_empty=navigation.points is None and navigation.arc is None
            and navigation.scene is None and navigation.time is None and navigation.progress==0,
        committed_route_empty=route.world is None and route.arc is None
            and route.scene is None and route.progress==0,
        local_motion_zero=motion.x==motion.z==motion.yaw==0,
        previous_action_zero=motion.last_accel==motion.last_rate==0)
    return dict(checks=checks,reset_method='fresh per-episode control objects',
        initial_speed_mps=float(motion.speed),initial_steering_rad=float(motion.steer),
        initial_timestamp=float(motion.timestamp),formal_evaluation_admitted=False)
