@_polygon_backend_scope
def optimize(reference, occupancy, info, warm=None, maxiter=220, reliability=None, vehicle=None, control_interval_s=None, _forecast_recovery=False, _reuse_verified_warm=False, _continuation=None, _search_policy=None, _warm_elapsed_s=None, _check_continuation_only=False, _runtime_policy=None, _problem_callback=None, _proposal_provider=None):
    runtime_policy=json.loads((Path(__file__).with_name('runtime_policy.json')).read_text()) if _runtime_policy is None else dict(_runtime_policy)
    memo=Memo()
    fallback_builder=None
    query_caches={}
    seed_solutions={}
    seed_cache_hits=0
    def _finish(result):
        result['runtime_performance_policy']=runtime_policy
        result['prefix_rollout_metrics']=dict(prefix.metrics) if prefix is not None else None
        result['query_cache_stats']={name:dict(value) for name,value in query_caches.items()}
        result['same_problem_seed_cache']={'hits':seed_cache_hits,'unique_solved_starts':len(seed_solutions),'cross_frame_reuse':False}
        result['planning_input_source']=(reliability or {}).get('planning_input_source','fresh_model_forecast')
        if not _check_continuation_only and fallback_builder is not None and (result.get('selected') is None or (reliability or {}).get('need_braking_fallback')):
            result['braking_fallback']=fallback_builder()
        result.setdefault('performance_cache_layers',[]).append(dict(control_interval_s=control_interval_s,forecast_recovery=_forecast_recovery,caches=memo.stats()))
        return result
    from current_reference import prepare_report,ROAD_SOFT_COST_WEIGHT
    reliability=prepare_report(reliability)
    # A previous solution may seed the solver; it never bypasses fresh optimization.
    _reuse_verified_warm=False
    vehicle = vehicle or load_vehicle_config()
    if control_interval_s is None:
        control_interval_s=float(json.loads(((Path(__file__).resolve().parents[7]/'waymo23-native-import')/'configs/planning.json').read_text())['control_interval_s'])
    if not np.isfinite(control_interval_s) or control_interval_s<.5 or not np.isclose(control_interval_s/vehicle.dt,round(control_interval_s/vehicle.dt)) or not np.isclose(control_interval_s/.5,round(control_interval_s/.5)):
        raise ValueError('Invalid planning control interval')
    interpolation=(reliability or {}).get("control_interpolation","constant")
    if interpolation not in ("constant","linear"):raise ValueError("Invalid control interpolation")
    horizon=6*control_interval_s
    substeps = int(round(.5/vehicle.dt))
    from prefix_rollout import PrefixRollout
    prefix=None
    @memo.wrap
    def simulate_controls(controls, speed, steer, offset):
        return prefix(controls) if prefix is not None else rollout(controls, speed, steer, offset, vehicle.dt, vehicle.wheelbase_m,control_interval_s,interpolation=interpolation)
    def simulate(controls,speed,steer,offset):
        return simulate_controls(np.asarray(controls,dtype=float).reshape(6,2),speed,steer,offset)
    reference = np.asarray(reference, dtype=float).reshape(6,2)
    original_neural_reference=reference.copy()
    # The frozen neural head supplies only 3s. Beyond it, continue its final
    # navigation tangent; this is a deterministic planner goal, not a new NN
    # prediction or an asset-derived bypass path.
    extra_times=np.arange(3.5,horizon+1e-8,.5)
    if len(extra_times):
        reference=np.vstack([reference,reference[-1]+(extra_times[:,None]-3.)*(reference[-1]-reference[-2])/.5])
    from temporal_context import AgentlessForecast
    agentless=isinstance(occupancy,AgentlessForecast)
    static_snapshot=(reliability or {}).get('declared_static_epoch') is not None
    if agentless:
        contract=(reliability or {}).get('agentless_foreground_contract',{})
        if contract.get('native_skip_reason')!='zero_motion_queries' or contract.get('raster_fabricated') is not False or contract.get('physical_scene_asserted_empty') is not False:raise ValueError('Missing healthy agentless foreground contract')
        masks=None;rc=np.empty((0,2),dtype=int)
    else:
        masks=np.asarray(occupancy)
        if masks.ndim==5:masks=masks.squeeze(2)
        if masks.shape!=(1,5,200,200) or not np.isfinite(masks).all():raise ValueError('Unexpected occupancy: '+str(masks.shape))
        if static_snapshot:masks=np.repeat(masks[:,:1],5,axis=1)
        rc=np.argwhere(np.any(masks[0]!=0,axis=0))
    occupied=(rc[:,[1,0]].astype(float)-100.)*.5+.25
    from context_factory import make_temporal
    temporal_occupancy=make_temporal(masks,info,reliability or {},vehicle)
    from drivable_area import DrivableArea
    road_contract=(reliability or {}).get('predicted_drivable_area')
    if (reliability or {}).get('require_predicted_drivable_area') and road_contract is None:
        raise ValueError('Missing required predicted drivable area')
    road=DrivableArea(road_contract,vehicle) if road_contract is not None else None
    if road is not None:
        if runtime_policy.get('compiled_road') and hasattr(road,'cell_boxes'):
            from fast_road import NativeRoadClearance
            road.clearance=NativeRoadClearance(road)
        from exact_queries import RoadPoseCache
        if runtime_policy.get('bulk_pose_cache'):
            from bulk_road_cache import BulkRoadPoseCache as RoadPoseCache
        road_cache=RoadPoseCache(road.clearance);road.clearance=road_cache;query_caches['road_pose']=road_cache.metrics
    road_mode=(reliability or {}).get('predicted_road_constraint_mode','hard')
    if road_mode not in ('hard','soft'):raise ValueError('Invalid predicted road constraint mode')
    road_hard=road_mode=='hard'
    c = np.asarray(info["model_lidar_frame"]["virtual_to_ego"],dtype=float)
    if c.shape != (4,4) or not np.isfinite(c).all() or not np.allclose(c[:3,:3], np.eye(3)):
        raise ValueError("Only axis-aligned virtual mount supported")
    offset = c[:2,3]
    speed, steer = float(info["ego_velo"]), float(info["ego_steer"])
    if speed < -1e-6 or not np.isfinite([speed,steer]).all():
        raise ValueError("Invalid initial motion state")
    prefix=PrefixRollout(speed,steer,offset,vehicle.dt,vehicle.wheelbase_m,control_interval_s,interpolation) if runtime_policy.get('rollout_prefix_reuse') else None
    initial = np.array([-offset[0],-offset[1],0.,speed,steer])
    reaction = float((reliability or {}).get('config',{}).get('confirmation_seconds',.2))
    if not np.isfinite(reaction) or reaction<=0:raise ValueError('Invalid reaction interval')
    braking_states=[initial.copy()]
    for tick in range(int(round(horizon/vehicle.dt))):
        state=braking_states[-1]
        acc=0. if tick*vehicle.dt<reaction-1e-9 else -min(vehicle.emergency_braking_mps2,state[3]/vehicle.dt)
        braking_states.append(integrate_state(state,[acc,0.],vehicle.dt,vehicle.wheelbase_m))
    braking_states=np.asarray(braking_states)
    guard=PlanningGuard(reliability,vehicle,braking_states,np.arange(len(braking_states))*vehicle.dt,current_occupancy=None if masks is None or (reliability or {}).get('retained_forecast') else masks[0,0])
    from polygon_backend import EpistemicRisk
    epistemic=EpistemicRisk(reliability,info,vehicle)
    risk_probes=np.zeros((len(reference)+1,5));risk_probes[0]=initial;risk_probes[1:,:2]=reference-offset
    if epistemic.near_path(risk_probes):
        guard.speed_limit=min(guard.caution_speed,guard.speed_limit) if guard.speed_limit is not None else guard.caution_speed
    audit=guard.audit()
    audit['epistemic_unknown']=dict(points=len(epistemic.points),policy='risk_cost_and_reachable_caution_cap',
        hard_boundary_band_preserved=True,free_space_asserted=False,near_reference=epistemic.near_path(risk_probes))
    audit['occupancy']=temporal_occupancy.audit()
    audit['drivable_area']=road.audit() if road is not None else dict(enabled=False,reason='legacy request without road contract')
    audit['predicted_road_constraint_mode']=road_mode
    audit['road_soft_cost_weight']=ROAD_SOFT_COST_WEIGHT if not road_hard else 0.0
    from road_soft_cost import SoftRoadPenalty
    road_soft=SoftRoadPenalty(road,vehicle) if road is not None and not road_hard else None
    audit['road_soft_cost_policy']=road_soft.audit() if road_soft is not None else None
    audit['navigation_anchor']=(reliability or {}).get('navigation_anchor_audit')
    audit['original_neural_reference']=original_neural_reference.tolist()
    audit['control_interpolation']=interpolation
    audit['first_applied_interval_s']=vehicle.dt
    audit['forecast_recovery']=bool(_forecast_recovery)
    audit['declared_static_snapshot']=static_snapshot
    if static_snapshot and masks is not None:audit['occupancy']['future_policy']='declared immutable snapshot: hold current raster'
    if _forecast_recovery:
        guard.speed_limit=min(guard.caution_speed,guard.speed_limit) if guard.speed_limit is not None else guard.caution_speed
        audit['speed_limit_mps']=guard.speed_limit
    @memo.wrap
    def clear(states):
        times=np.arange(len(states))*vehicle.dt
        raw=temporal_occupancy.clearance(states,np.zeros(len(states)) if _forecast_recovery else times)
        clearance=np.minimum(raw,guard.clearance(states,times))
        clearance=clearance-sweep_allowance(states,vehicle)
        if road is not None and road_hard:clearance=np.minimum(clearance,road.swept_clearance(states))
        return clearance

    from depth_ambiguity import ObservationViability
    ambiguity=ObservationViability(reliability or {},info,vehicle,motion_guard=guard)
    @memo.wrap
    def robust_clear(states):
        times=np.arange(len(states))*vehicle.dt
        gap=np.minimum(temporal_occupancy.clearance(states,np.zeros(len(states)) if _forecast_recovery else times),getattr(guard,'full_clearance',guard.clearance)(states,times))-sweep_allowance(states,vehicle)
        if road is not None and road_hard:gap=np.minimum(gap,road.swept_clearance(states))
        return gap
    ambiguity.known_clear=robust_clear
    audit['depth_ambiguity']=ambiguity.audit()
    fallback_clear=lambda states:np.minimum(robust_clear(states),ambiguity.clearance(states)) if ambiguity.active else clear(states)
    from motion_policy import braking_plan,speed_budget
    fallback_builder=lambda:braking_plan(initial,original_neural_reference-offset,vehicle,fallback_clear,road,forecast_available=True,risk_cost=lambda states:guard.soft_cost(states,np.arange(len(states))*vehicle.dt)+epistemic.cost(states))
    uncertain=agentless or bool((reliability or {}).get('retained_forecast')) or bool((reliability or {}).get('perception_degraded')) or bool((reliability or {}).get('need_braking_fallback'))
    if uncertain:
        cap,budget=speed_budget(initial,vehicle,temporal_occupancy,reaction)
        guard.speed_limit=min(cap,guard.speed_limit) if guard.speed_limit is not None else cap
        audit['confidence_speed_budget']=budget
        audit['speed_limit_mps']=guard.speed_limit
    initial_raw=float(temporal_occupancy.clearance(initial[None],[0.])[0])
    initial_proposal=float(guard.clearance(initial[None],np.zeros(1))[0])
    initial_road=float(road.clearance(initial[None])[0]) if road is not None else float('inf')
    if road_hard and initial_road < -1e-5:
        return _finish(dict(selected=None,trials=[],reason='fixed_initial_drivable_conflict',
            initial_conflict=dict(center_xy=(-offset).tolist(),predicted_road_clearance_m=initial_road),
            guard=audit,ground_truth_used_by_optimizer=False,
            policy='predicted road excludes current footprint; do not fabricate road under ego'))
    initial_clearance=min(initial_raw,initial_proposal)
    if initial_clearance < -1e-5:
        return _finish(dict(selected=None,trials=[],reason="fixed_initial_occupancy_conflict",
            initial_conflict=dict(center_xy=(-offset).tolist(),clearance_m=initial_clearance,
                raw_occupancy_clearance_m=initial_raw,
                hard_proposal_clearance_m=initial_proposal if np.isfinite(initial_proposal) else None,
                introduced_by_uncertainty_guard=bool(initial_raw>=-1e-5 and initial_proposal< -1e-5)),
            occupied_union_cells=int(len(rc)),uncertainty_guard_enabled=reliability is not None,
            ground_truth_used_by_optimizer=False,guard=audit,
            policy="fixed initial state infeasible; bounded braking and continuous reassessment",
            steering_physical_limit_validated=False))
    rate_limit = np.deg2rad(vehicle.max_steer_rate_degrees)
    bounds = [b for _ in range(6) for b in [(vehicle.min_acc_mps2,vehicle.max_acc_mps2),(-rate_limit,rate_limit)]]


    @memo.wrap
    def constraints(flat):
        states,_ = simulate(flat,speed,steer,offset)
        constraints = [clear(states), states[:,3], vehicle.max_speed_mps-states[:,3],
                       np.deg2rad(vehicle.max_steer_degrees)-np.abs(states[:,4])]
        if guard.speed_limit is not None:
            # An instantaneous speed cap is infeasible when already moving
            # faster: enforce a reachable braking envelope instead.
            times = np.arange(len(states))*vehicle.dt
            envelope = np.maximum(guard.speed_limit,
                                  speed-vehicle.emergency_braking_mps2*times)
            constraints.append(envelope-states[:,3])
        if ambiguity.active:constraints.append(np.array([ambiguity.gap(initial,np.asarray(flat).reshape(6,2)[0])]))
        return np.concatenate(constraints)

    navigation_reference=reference.copy()
    from native_objective import NativeAlignedObjective
    aligned=NativeAlignedObjective(original_neural_reference,vehicle,road_soft,temporal_occupancy,guard,clear)
    aligned.cruise_preference=float((reliability or {}).get("cruise_speed_preference_mps",speed))
    aligned.epistemic=epistemic
    query_caches['ttc']=aligned.forecast_memo.metrics['_forecast_query']
    audit['native_alignment']=aligned.audit()
    @memo.wrap
    def objective(flat):
        controls=np.asarray(flat).reshape(6,2)
        states,lidar=simulate(controls,speed,steer,offset)
        return aligned.cost(states,lidar,controls,navigation_reference)

    audit['navigation_reference']=dict(source='current_frame_neural_trajectory',
        original_neural_reference=original_neural_reference.tolist(),tracking_reference=navigation_reference.tolist(),
        native_prediction_horizon_s=3.,planning_horizon_s=horizon,
        tail='current frame terminal tangent only; not a neural future observation',
        history_navigation_override_used=False,road_route_used_as_seed_only=True,ground_truth_used=False)
    from mpc_schedule import check_continuation,SearchSchedule,POLICY
    continued,schedule_audit=check_continuation(_continuation,initial,offset,vehicle,clear,guard,aligned,navigation_reference,control_interval_s,interpolation,_search_policy or POLICY,ambiguity=ambiguity)
    audit['scheduling']=schedule_audit
    if continued is not None:
        return _finish(dict(selected=continued,trials=[],forecast_recovery=False,planning_horizon_s=horizon,neural_horizon_s=3.,guard=audit,reason=None,ground_truth_used_by_optimizer=False,policy='fresh_checked_remaining_plan'))
    if _check_continuation_only:
        return _finish(dict(selected=None,trials=[],guard=audit,reason=schedule_audit['trigger'],ground_truth_used_by_optimizer=False))
    if _problem_callback is not None or _proposal_provider is not None:
        from direct_problem import PlanningProblem
        problem=PlanningProblem(reference=original_neural_reference,info=info,reliability=reliability,vehicle=vehicle,initial=initial,offset=offset,interval=control_interval_s,interpolation=interpolation,bounds=bounds,simulate=simulate,constraints=constraints,objective=objective,clear=clear,aligned=aligned,guard=guard,temporal=temporal_occupancy,road=road,audit=audit)
        if _problem_callback is not None:return _problem_callback(problem)
        decision=_proposal_provider(problem)
        audit['student_policy']=decision['audit']
        if decision.get('selected') is not None:
            return _finish(dict(selected=decision['selected'],trials=[],forecast_recovery=False,planning_horizon_s=horizon,neural_horizon_s=3.,guard=audit,reason=None,ground_truth_used_by_optimizer=False,policy='direct_distilled_policy_checked_v1'))
    seeds = []
    if road is not None:
        from mpc_perf_v1.road_route_seed import propose
        route_seed,route_audit=propose(reference-offset,initial,road,temporal_occupancy,guard,vehicle,control_interval_s,road_hard=road_hard)
        audit['road_route_proposal']=route_audit
        if route_seed is not None:seeds.append(('predicted_road_route',route_seed))
    if warm is not None:
        previous = np.asarray(warm,dtype=float).reshape(6,2)
        shifted = np.stack([np.interp(np.arange(6)*control_interval_s+(vehicle.dt if _warm_elapsed_s is None else _warm_elapsed_s),
                            np.arange(6)*control_interval_s, previous[:,j]) for j in range(2)],axis=1)
        seeds.insert(0,("warm",shifted))
    # The old constant -1.9 seed crossed v=0 and started infeasible. Build a
    # reachable stop at the actual control interval without negative speed.
    stopping=np.zeros((6,2));remaining=speed
    for i in range(6):
        stopping[i,0]=-min(vehicle.emergency_braking_mps2,remaining/control_interval_s)
        remaining=max(0.,remaining+stopping[i,0]*control_interval_s)
    seeds.append(('reachable_stop',stopping))
    if interpolation=='linear':
        ticks=np.arange(int(round(horizon/vehicle.dt)))*vehicle.dt
        basis=np.eye(6)
        weights=np.array([np.interp(ticks,np.arange(6)*control_interval_s,basis[:,j]).sum()*vehicle.dt for j in range(6)])
        linear_stop=np.zeros((6,2));remaining=speed
        for j,weight in enumerate(weights):
            linear_stop[j,0]=-min(vehicle.emergency_braking_mps2,remaining/weight)
            remaining=max(0.,remaining+linear_stop[j,0]*weight)
        seeds.append(('linear_reachable_stop',linear_stop))
    for sign in (-1.,1.):
        turning_stop=stopping.copy();turning_stop[:2,1]=sign*rate_limit
        seeds.append(('reachable_left_stop' if sign<0 else 'reachable_right_stop',turning_stop))
    seeds.append(("coast",np.zeros((6,2))))
    brake = np.zeros((6,2)); brake[:,0] = -1.9
    seeds.append(("brake",brake))
    left = np.array([[-1.5,-.24],[-1.5,-.20],[-1.,.05],[-.5,.20],[0,.15],[0,0.]])
    seeds.append(("left_brake",left))
    right = left.copy(); right[:,1] *= -1
    seeds.append(("right_brake",right))
    # Diverse physically bounded starts explore either side of a blocking
    # proposal. They use neither object truth nor a hand-authored bypass path.
    for sign in (-1.,1.):
        escape=np.zeros((6,2));escape[:,0]=[-vehicle.emergency_braking_mps2,-vehicle.emergency_braking_mps2,0.,.5,.5,.5]
        escape[:,1]=sign*rate_limit*np.array([1.,1.,0.,-1.,-1.,0.])
        seeds.append(('escape_left' if sign<0 else 'escape_right',escape))
    # Replace accelerating-to-physical-max lane-change starts with cruising
    # detours. Same two symmetric starts; all candidate constraints unchanged.
    from cruise_seeds import propose as cruising_detours
    detour_seeds,detour_audit=cruising_detours(initial,vehicle,guard,temporal_occupancy,horizon,control_interval_s)
    seeds.extend(detour_seeds);audit['cruising_detour_starts']=detour_audit
    trials = []
    selected = None
    search=SearchSchedule(_search_policy,maxiter) if _search_policy else None
    if search:audit['search_schedule']=search.audit
    consensus=None
    if search is None and runtime_policy.get('consensus_search'):
        if runtime_policy.get('adaptive_consensus',True):
            from consensus_search import ConsensusSearch
            quality_speed=min(aligned.cruise_preference,speed)
        else:
            from baseline_consensus import ConsensusSearch
            quality_speed=aligned.cruise_preference
        consensus=ConsensusSearch(POLICY,max(0.,quality_speed-vehicle.emergency_braking_mps2*POLICY['replan_period_s']),aligned.stop)
        audit['search_schedule']=consensus.audit
    candidates=consensus.candidates(seeds,maxiter) if consensus else (search.candidates(seeds) if search else ((name,seed,maxiter,'legacy') for name,seed in seeds))
    if consensus is not None and runtime_policy.get('ranked_expansion'):
        from ranked_search import candidates as ranked_candidates
        def current_seed_score(name,seed):
            if name!='linear_reachable_stop':seed=bounded_seed(seed,speed,steer,control_interval_s,vehicle)
            return float(constraints(seed.ravel()).min()),float(objective(seed.ravel()))
        candidates=_student_candidates(ranked_candidates,consensus,seeds,maxiter,current_seed_score,lambda: _student_provider(_student_problem(reference=original_neural_reference,info=info,reliability=reliability,vehicle=vehicle,initial=initial,offset=offset,interval=control_interval_s,interpolation=interpolation,bounds=bounds,simulate=simulate,constraints=constraints,objective=objective,clear=clear,aligned=aligned,guard=guard,temporal=temporal_occupancy,road=road,audit=audit),seeds,current_seed_score),audit)
    for name, seed, iteration_limit, search_stage in candidates:
        if name!='linear_reachable_stop':seed=bounded_seed(seed,speed,steer,control_interval_s,vehicle)
        seed_residual=float(constraints(seed.ravel()).min())
        warm_reference_match=True
        warm_reference_max_deviation=None
        if name=='warm':
            _,warm_lidar=simulate(seed,speed,steer,offset)
            warm_reference_max_deviation=float(np.linalg.norm(warm_lidar[substeps::substeps]-navigation_reference,axis=1).max())
            warm_reference_match=warm_reference_max_deviation<=vehicle.half_width_m
        reuse=bool(name=='warm' and _reuse_verified_warm and seed_residual>=-1e-5 and warm_reference_match)
        duplicate=False
        solve_started=time.perf_counter()
        seed_key=np.ascontiguousarray(seed,dtype=float).tobytes()
        if not reuse and (seed_key,iteration_limit) in seed_solutions:
            from types import SimpleNamespace
            saved=seed_solutions[(seed_key,iteration_limit)];sol=SimpleNamespace(x=saved['x'].copy(),success=saved['success'],message=saved['message']);duplicate=True;seed_cache_hits+=1
        elif reuse:
            from types import SimpleNamespace
            sol=SimpleNamespace(x=seed.ravel(),success=True,message='Fresh full-constraint verification; optimization skipped')
        else:
            if runtime_policy.get('joint_difference'):
                from joint_difference import solve
                sol=solve(minimize,objective,seed.ravel(),constraints,bounds,{"maxiter":iteration_limit,"ftol":1e-6,"disp":False})
            elif runtime_policy.get('scaled_slsqp'):
                from scaled_solver import solve
                sol=solve(minimize,objective,seed.ravel(),constraints,bounds,{"maxiter":iteration_limit,"ftol":1e-6,"disp":False})
            else:
                sol = minimize(objective,seed.ravel(),method="SLSQP",bounds=bounds,
                               constraints=[{"type":"ineq","fun":constraints}],
                               options={"maxiter":iteration_limit,"ftol":1e-6,"disp":False})
            seed_solutions[(seed_key,iteration_limit)]={'x':np.array(sol.x,copy=True),'success':bool(sol.success),'message':str(sol.message)}
        candidate=sol.x
        candidate_residual=float(constraints(candidate).min())
        used_seed=bool(seed_residual>=-1e-5 and (not np.isfinite(candidate_residual) or candidate_residual< -1e-5))
        if used_seed:candidate=seed.ravel()
        states,lidar = simulate(candidate,speed,steer,offset)
        residual = float(constraints(candidate).min())
        clearance = float(clear(states).min())
        controls = candidate.reshape(6,2)
        controls_valid = bool(np.all(controls[:,0] >= vehicle.min_acc_mps2-1e-8)
                              and np.all(controls[:,0] <= vehicle.max_acc_mps2+1e-8)
                              and np.all(np.abs(controls[:,1]) <= rate_limit+1e-8))
        feasible = bool(np.isfinite(states).all()
                        and residual >= -1e-5 and controls_valid)
        row = dict(start_key=seed_key.hex(),seed=name,solver_success=None if reuse else bool(sol.success),optimization_skipped=bool(reuse or duplicate),identical_seed_same_problem_reused=duplicate,optimization_skip_scope='same_current_problem_identical_start' if duplicate else None,message=str(sol.message),
                   feasible=feasible,minimum_constraint=residual,seed_name_is_final_direction=False,
                   lateral_extent_m=[float(states[:,0].min()-states[0,0]),float(states[:,0].max()-states[0,0])],
                   clearance_lower_bound_m=clearance,objective=float(objective(candidate)),
                   used_verified_seed=used_seed,warm_reference_match=warm_reference_match,warm_reference_max_deviation_m=warm_reference_max_deviation,controls=controls.tolist(),states=states.tolist(),lidar=lidar.tolist(),control_interval_s=control_interval_s,
                   future_occupancy_clearance_m=float((temporal_occupancy.clearance(states,np.arange(len(states))*vehicle.dt)-sweep_allowance(states,vehicle)).min()))
        row['native_alignment_proxy']=aligned.components(states,lidar,controls)
        row['soft_risk_cost']=float(guard.soft_cost(states,np.arange(len(states))*vehicle.dt)+epistemic.cost(states))
        row.update(search_stage=search_stage,iteration_limit=iteration_limit,solver_iterations=getattr(sol,'nit',None),objective_evaluations=getattr(sol,'nfev',None),solve_wall_seconds=time.perf_counter()-solve_started)
        if hasattr(sol,'joint_difference_metrics'):row['joint_difference_metrics']=sol.joint_difference_metrics
        trials.append(row)
        if name.startswith('student_initial_'):
            admitted=_student_admit(row,consensus)
            row['student_initial_only']=True
            row['student_post_solve_admitted']=admitted
            if not admitted:
                feasible=False
                row['feasible']=False
        if search:search.observe(row)
        if consensus:consensus.observe(row)
        if feasible and (selected is None or row['objective'] < selected['objective']):
            selected = row
        if feasible and name=='warm' and search is None and consensus is None:
            if reuse:break
            # Keep a verified moving continuation, but never let the first
            # feasible stationary plan suppress side-escape exploration.
            moving=states[-1,3]>=min(speed,guard.caution_speed)-1e-6
            progressing=states[-1,1]-states[0,1]>=max(0.,reference[5,1])
            neural_stop=np.linalg.norm(reference[5])<vehicle.half_length_m
            if ((moving and progressing) or neural_stop) and warm_reference_match: break
    if selected is None and not _check_continuation_only:
        from feasibility_restore import restore
        from spatial_reference import POLICY as restoration_policy
        restored,attempts=restore(trials,constraints,objective,bounds,maxiter=min(maxiter,restoration_policy['feasibility_iterations']),starts=restoration_policy['feasibility_starts'])
        audit['feasibility_restoration']=dict(attempts=attempts,hard_constraints_relaxed=False)
        if restored is not None:
            controls=np.asarray(restored['x']).reshape(6,2);states,lidar=simulate(controls,speed,steer,offset)
            selected=dict(seed='feasibility_restored_'+restored['seed'],solver_success=True,optimization_skipped=False,feasible=True,minimum_constraint=restored['minimum_constraint'],clearance_lower_bound_m=float(clear(states).min()),objective=restored['objective'],controls=controls.tolist(),states=states.tolist(),lidar=lidar.tolist(),control_interval_s=control_interval_s,native_alignment_proxy=aligned.components(states,lidar,controls),soft_risk_cost=float(guard.soft_cost(states,np.arange(len(states))*vehicle.dt)+epistemic.cost(states)),seed_name_is_final_direction=False)
            trials.append(selected)
    if selected is None and control_interval_s>.5 and not _forecast_recovery:
        # A coarse long horizon must not remove a physically reachable stop
        # available on the original fine control grid.
        fallback=optimize(np.asarray(reference[:6]),occupancy,info,warm=None,maxiter=maxiter,
                          reliability=reliability,vehicle=vehicle,control_interval_s=.5,_search_policy=_search_policy,_runtime_policy=runtime_policy)
        fallback['long_horizon_attempt']=dict(horizon_s=horizon,
            feasible=False,trials=[dict(seed=t['seed'],minimum_constraint=t['minimum_constraint'],message=t['message']) for t in trials])
        if fallback['selected'] is not None:return _finish(fallback)
        restored=optimize(np.asarray(reference[:6]),occupancy,info,warm=warm,maxiter=maxiter,
            reliability=reliability,vehicle=vehicle,control_interval_s=control_interval_s,_forecast_recovery=True,_search_policy=_search_policy,_runtime_policy=runtime_policy)
        restored['forecast_infeasible_attempt']=dict(long=fallback['long_horizon_attempt'],
            fine_trials=[dict(seed=t['seed'],minimum_constraint=t['minimum_constraint']) for t in fallback['trials']])
        return _finish(restored)
    return _finish(dict(selected=selected,trials=trials,forecast_recovery=bool(_forecast_recovery),
                occupancy_constraint_policy=('current mask held hard; confirmed motion hard; future masks risk cost; reachable caution speed' if _forecast_recovery else 'hard matching-time forecast'),planning_horizon_s=horizon,neural_horizon_s=3.,
                reference_extension="last neural navigation tangent; no object truth",occupied_union_cells=int(len(rc)),
                uncertainty_guard_enabled=reliability is not None,guard=audit,
                ground_truth_used_by_optimizer=False,
                reason=None if selected is not None else 'no_feasible_plan',
                policy="verified progressing warm continuation; feasible multistart; fine-grid stopping; forecast-risk recovery with hard observed geometry",
                steering_physical_limit_validated=False))
