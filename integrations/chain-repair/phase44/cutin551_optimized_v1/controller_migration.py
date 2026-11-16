def apply(source):
 mark='    for name, seed, iteration_limit, search_stage in candidates:\n'
 block="""    if consensus is not None and runtime_policy.get('ranked_expansion'):
        from ranked_search import candidates as ranked_candidates
        def current_seed_score(name,seed):
            if name!='linear_reachable_stop':seed=bounded_seed(seed,speed,steer,control_interval_s,vehicle)
            return float(constraints(seed.ravel()).min()),float(objective(seed.ravel()))
        candidates=ranked_candidates(consensus,seeds,maxiter,current_seed_score)
"""
 source=source.replace(mark,block+mark)
 source=source.replace("            if runtime_policy.get('scaled_slsqp'):\n","""            if runtime_policy.get('joint_difference'):
                from joint_difference import solve
                sol=solve(minimize,objective,seed.ravel(),constraints,bounds,{"maxiter":iteration_limit,"ftol":1e-6,"disp":False})
            elif runtime_policy.get('scaled_slsqp'):
""")
 return source.replace('        trials.append(row)\n',"        if hasattr(sol,'joint_difference_metrics'):row['joint_difference_metrics']=sol.joint_difference_metrics\n        trials.append(row)\n")
