"""Road-constrained A* proposal and bounded pursuit seed, never a final control.

Only current predicted road/obstacles and the neural navigation goal are used.
The MPC must revalidate full time-dependent geometry and physical constraints.
"""
import heapq
import numpy as np


def grid_path(initial,goal,is_clear,bounds,resolution=.5):
    start=np.asarray(initial,dtype=float)[:2];goal=np.asarray(goal,dtype=float)
    low=np.asarray(bounds,dtype=float)[:2];high=np.asarray(bounds,dtype=float)[2:]
    if start.shape!=(2,) or goal.shape!=(2,) or np.any(high<=low):raise ValueError('Invalid road route bounds')
    # Anchor the lattice at the actual start so a snap cannot erase a narrow
    # initial conflict. The map/metric bounds remain fixed independent of scene.
    ix=np.arange(int(np.ceil((low[0]-start[0])/resolution)),int(np.floor((high[0]-start[0])/resolution))+1)
    iy=np.arange(int(np.ceil((low[1]-start[1])/resolution)),int(np.floor((high[1]-start[1])/resolution))+1)
    xx,yy=np.meshgrid(ix,iy);coordinates=np.c_[xx.ravel(),yy.ravel()];points=start+coordinates*resolution
    free=np.zeros(len(points),bool)
    for i in range(0,len(points),64):free[i:i+64]=is_clear(points[i:i+64])
    nodes={tuple(key):point for key,point,valid in zip(coordinates,points,free) if valid}
    if (0,0) not in nodes:return None,dict(reason='initial_road_route_cell_blocked')
    targets=[key for key,point in nodes.items() if point[1]>=max(start[1]+resolution,goal[1]-resolution)]
    if not targets:return None,dict(reason='no_forward_road_frontier')
    target=min(targets,key=lambda key:float(np.linalg.norm(nodes[key]-goal)))
    costs={(0,0):0.};parent={};heap=[(float(np.linalg.norm(start-nodes[target])),0.,(0,0))];closed=set()
    while heap:
        _,cost,node=heapq.heappop(heap)
        if node in closed:continue
        closed.add(node)
        if node==target:
            path=[node]
            while path[-1]!=(0,0):path.append(parent[path[-1]])
            return np.array([nodes[key] for key in reversed(path)]),dict(reason='found',expanded=len(closed),resolution_m=resolution)
        for dx,dy in [(-1,0),(1,0),(0,1),(0,-1),(-1,1),(1,1),(-1,-1),(1,-1)]:
            nxt=(node[0]+dx,node[1]+dy)
            if nxt not in nodes or nxt in closed:continue
            if dx and dy and ((node[0]+dx,node[1]) not in nodes or (node[0],node[1]+dy) not in nodes):continue
            distance=float(np.hypot(dx,dy)*resolution);new=cost+distance
            if new>=costs.get(nxt,float('inf')):continue
            costs[nxt]=new;parent[nxt]=node
            heapq.heappush(heap,(new+float(np.linalg.norm(nodes[nxt]-nodes[target])),new,nxt))
    return None,dict(reason='no_connected_road_route',expanded=len(closed))


def pursuit_controls(path,initial,vehicle,interval=1.,target_speed=2.):
    from closed172_common import integrate_state
    state=np.asarray(initial,dtype=float).copy();path=np.asarray(path,dtype=float);controls=[];index=0
    for _ in range(6):
        nearest=index+int(np.argmin(np.linalg.norm(path[index:]-state[:2],axis=1)));index=nearest
        distances=np.linalg.norm(path[index:]-state[:2],axis=1)
        ahead=np.flatnonzero(distances>=vehicle.wheelbase_m)
        point=path[index+(int(ahead[0]) if len(ahead) else len(distances)-1)]
        delta=point-state[:2];yaw=state[2]
        right=float(delta@np.array([np.cos(yaw),-np.sin(yaw)]))
        forward=float(delta@np.array([np.sin(yaw),np.cos(yaw)]))
        if forward<=0:desired=0.
        else:desired=float(np.arctan2(2*vehicle.wheelbase_m*right,max(float(delta@delta),1e-6)))
        desired=np.clip(desired,-np.deg2rad(vehicle.max_steer_degrees),np.deg2rad(vehicle.max_steer_degrees))
        rate=float(np.clip((desired-state[4])/interval,-np.deg2rad(vehicle.max_steer_rate_degrees),np.deg2rad(vehicle.max_steer_rate_degrees)))
        acc=float(np.clip((target_speed-state[3])/interval,vehicle.min_acc_mps2,vehicle.max_acc_mps2))
        controls.append([acc,rate])
        for _ in range(int(round(interval/vehicle.dt))):state=integrate_state(state,[acc,rate],vehicle.dt,vehicle.wheelbase_m)
    return np.array(controls)


def propose(reference,initial,road,temporal,guard,vehicle,interval):
    goal=np.asarray(reference[-1],dtype=float)
    # Geometry uses vehicle-center coordinates; reference is the virtual LiDAR
    # path. The static offset is applied by the caller before this function.
    width=max(4*vehicle.wheelbase_m,abs(goal[0]-initial[0])+2*vehicle.wheelbase_m)
    low=np.maximum(road.bounds[:2],[initial[0]-width,initial[1]-vehicle.wheelbase_m])
    high=np.minimum(road.bounds[2:],[initial[0]+width,max(goal[1],initial[1])+2*vehicle.wheelbase_m])
    def clear(points):
        states=np.zeros((len(points),5));states[:,:2]=points
        times=np.zeros(len(states))
        # These are independent spatial queries at t=0, not a temporal path.
        semantic=np.array([guard.clearance(state[None],np.zeros(1))[0] for state in states])
        return np.minimum.reduce([road.clearance(states),temporal.clearance(states,times),semantic])>=0
    path,audit=grid_path(initial,goal,clear,np.r_[low,high])
    audit.update(ground_truth_used=False,scope='current geometry road-route seed only; requires MPC temporal revalidation')
    if path is None:return None,audit
    speed=min(guard.caution_speed,guard.speed_limit) if guard.speed_limit is not None else guard.caution_speed
    controls=pursuit_controls(path,initial,vehicle,interval,speed)
    audit['path_xy']=path.tolist()
    return controls,audit


def spatial_clearance(states,road,temporal,guard):
    """Independent present-time queries, never a fake temporal sequence."""
    states=np.asarray(states,dtype=float);times=np.zeros(len(states))
    semantic=np.array([guard.clearance(state[None],np.zeros(1))[0] for state in states])
    return np.minimum.reduce([road.clearance(states),temporal.clearance(states,times),semantic])


def detour_reference(reference,initial,path,offset):
    """Sample a predicted-geometry path using the NN's arc-length schedule.

    No side preference, object identity, asset pose or simulator road labels.
    A short route ends at its endpoint; extrapolation never cuts a corner.
    """
    reference=np.asarray(reference,dtype=float);path=np.asarray(path,dtype=float)
    center_reference=reference-np.asarray(offset)
    requested=np.cumsum(np.linalg.norm(np.diff(np.vstack([np.asarray(initial)[:2],center_reference]),axis=0),axis=1))
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    if len(path)<2 or np.any(np.diff(arc)<=0):raise ValueError('Invalid detour polyline')
    return np.c_[np.interp(requested,arc,path[:,0]),np.interp(requested,arc,path[:,1])]+offset
