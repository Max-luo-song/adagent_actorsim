"""CPU verification of integration, transforms, collision checks and feasible MPC."""
import ast
import math
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from closed172_common import (B,initial_native_pose,virtual_global_pose,native_delta,
    integrate_state,rollout,polygon_check,optimize,atomic_json,wait_json)


def main():
    rng = np.random.default_rng(172)
    c = np.eye(4);c[:3,3] = [0.,.985793,1.84019]
    for _ in range(30):
        native = np.eye(4)
        native[:3,:3] = Rotation.from_euler("XYZ",rng.normal(size=3)*.3).as_matrix()
        native[:3,3] = rng.normal(size=3)*10
        virtual = virtual_global_pose(native,c)
        info = dict(explicit_l2g_r_mat=virtual[:3,:3],explicit_l2g_t=virtual[:3,3],
                    model_lidar_frame=dict(virtual_to_ego=c))
        np.testing.assert_allclose(initial_native_pose(info),native,atol=1e-12)
        controls = np.c_[rng.uniform(-1,1,6),rng.uniform(-.1,.1,6)]
        states,lidar = rollout(controls,4.,.01,c[:2,3])
        for state,predicted in zip(states,lidar):
            future = native @ native_delta(state[0]+c[0,3],state[1]+c[1,3],state[2])
            mount_global = virtual_global_pose(future,c)
            actual = (np.linalg.inv(virtual) @ mount_global)[:2,3]
            np.testing.assert_allclose(predicted,actual,atol=2e-12)

    # Reproduce the returned frame165 trajectory from its rounded controls.
    controls = np.array([[-1.9755,.0276],[-1.0166,.0171],[-.0037,.008],
                         [.6292,.0034],[.8443,.0021],[.8291,.0018]])
    states,lidar = rollout(controls,4.610465579924064,0.,c[:2,3])
    expected = np.array([[.0032,1.0231],[.0200,2.6819],[.0559,4.2381],
                         [.1191,5.8875],[.2226,7.7248],[.3800,9.7677]])
    np.testing.assert_allclose(states[5::5,:2],expected,atol=.002)
    np.testing.assert_allclose(lidar[0],[0,0])

    # Compare executed first action against the exact planar update formula.
    old = np.array([3.,4.,.31,4.7,-.04])
    action = [-1.2,.1]
    actual = integrate_state(old,action)
    speed = old[3]+action[0]*.1
    steer = old[4]+action[1]*.1
    expected = [old[0]+speed*math.sin(old[2])*.1,
                old[1]+speed*math.cos(old[2])*.1,
                old[2]+speed*math.tan(steer)/2.7*.1,speed,steer]
    np.testing.assert_allclose(actual,expected,atol=1e-14)

    base = os.environ.get("BASE")
    if base:
        source = Path(base)/"waymo23-native-import/live_env.py"
        if source.is_file():
            tree = ast.parse(source.read_text())
            node = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == "Motion")
            scope = dict(np=np,math=math)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
                         str(source),"exec"),scope)
            motion = scope["Motion"](dt=.1,wheelbase=2.7,start_speed=old[3])
            motion.x,motion.z,motion.yaw,motion.speed,motion.steer = map(float,old)
            motion.advance(dict(acc=action[0],steer_rate=action[1]))
            np.testing.assert_allclose([motion.x,motion.z,motion.yaw,motion.speed,motion.steer],
                                       actual,atol=1e-12)
            print("PASS: current server Motion.advance matches optimizer")

    square = np.array([[0.,0],[1,0],[1,1],[0,1]])
    assert polygon_check(square,square+.5)["overlap"]
    assert not polygon_check(square,square+[2,0])["overlap"]
    np.testing.assert_allclose(polygon_check(square,square+[2,0])["gap_m"],1.)
    assert polygon_check(square,square+[1,0])["overlap"]

    class Process:
        def poll(self):
            return None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/"done.json"
        atomic_json(path,dict(status="ok",value=172))
        assert wait_json(path,Process(),timeout=.1)["value"] == 172
        assert not path.with_suffix(".json.tmp").exists()
        atomic_json(path,dict(status="error",error="expected error"))
        try:
            wait_json(path,Process(),timeout=.1)
        except RuntimeError as exc:
            assert "expected error" in str(exc)
        else:
            raise AssertionError("Worker errors must propagate")

    occupancy = np.zeros((1,5,1,200,200),dtype=np.uint8)
    occupancy[:,:,0,150,150] = 1  # far enough to permit straight travel
    reference = np.c_[np.zeros(6),np.arange(1,7)*.5]
    info = dict(ego_velo=1.,ego_steer=0.,
                model_lidar_frame=dict(virtual_to_ego=c.tolist()))
    result = optimize(reference,occupancy,info,maxiter=30)
    assert result["selected"] is not None
    selected = result["selected"]
    assert selected["minimum_constraint"] >= -1e-5
    np.testing.assert_allclose(np.asarray(selected["lidar"])[5::5],reference,atol=1e-6)

    # Every shipped Python entry point must parse without GPU dependencies.
    for path in Path(__file__).parent.glob("*.py"):
        ast.parse(path.read_text(),filename=str(path))
    print("PASS: pose roundtrip, rotated LiDAR mount, frame165 rollout, native update order")
    print("PASS: polygon collision/gap, IPC errors, feasible constrained MPC, source parsing")
    print("NOT RUN: remote GPU renderer, UniAD model, end-to-end server closed loop")


if __name__ == "__main__":
    main()
