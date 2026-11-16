"""Three-camera native render smoke test; never issues an N1 certificate."""
import argparse
import json
import os
from pathlib import Path
import sys
import time



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1]
    os.environ["BASE"] = str(base)
    source = Path(args.source_report).resolve()
    output = Path(args.output).resolve()
    startup_probe = json.loads(source.read_text()).get("uniad_startup_probe")
    if startup_probe is not None:
        startup_probe = Path(startup_probe).resolve()
        if base not in startup_probe.parents or not startup_probe.is_file():
            raise ValueError("Startup probe must be a workspace JSON file")
        from uniad_startup_probe import run
        run(startup_probe,output)
        return
    uniad_probe = json.loads(source.read_text()).get("unified_uniad_probe")
    if uniad_probe is not None:
        uniad_probe = Path(uniad_probe).resolve()
        if base not in uniad_probe.parents or not uniad_probe.is_file():
            raise ValueError("Unified UniAD probe must be a workspace JSON file")
        sys.path.insert(0, str(base/"integrations/sr-hugsim"))
        from unified_uniad_probe import run
        run(uniad_probe, output)
        return
    depth_probe = json.loads(source.read_text()).get("static_depth_probe")
    if depth_probe is not None:
        depth_probe = Path(depth_probe).resolve()
        if base not in depth_probe.parents or not depth_probe.is_file():
            raise ValueError("Static depth request must be a workspace file")
        sys.path.insert(0, str(base/"integrations/sr-hugsim"))
        from static_depth_probe import run
        run(depth_probe, output)
        return
    road_probe = json.loads(source.read_text()).get("road_constraint_preflight")
    if road_probe is not None:
        road_probe = Path(road_probe).resolve()
        if base not in road_probe.parents or not road_probe.is_file():
            raise ValueError("Road preflight must be a workspace JSON file")
        sys.path.insert(0, str(base/"integrations/sr-hugsim"))
        from road_constraint_preflight import run
        run(road_probe, output)
        return
    comparison = json.loads(source.read_text()).get("unified_open_closed_probe")
    if comparison is not None:
        comparison = Path(comparison).resolve()
        if base not in comparison.parents or not comparison.is_file():
            raise ValueError("Open/closed comparison must be a workspace JSON file")
        sys.path.insert(0, str(base/"integrations/sr-hugsim"))
        from unified_open_closed_probe import run
        run(comparison, output)
        return
    probe = json.loads(source.read_text()).get("unified_interface_probe")
    if probe is not None:
        probe = Path(probe).resolve()
        if base not in probe.parents or not probe.is_file():
            raise ValueError("Unified-interface probe must be a workspace JSON file")
        sys.path.insert(0, str(base/"integrations/sr-hugsim"))
        from unified_smoke import run
        run(probe, output)
        return
    output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not visible to the HUGSIM Python environment")
    sys.path[:0] = [str(base/"code/HUGSIM"), str(base/"code/HUGSIM/sim")]
    # Reuse the existing source-node adapter, driven by the supplied metadata.
    import waymo552_three as adapter
    adapter.SOURCE_REPORT = str(source)
    began = time.monotonic()
    env = None
    try:
        env = adapter.make_env(scale=1.)
        env.capture_edit_evidence = True
        torch.cuda.synchronize()
        obs = env.observe()
        torch.cuda.synchronize()
        from PIL import Image
        arrays, cameras = {}, {}
        for name, rgb in obs["rgb"].items():
            depth = obs["depth"][name]
            evidence = env._edit_validation_views[name]
            alpha = evidence["alpha"]
            if not rgb.any() or not np.isfinite(depth).all() or not (depth > 0).any():
                raise RuntimeError("Empty/nonfinite native render: "+name)
            if not np.isfinite(alpha).all() or not ((alpha >= 0) & (alpha <= 1)).all():
                raise RuntimeError("Invalid renderer alpha: "+name)
            Image.fromarray(rgb).save(output/(name+".png"))
            for key, value in dict(rgb=rgb, depth=depth, alpha=alpha,
                                   K=evidence["K"], camera_to_world=evidence["camera_to_world"]).items():
                arrays[name+"__"+key] = value
            cameras[name] = dict(shape=list(rgb.shape), nonblack_fraction=float((rgb.max(-1)>0).mean()),
                                 positive_depth_fraction=float((depth>0).mean()),
                                 opaque_fraction=float((alpha>=.8).mean()))
        if len(cameras) != 3:
            raise RuntimeError("Expected the three calibrated source cameras")
        np.savez_compressed(output/"render_evidence.npz", **arrays)
        report = dict(status="native_three_camera_smoke_passed", source_report=str(source),
                      gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                      wall_seconds=time.monotonic()-began, cameras=cameras,
                      checkpoint_loads=env.checkpoint_loads,
                      n1_independent_references_supplied=False, n1_edit_certified=False,
                      closed_loop_run=False)
        (output/"render_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
