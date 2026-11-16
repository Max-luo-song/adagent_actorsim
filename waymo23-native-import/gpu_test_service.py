"""Allowlisted GPU test executor over the project's shared filesystem.

Start once in the GPU terminal. Requests contain typed test actions, never shell
commands. It stops after an idle timeout, a STOP file, or the job budget.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

BASE = Path(__file__).resolve().parents[1]
SERVICE = BASE/"outputs/gpu_test_service"


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def workspace_file(value):
    path = Path(value).resolve()
    if BASE not in path.parents or not path.is_file():
        raise ValueError("Test metadata/certificates must be existing files inside this project")
    return str(path)


def build_command(request, output):
    action, args = request["action"], request.get("args", {})
    hugpy = BASE/"code/HUGSIM/.pixi/envs/default/bin/python"
    unipy = BASE/"tools/pixi-uniad-legacy/.pixi/envs/default/bin/python"
    scripts = BASE/"waymo23-native-import"
    if action == "build_source_sequence":
        if set(args)-{"source_report", "start_frame", "frames", "frame_stride", "infer_after_render"}:
            raise ValueError("Unknown sequence test arguments")
        frame, count, stride = int(args["start_frame"]), int(args.get("frames", 3)), int(args.get("frame_stride", 5))
        if frame < 0 or not 1 <= count <= 200 or not 1 <= stride <= 20:
            raise ValueError("Invalid frame budget")
        return [str(hugpy), str(scripts/"build_source_sequence.py"),
                "--source-report", workspace_file(args["source_report"]), "--start-frame", str(frame),
                "--frames", str(count), "--frame-stride", str(stride), "--output", str(output)]
    if action == "infer_sequence":
        if set(args)-{"sequence_report", "max_frames"}:
            raise ValueError("Unknown inference test arguments")
        count = int(args.get("max_frames", 3))
        if not 1 <= count <= 200:
            raise ValueError("Invalid inference budget")
        return [str(unipy), str(scripts/"run_reliable_sequence.py"),
                "--sequence-report", workspace_file(args["sequence_report"]),
                "--max-frames", str(count), "--output", str(output)]
    if action == "sr_pbr_compatibility":
        if set(args) != {"source_report"}:
            raise ValueError("Unknown SR material compatibility arguments")
        return [str(hugpy), str(BASE/"integrations/sr-hugsim/pbr_render_probe.py"),
                "--source-report", workspace_file(args["source_report"]), "--output", str(output)]
    if action == "render_scene":
        if set(args) != {"source_report"}:
            raise ValueError("Unknown render test arguments")
        return [str(hugpy), str(scripts/"gpu_render_smoke.py"),
                "--source-report", workspace_file(args["source_report"]), "--output", str(output)]
    if action == "n1_closed_loop":
        if set(args) != {"case"}:
            raise ValueError("Unknown closed-loop test arguments")
        return [str(hugpy), str(scripts/"run_n1_closed_loop.py"),
                "--case", workspace_file(args["case"]), "--output", str(output)]
    if action == "compare_perception_pair":
        if set(args) != {"pair"}:
            raise ValueError("Unknown paired diagnostic arguments")
        return [str(hugpy), str(scripts/"paired_edit_diagnostic.py"),
                "--pair", workspace_file(args["pair"]), "--output", str(output)]
    if action == 'n1_vehicle_stage_probe_v1':
        if set(request) != {"id", "action", "args"} or args != {"bundle": 'vehicle552-stage-v1'}:
            raise ValueError("Expected exactly the fixed reviewed vehicle evidence bundle")
        import hashlib
        candidate = BASE/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/trusted_edit_eval_round1/vehicle_stage_service_candidate'
        pins = {'common.py': '2f5320479315dfb1f0866a46fb088d852968719328d374292a280d1aeacadf44', 'process_adapter.py': '435e1a34a05a86201231ff0206a0d4329eeceddf209a35f2d2b983664673fd9a', 'worker.py': 'ab0e1690cc80f42183dc9844d1f1efa225612b254fef4864f9bf3ff9c051077f', 'cpu_fixture.py': '49d0aaf9a41d35cc05b965183c6b5b05916b836e6f3adec2e0057f6c535d6be2', 'stage_probe.py': '838c6df9954d86fb4cf08e9e5798a980fbc4538a9fbfdd6b5a4f9d9271cc5bae', 'plan.json': '5b2e70068a0ac3c09d6f6e137f47b8af969f952b21b71437ce72097a321f4c12'}
        for name, expected in pins.items():
            path = candidate/name
            if path.is_symlink() or not path.is_file() or path.resolve() != path.absolute():
                raise ValueError("Missing or redirected reviewed candidate file: "+name)
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Reviewed candidate changed: "+name)
        return [str(hugpy), str(candidate/"stage_probe.py"), "--output", str(output)]
    if action == 'midrun_edit_probe_v1':
        if set(request) != {"id", "action", "args"} or args != {"bundle": 'scene552-midrun-v1'}:
            raise ValueError("Expected exactly the fixed reviewed midrun edit bundle")
        import hashlib
        bundle = BASE/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/experiment_mechanism/midrun_probe'
        pins = {'probe.py': '1ba34c5327115c01f199487b5cd959f5c71e182968f57601439bd149d04cffa9', 'plan.json': '98996d340e5659e028fc1b06189c38dc3acc194dbb561fbc7f94e0fb21818006'}
        for name, expected in pins.items():
            path = bundle/name
            if path.is_symlink() or not path.is_file() or path.resolve() != path.absolute():
                raise ValueError("Missing or redirected reviewed bundle file: "+name)
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Reviewed bundle changed: "+name)
        return [str(hugpy), str(bundle/"probe.py"), "--plan", str(bundle/"plan.json"), "--output", str(output)]
        raise ValueError("Unsupported test action; arbitrary command execution is disabled")


def enqueue(identity, action, args):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", identity):
        raise ValueError("Invalid test identity")
    queue = SERVICE/"requests"
    queue.mkdir(parents=True, exist_ok=True)
    path = queue/(identity+".json")
    if path.exists() or (SERVICE/"jobs"/identity).exists():
        raise ValueError("Test identity already exists")
    request = dict(id=identity, action=action, args=args)
    # Validate before submission, as well as immediately before execution.
    build_command(request, SERVICE/"jobs"/identity/"result")
    atomic_json(path, request)
    return path


def serve(idle_seconds=1800., max_jobs=8, job_timeout=2400.):
    SERVICE.mkdir(parents=True, exist_ok=True)
    lock = (SERVICE/"worker.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("A GPU test worker is already active")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], text=True).strip()
    started, last_work, completed = time.time(), time.monotonic(), 0
    print("GPU test worker:", gpu, "queue:", SERVICE, flush=True)
    heartbeat = dict(pid=os.getpid(), gpu=gpu, started_utc_epoch=started)
    try:
        while completed < max_jobs and time.monotonic()-last_work < idle_seconds:
            if (SERVICE/"STOP").exists():
                break
            pending = sorted((SERVICE/"requests").glob("*.json"))
            atomic_json(SERVICE/"heartbeat.json", dict(heartbeat, status="idle", last_seen_utc_epoch=time.time(), completed_jobs=completed))
            if not pending:
                time.sleep(2.)
                continue
            queued = pending[0]
            request = json.loads(queued.read_text())
            identity = request["id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", identity) or queued.stem != identity:
                raise ValueError("Invalid queued identity")
            folder = SERVICE/"jobs"/identity
            folder.mkdir(parents=True, exist_ok=False)
            queued.replace(folder/"request.json")
            status = dict(id=identity, action=request["action"], started_utc_epoch=time.time(), status="running")
            atomic_json(folder/"status.json", status)
            atomic_json(SERVICE/"heartbeat.json", dict(heartbeat, status="running", job=identity, last_seen_utc_epoch=time.time()))
            process = None
            try:
                command = build_command(request, folder/"result")
                env = dict(os.environ, BASE=str(BASE), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8",
                           MKL_NUM_THREADS="8", PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python",
                           MPLCONFIGDIR=str(folder/"matplotlib"))
                for key in ("LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME"):
                    env.pop(key, None)
                with (folder/"run.log").open("w") as log:
                    process = subprocess.Popen(command, cwd=BASE, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    began = time.monotonic()
                    while process.poll() is None:
                        if time.monotonic()-began > job_timeout:
                            raise TimeoutError("GPU test exceeded its fixed runtime budget")
                        atomic_json(SERVICE/"heartbeat.json", dict(heartbeat, status="running", job=identity,
                                    last_seen_utc_epoch=time.time(), elapsed_seconds=time.monotonic()-began))
                        time.sleep(5.)
                status.update(status="passed" if process.returncode == 0 else "failed", exit_code=process.returncode)
                if process.returncode == 0 and request["action"] == "build_source_sequence" and request["args"].get("infer_after_render"):
                    enqueue(identity+"-infer", "infer_sequence", dict(sequence_report=str(folder/"result/sequence_report.json"),
                                                                    max_frames=int(request["args"].get("frames", 3))))
            except BaseException as exc:
                status.update(status="error", error=repr(exc))
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10.)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
            finally:
                status["finished_utc_epoch"] = time.time()
                atomic_json(folder/"status.json", status)
                print(identity, status["status"], "log:", folder/"run.log", flush=True)
                completed += 1
                last_work = time.monotonic()
    finally:
        atomic_json(SERVICE/"heartbeat.json", dict(heartbeat, status="stopped", last_seen_utc_epoch=time.time(), completed_jobs=completed))
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idle-seconds", type=float, default=1800.)
    parser.add_argument("--max-jobs", type=int, default=8)
    parser.add_argument("--job-timeout", type=float, default=2400.)
    args = parser.parse_args()
    if args.idle_seconds <= 0 or not 1 <= args.max_jobs <= 50 or args.job_timeout <= 0:
        parser.error("Invalid worker runtime budget")
    serve(args.idle_seconds, args.max_jobs, args.job_timeout)
