"""Standard-library-only startup telemetry, independent of numerical packages."""
import json
import os
from pathlib import Path
import time

_STARTED=time.monotonic()
_EVENTS=[]
_REPORT=os.environ.get('UNIAD_STARTUP_REPORT')


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)


def configure(path):
    global _REPORT
    _REPORT=str(path)
    _flush()


def _flush():
    if _REPORT:
        atomic_json(_REPORT,dict(schema='uniad_startup_timing_v1',pid=os.getpid(),events=_EVENTS))


def mark(stage,**details):
    now=time.monotonic()-_STARTED
    event=dict(stage=stage,elapsed_seconds=now,
               delta_seconds=now-(_EVENTS[-1]['elapsed_seconds'] if _EVENTS else 0.),
               utc_epoch=time.time(),**details)
    _EVENTS.append(event)
    _flush()
    if _REPORT:print('UniAD startup: %s %.3fs (+%.3fs)'%(stage,now,event['delta_seconds']),flush=True)
    return event
