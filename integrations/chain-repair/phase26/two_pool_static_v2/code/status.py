import os
from common import *
x=read(OUT/'progress.json') if (OUT/'progress.json').exists() else {'stage':'not_started'}
print('Stage:',x['stage'],'| Device:',x.get('device'), '| Scene:',x.get('scene','-'))
files=[OUT/'progress.json']+[ASSETS/str(s)/'semantic19/progress.json' for s in SCENES];latest=max((p.stat().st_mtime for p in files if p.exists()),default=0)
print('Last progress write:',round(__import__('time').time()-latest,1),'seconds ago (process IDs may be in another namespace)')
for s in SCENES:
 p=ASSETS/str(s)/'semantic19/progress.json';m=ASSETS/str(s)/'semantic19/semantic_manifest.json';scan=OUT/'scans'/str(s)/'scan.json'
 if scan.exists():
  d=read(scan);print('Scene',s,'scan:',d['status'],'|',d['reason'])
 elif m.exists():print('Scene',s,'semantics complete; map pending')
 elif p.exists():
  d=read(p);print('Scene',s,'semantic images:',str(d['completed'])+'/'+str(d['total']))
 else:print('Scene',s,'queued')
if x.get('error'):print('Error:',x['error'])
