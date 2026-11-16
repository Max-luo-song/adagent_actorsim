import argparse,os
from common import *
def main(scene):
 os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_cache')
 expected=(OUT/'preregistration.sha256').read_text().split()[0]
 if sha(OUT/'preregistration.json')!=expected:raise ValueError('Changed map/scan preregistration')
 frozen=read(OUT/'core_registration_frozen.json')
 for entry in frozen['scenes'].values():verify(entry)
 import boundaries
 result=boundaries.build(scene)
 print('MAP COMPLETE',scene,'support',result['support'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scene',type=int,required=True);a=p.parse_args();main(a.scene)
