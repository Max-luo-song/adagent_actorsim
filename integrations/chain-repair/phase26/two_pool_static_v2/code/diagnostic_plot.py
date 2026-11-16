import os
import numpy as np
from shapely.geometry import shape
from common import *
from scan import body_polygon

def draw(scene):
 os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_cache')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 x=read(OUT/'scans'/str(scene)/'scan.json');poly=shape(read(MAPS/str(scene)/'map.json')['geometry']);poses=np.load(ASSETS/str(scene)/'poses_and_clock.npz')['relative_ego_to_sr_world'];selected=x['diagnostics'];fig,axs=plt.subplots(1,len(selected),figsize=(7*len(selected),7),squeeze=False)
 def fill(ax,g,color,alpha,label=None):
  if g.is_empty:return
  parts=list(g.geoms) if hasattr(g,'geoms') else [g]
  for i,part in enumerate(parts):
   if part.geom_type!='Polygon':continue
   xx,yy=part.exterior.xy;ax.fill(xx,yy,color=color,alpha=alpha,label=label if i==0 else None)
   for hole in part.interiors:xx,yy=hole.xy;ax.fill(xx,yy,color='white')
 for ax,item in zip(axs[0],selected):
  frame=item['frame'];body=body_polygon(poses[frame]);outer=body.buffer(.5,quad_segs=32);fill(ax,poly,'green',.25,'Observed admitted road');fill(ax,outer.difference(poly),'red',.65,'Unadmitted footprint + ring');fill(ax,body,'blue',.2,'Ego body');xx,yy=outer.exterior.xy;ax.plot(xx,yy,'k--',label='0.5m margin');ax.plot(poses[:,0,3],poses[:,1,3],':',color='gray',label='Original poses, review only');cx,cy=body.centroid.coords[0];ax.set_xlim(cx-5,cx+5);ax.set_ylim(cy-5,cy+5);ax.set_aspect('equal',adjustable='box');ax.set_title('Scene%d frame%d: blank is unknown/nonroad'%(scene,frame));ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(OUT/'scans'/str(scene)/'footprint_diagnostic.png',dpi=150);plt.close(fig)
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--scene',type=int,required=True);a=p.parse_args();draw(a.scene)
