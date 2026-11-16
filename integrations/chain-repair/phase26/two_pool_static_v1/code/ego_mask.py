"""Explicit source-pixel ego exclusion. Missing masks never imply empty ego."""
import numpy as np
from PIL import Image,ImageDraw
from common import read,verify
REVISION='explicit_source_pixel_ego_v1'

def rasterize(polygons,shape_wh,*,empty_reviewed=False):
 w,h=map(int,shape_wh)
 if min(w,h)<=0:raise ValueError('Invalid image dimensions')
 if not polygons and not empty_reviewed:raise ValueError('Empty ego annotation requires explicit reviewed-empty authority')
 im=Image.new('L',(w,h),0);draw=ImageDraw.Draw(im)
 for polygon in polygons:
  points=np.asarray(polygon,float)
  if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all():raise ValueError('Invalid ego polygon')
  if ((points<0).any() or (points[:,0]>w-1).any() or (points[:,1]>h-1).any()):raise ValueError('Ego polygon outside source image')
  draw.polygon([tuple(v) for v in points],fill=255)
 return np.array(im,dtype=np.uint8)

def load_mask(manifest,camera,shape_hw):
 if manifest.get('revision')!=REVISION:raise ValueError('Unregistered ego mask release')
 entry=manifest['cameras'][str(camera)]
 if entry.get('authority') not in ('explicit_raw_image_annotation','dataset_provided_ego_mask'):raise ValueError('Missing ego annotation authority')
 verify(entry['mask']);mask=np.asarray(Image.open(entry['mask']['path']))
 if mask.shape!=tuple(shape_hw) or not np.isin(mask,(0,255)).all():raise ValueError('Mask must use exact original pixel shape and binary encoding')
 if not np.any(mask) and not entry.get('explicit_reviewed_empty',False):raise ValueError('Unreviewed all-zero ego mask')
 return mask>0

def exclude(valid,pixels,mask):
 valid=np.asarray(valid,bool);pixels=np.asarray(pixels);mask=np.asarray(mask,bool)
 if pixels.shape!=(len(valid),2):raise ValueError('Projection shape mismatch')
 if pixels.dtype.kind not in 'iu':raise ValueError('Pixels must be integer source coordinates')
 if (pixels<0).any() or (pixels[:,0]>=mask.shape[1]).any() or (pixels[:,1]>=mask.shape[0]).any():raise ValueError('Out-of-range pixel')
 return valid & ~mask[pixels[:,1],pixels[:,0]]
