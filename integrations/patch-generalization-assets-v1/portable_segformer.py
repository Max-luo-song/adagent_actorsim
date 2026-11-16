"""Evaluation-only adapter for existing SegFormer inference source and weights.
Backbone, head forward and sliding-window code execute from original AST.
MMCV training/registry wrappers are omitted; weights load strictly, including
unused inherited conv_seg. No training or parameter modification is allowed.
"""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand
Path = _HugsimPath

import ast,copy,math,types,hashlib,sys,time
from functools import partial
pass  # Path is provided by the portable boundary above
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scene_data import OMNI,CLASSES,sha
SOURCE=OMNI/'third_party/SegFormer'

def resize(input,size=None,scale_factor=None,mode='nearest',align_corners=None,warning=True):
 return F.interpolate(input,size=size,scale_factor=scale_factor,mode=mode,align_corners=align_corners)

class EvalDropPath(nn.Identity):
 def __init__(self,drop_prob=0.):super().__init__();self.drop_prob=drop_prob
 def forward(self,x):
  if self.training:raise RuntimeError('Evaluation-only adapter cannot train')
  return x

class EvalConvModule(nn.Module):
 def __init__(self,in_channels,out_channels,kernel_size,norm_cfg=None,**kwargs):
  super().__init__()
  if norm_cfg.get('type')!='SyncBN' or kwargs:raise ValueError('Unexpected original head config')
  self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,bias=False);self.bn=nn.SyncBatchNorm(out_channels);self.activate=nn.ReLU(inplace=True)
 def forward(self,x):
  if self.training:raise RuntimeError('Evaluation-only adapter cannot train')
  return self.activate(self.bn(self.conv(x)))

class EvalBaseHead(nn.Module):
 def __init__(self,in_channels,channels,num_classes,dropout_ratio=.1,in_index=None,input_transform=None,align_corners=False,**kwargs):
  super().__init__();self.in_channels=in_channels;self.channels=channels;self.num_classes=num_classes;self.in_index=in_index;self.input_transform=input_transform;self.align_corners=align_corners;self.conv_seg=nn.Conv2d(channels,num_classes,1);self.dropout=nn.Dropout2d(dropout_ratio)
 def _transform_inputs(self,inputs):
  if self.input_transform!='multiple_select':raise ValueError('Unexpected head input transform')
  return [inputs[i] for i in self.in_index]

def load_original_classes():
 backbone=SOURCE/'mmseg/models/backbones/mix_transformer.py';head=SOURCE/'mmseg/models/decode_heads/segformer_head.py';segmentor=SOURCE/'mmseg/models/segmentors/encoder_decoder.py'
 scope={'__name__':__name__,'torch':torch,'nn':nn,'F':F,'np':np,'math':math,'partial':partial,'DropPath':EvalDropPath,'to_2tuple':lambda x:(x,x) if isinstance(x,int) else x,'trunc_normal_':nn.init.trunc_normal_,'resize':resize,'ConvModule':EvalConvModule,'BaseDecodeHead':EvalBaseHead}
 audit={}
 for path,wanted in [(backbone,{'Mlp','Attention','Block','OverlapPatchEmbed','MixVisionTransformer','DWConv','mit_b5'}),(head,{'MLP','SegFormerHead'})]:
  tree=ast.parse(path.read_text());nodes=[]
  for node in tree.body:
   if isinstance(node,ast.ClassDef) and node.name in wanted:
    original=copy.deepcopy(node);node.decorator_list=[];nodes.append(node)
    # Registration decorators are the sole AST change for model classes.
    original.decorator_list=[]
    if ast.dump(original)!=ast.dump(node):raise ValueError('Model AST changed')
  if {n.name for n in nodes}!=wanted:raise ValueError('Missing original inference class')
  exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),scope);audit[str(path)]=sha(path)
 tree=ast.parse(segmentor.read_text());klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EncoderDecoder');method=copy.deepcopy(next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='slide_inference'));exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(segmentor),'exec'),scope);audit[str(segmentor)]=sha(segmentor)
 return scope,audit

class PortableSegformer(nn.Module):
 def __init__(self,device='cpu'):
  super().__init__();scope,self.source_hashes=load_original_classes();self.backbone=scope['mit_b5']();self.decode_head=scope['SegFormerHead'](feature_strides=[4,8,16,32],in_channels=[64,128,320,512],in_index=[0,1,2,3],channels=128,dropout_ratio=.1,num_classes=19,norm_cfg=dict(type='SyncBN',requires_grad=True),align_corners=False,decoder_params=dict(embed_dim=768))
  self.num_classes=19;self.align_corners=False;self.test_cfg=types.SimpleNamespace(mode='slide',crop_size=(1024,1024),stride=(768,768));self.original_slide=types.MethodType(scope['slide_inference'],self)
  self.weights=SOURCE/'pretrained/segformer.b5.1024x1024.city.160k.pth';checkpoint=torch.load(self.weights,map_location='cpu',weights_only=False);state=checkpoint['state_dict'];self.load_state_dict(state,strict=True);self.weight_key_count=len(state);self.to(device);self.eval();self.device_name=device
  if list(checkpoint['meta']['CLASSES'])!=CLASSES:raise ValueError('Checkpoint taxonomy mismatch')
 def encode_decode(self,x,img_meta):
  return resize(self.decode_head(self.backbone(x)),size=x.shape[2:],mode='bilinear',align_corners=False)
 @torch.inference_mode()
 def predict(self,image,return_confidence=False):
  if self.training:raise RuntimeError('Cannot run training mode')
  from PIL import Image
  with Image.open(image) as im:source=im.convert('RGB');w,h=source.size
  scale=min(2048/max(h,w),1024/min(h,w));new_w=int(w*scale+.5);new_h=int(h*scale+.5)
  resized=source.resize((new_w,new_h),Image.Resampling.BILINEAR)
  rgb=np.asarray(resized,dtype=np.float32).copy();rgb-=np.array([123.675,116.28,103.53],dtype=np.float32);rgb/=np.array([58.395,57.12,57.375],dtype=np.float32);tensor=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).unsqueeze(0).to(self.device_name)
  logits=self.original_slide(tensor,[{'ori_shape':(h,w,3)}],True);labels=logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
  if return_confidence:
   probabilities=logits.softmax(1);top=probabilities.topk(2,dim=1).values[0];confidence=(top[0]*255).round().to(torch.uint8).cpu().numpy();margin=((top[0]-top[1])*255).round().to(torch.uint8).cpu().numpy()
   return labels,confidence,margin
  return labels
 def audit(self):
  return {'backend':'portable_original_ast_pillow_float32_v1','strict_checkpoint_load':True,'weight_keys':self.weight_key_count,'model_sha256':sha(self.weights),'original_source_sha256':self.source_hashes,'source_forward_ast_unchanged':True,'drop_path_disabled_for_evaluation':True,'syncbn_evaluation':True,'preprocessing':'Pillow RGB decode/bilinear resize, original keep-ratio 2048x1024 and RGB normalize; original 1024 crop 768 stride','preprocessing_bitwise_equivalent_to_original_opencv':False,'training_performed':False}
