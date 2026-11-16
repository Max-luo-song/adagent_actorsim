"""Explicit original Framework isolated dependencies, without global installs."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

import argparse,importlib,json,os,shutil,subprocess,sys
from paths import *
ENV_FIX=PHASE26/'bridge/env_fix'
def configure():
 source=ENV_FIX/'nvdiffrast_build_src';libs=sorted((source/'build').glob('lib.linux-*'));meta=ENV_FIX/'nvdiffrast_meta'
 if len(libs)!=1 or not (source/'nvdiffrast/torch/__init__.py').is_file() or not list(libs[0].glob('_nvdiffrast_c*.so')):raise RuntimeError('Original isolated nvdiffrast build missing; no automatic package substitution')
 for path in (source,libs[0],meta):
  if str(path) not in sys.path:sys.path.insert(0,str(path))
 os.environ['SMPL_NEUTRAL_NPZ']=str(PHASE26/'bridge/smpl_tools/resources/smpl_neutral/SMPL_NEUTRAL.npz')
 os.environ['CUDA_HOME']='/usr/local/cuda-12.4'
 os.environ['PATH']='/usr/local/cuda-12.4/bin:'+os.environ.get('PATH','')
 os.environ['TORCH_EXTENSIONS_DIR']=str(ENV_FIX/'torch_ext_isolated')
 return [source,libs[0],meta]
def ffmpeg_executable():
 # Reuse imageio's existing bundled binary; video must not depend on login PATH.
 import imageio_ffmpeg
 candidates=sorted((Path(imageio_ffmpeg.__file__).parent/'binaries').glob('ffmpeg-linux-x86_64-*'))
 if len(candidates)!=1 or not os.access(candidates[0],os.X_OK):raise RuntimeError('Bundled ffmpeg executable unavailable')
 return str(candidates[0])
def imports():
 configure()
 print("Checking Framework compiled dependencies...",flush=True)
 import torch
 import nvdiffrast.torch as dr
 import _nvdiffrast_c
 # Match all imports delayed inside FrameworkSceneBackend.__init__, plus the
 # exact trainer type selected by the real scene checkpoint configuration.
 framework=PHASE26/'framework_repo'
 for folder in (framework,ROOT/'code/HUGSIM'):
  if str(folder) not in sys.path:sys.path.insert(0,str(folder))
 from datasets.driving_dataset_novel_view import DrivingDatasetNovelView
 from omegaconf import OmegaConf
 from utils.misc import import_str
 source=_hugsim_expand(json.loads((SCENE_BUNDLE/'source_manifest.json').read_text()));cfg=OmegaConf.load(Path(source['source_checkpoint']).parent/'config.yaml');trainer=import_str(cfg.trainer.type)
 if not callable(trainer):raise RuntimeError('Invalid actual Framework trainer type')
 configured=[]
 def check_types(value):
  if isinstance(value,dict):
   target=value.get('type')
   if isinstance(target,str) and target.startswith(('models.','datasets.')):import_str(target);configured.append(target)
   for item in value.values():check_types(item)
  elif isinstance(value,list):
   for item in value:check_types(item)
 check_types(OmegaConf.to_container(cfg,resolve=False))
 print('Actual dataset, trainer and configured model classes imported',flush=True)
 if not Path(os.environ['SMPL_NEUTRAL_NPZ']).is_file():raise FileNotFoundError('Original SMPL resource missing')
 subprocess.run([ffmpeg_executable(),'-version'],check=True,timeout=15,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 for name in ('flock','timeout','tee'):
  if shutil.which(name) is None:raise RuntimeError('Required executable unavailable: '+name)
 return dict(nvdiffrast=str(Path(dr.__file__).resolve()),extension=str(Path(_nvdiffrast_c.__file__).resolve()),dataset=DrivingDatasetNovelView.__module__,trainer=cfg.trainer.type,configured_types=configured,torch=torch.__version__,native_dependency_version_note='Reuse existing official nvdiffrast 0.4.0 diagnostic build; original Framework pin is diffrp-nvdiffrast 0.3.3.1')
def gpu_idle(executable=None):
 command=executable or shutil.which('nvidia-smi')
 if not command:raise RuntimeError('nvidia-smi unavailable; no simulation launched')
 result=subprocess.run([str(command),'--query-compute-apps=pid','--format=csv,noheader,nounits'],check=True,timeout=15,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 lines=[line.strip() for line in result.stdout.splitlines() if line.strip()]
 if any(line.isdecimal() for line in lines):raise RuntimeError('GPU compute process active; no simulation launched')
 if lines and lines!=['No running processes found']:raise RuntimeError('Unrecognized GPU process query; no simulation launched: '+repr(lines))
def gpu_probe():
 import torch
 import nvdiffrast.torch as dr
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no simulation launched')
 # Actual operation used by Framework EnvLight; validates the compiled plugin.
 image=torch.ones((1,6,2,2,3),device='cuda');uv=torch.tensor([[[[0.,0.,1.]]]],device='cuda');sample=dr.texture(image,uv,filter_mode='linear',boundary_mode='cube');torch.cuda.synchronize()
 if sample.shape!=(1,1,1,3) or not torch.isfinite(sample).all() or not torch.allclose(sample,torch.ones_like(sample)):raise RuntimeError('nvdiffrast GPU texture probe failed')
 from gsplat import rasterization
 means=torch.tensor([[0.,0.,2.]],device='cuda');quats=torch.tensor([[1.,0.,0.,0.]],device='cuda');scales=torch.full((1,3),.1,device='cuda');opacities=torch.full((1,),.9,device='cuda');colors=torch.tensor([[1.,.2,.1]],device='cuda');view=torch.eye(4,device='cuda')[None];K=torch.tensor([[[10.,0.,8.],[0.,10.,8.],[0.,0.,1.]]],device='cuda')
 rendered,alpha,meta=rasterization(means,quats,scales,opacities,colors,view,K,16,16,render_mode='RGB+ED');torch.cuda.synchronize()
 if rendered.shape!=(1,16,16,4) or alpha.shape!=(1,16,16,1) or not torch.isfinite(rendered).all() or not bool((alpha>0).any()):raise RuntimeError('gsplat RGB+ED GPU probe failed')
 del image,uv,sample,means,quats,scales,opacities,colors,view,K,rendered,alpha,meta
 torch.cuda.empty_cache()
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true');parser.add_argument('--idle',action='store_true');args=parser.parse_args()
 if args.idle:gpu_idle();print('GPU process query succeeded; idle')
 else:
  evidence=imports()
  if args.gpu:gpu_probe();evidence['gpu_texture_probe_passed']=True
  print(json.dumps(evidence,indent=2))
