"""Future patches: compare independent episodes under the same frozen scenario."""

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

import argparse,json
pass  # Path is provided by the portable boundary above
FIELDS=('nc','dac','ttc','c','pdms','rc','hdscore')
def compare(baseline,candidate):
 def read(path):return _hugsim_expand(json.loads((Path(path)/'native_eval/native_scores.json').read_text()))
 a,b=read(baseline),read(candidate)
 if a.get('status')!='computed_native_metric_bounds' or b.get('status')!='computed_native_metric_bounds' or not a.get('comparison_key') or a['comparison_key']!=b.get('comparison_key'):raise ValueError('Missing native result or changed scenario/model seed/score definition')
 rows={}
 for key in FIELDS:
  old=a['scores']['adopted_plan'][key];new=b['scores']['adopted_plan'][key];rows[key]=dict(baseline=old,candidate=new,delta_lower=new['lower']-old['upper'],delta_upper=new['upper']-old['lower'])
 complete=a.get('episode_complete') is True and b.get('episode_complete') is True
 return dict(comparable=complete,same_protocol=True,episode_completion_improved=not a.get('episode_complete') and b.get('episode_complete'),complete_episodes=complete,metrics=rows,robust_hdscore_improvement=complete and rows['hdscore']['delta_lower']>0.,conclusion='Robust improvement' if complete and rows['hdscore']['delta_lower']>0. else 'Inconclusive; inspect interval overlap, termination and component metrics',raw_plan_score_is_not_an_independent_baseline=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--candidate',type=Path,required=True);args=parser.parse_args();print(json.dumps(compare(args.baseline,args.candidate),indent=2))
