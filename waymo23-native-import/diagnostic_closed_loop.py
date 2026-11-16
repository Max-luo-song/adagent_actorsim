"""Explicit user-authorized simulation experiment. No N1 certificate is issued."""
import json,os,random
from pathlib import Path
import numpy as np
from frozen_bev_experiment import fingerprint
from edit_trust_adapter import diagnostic_closed_loop
from shared_rules import shared_rules,validate_case_rules
from edit_trust import digest
BASE=Path(__file__).resolve().parents[1]
PLANNING='current'
PROFILE='calibration_geometry_v1'


def run_pair(request_path,output):
    from run_n1_closed_loop import run,implementation_hash
    request=json.loads(Path(request_path).read_text())
    if set(request)!={'experiment_type','case'} or request['experiment_type']!='diagnostic_closed_loop_v1':raise ValueError('Invalid diagnostic request')
    path=Path(request['case']).resolve()
    if BASE not in path.parents:raise ValueError('Case must stay in workspace')
    case=json.loads(path.read_text());validate_case_rules(case,planning_bev=PLANNING,model_profile=PROFILE)
    if not case.get('asset_report') or case.get('edit_trust_state'):raise ValueError('Diagnostic pair requires an uncertified asset, no N1 certificate')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    checkpoint=BASE/'code/UniAD_SIM/ckpts/uniad_base_e2e.pth'
    before=fingerprint(checkpoint);code=implementation_hash()
    report=dict(experiment_type='diagnostic_closed_loop_v1',n1_validation_suspended=True,n1_certified=False,
                formal_evaluation_admitted=False,training=False,planning_bev=PLANNING,model_profile=PROFILE,
                checkpoint_before=before,implementation_before=code,runs={})
    def save():(output/'experiment_report.json').write_text(json.dumps(report,indent=2))
    save();result=0
    try:
        for role in ('baseline','edited'):
            import torch
            random.seed(20261021);np.random.seed(20261021);torch.manual_seed(20261021);torch.cuda.manual_seed_all(20261021)
            os.environ['BEV_EXPERIMENT_SEED']='20261021'
            data=dict(case)
            if role=='baseline':data['asset_report']=None
            target=output/(role+'_case.json');target.write_text(json.dumps(data,indent=2))
            with diagnostic_closed_loop():
                code_result=run(target,output/role,planning_bev=PLANNING,model_profile=PROFILE,diagnostic=True)
            report['runs'][role]=json.loads((output/role/'closed_loop_report.json').read_text());save()
            if code_result:result=1
    except Exception as exc:
        report['error']=repr(exc);result=1
    finally:
        report['checkpoint_after']=fingerprint(checkpoint)
        report['checkpoint_unchanged']=report['checkpoint_after']==before
        report['implementation_unchanged']=implementation_hash()==code
        if not report['checkpoint_unchanged'] or not report['implementation_unchanged']:result=1
        report['execution_completed']=result==0;save()
    return result
