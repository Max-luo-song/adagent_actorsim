"""Drain the previous workflow and preserve all budget history before design API calls."""
import copy,fcntl,os,subprocess,sys,time
from common_flow import *
OLD=P26/'autonomous_edit_discovery_v2'


def acquire_previous(status,stop):
    OLD.mkdir(parents=True,exist_ok=True);lock=(OLD/'worker.lock').open('a')
    while True:
        if stop():raise RuntimeError('Operator stop while waiting for previous workflow')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);return lock
        except BlockingIOError:
            status('waiting_previous_workflow',previous='v15 discoveryv2; current episode/analysis is retained')
            time.sleep(15)


def settle_pending(status):
    # New user-directed multi-stage discovery does not run the old offset task first.
    # Preserve its exact checkpoint and inherit usage below; no historical state is edited.
    f=OLD/'handoff.json'
    if not f.exists():return
    root=Path(read(f)['campaign']);state=read(root/'state.json')
    if state.get('active_round'):
        write(OUT/'prior_patch_checkpoint.json',dict(campaign=str(root),state=asset(root/'state.json'),registry=asset(root/'candidate_registry.json') if (root/'candidate_registry.json').exists() else None,status=state['status'],active_round=state['active_round'].get('id'),disposition='retained pending; new multi-stage discovery takes priority by current user request',historical_state_modified=False,counters_reset=False))
        status('prior_patch_retained_for_new_action_discovery',campaign=str(root))


def initialize(host_pins,options):
    if (OUT/'mission.json').exists():
        m=read(OUT/'mission.json');verify_pins(m['host_pins']);verify(m['global_patch']);return m
    previous=read(OLD/'mission.json');ledger=read(OLD/'ledger.json');source=read(OLD/'source_state.json')
    handoff=OLD/'handoff.json';prior=None
    if handoff.exists():
        campaign=Path(read(handoff)['campaign']);prior=asset(campaign/'state.json');latest=read(campaign/'state.json')
        source['usage']=copy.deepcopy(latest['usage']);source['rounds_started']=latest['rounds_started'];source['history']=source.get('history',[])+latest.get('history',[]);source['prior_working_candidate']=latest.get('champion')
    else:
        source['usage']['gpu_attempts']=source['usage'].get('gpu_attempts',0)+ledger['gpu_attempts'];source['usage']['gpu_dispatches']=source['usage'].get('gpu_dispatches',0)+ledger['gpu_attempts']
    global_patch=read(P26/'global_patch.json');selected_global_policy()
    source['champion']=copy.deepcopy(global_patch['policy']);source['champion']['reference_metrics']=copy.deepcopy(source['champion']['metrics'])
    OUT.mkdir(parents=True,exist_ok=True)
    for name,value in [('source_state.json',source),('source_registry.json',read(OLD/'source_registry.json')),('source_config.json',read(OLD/'source_config.json')),('global_v15.json',source['champion'])]:write(OUT/name,value)
    if not (OUT/'global_parent_snapshot.json').exists():(OUT/'global_parent_snapshot.json').write_bytes((P26/'global_patch.json').read_bytes())
    allocation=copy.deepcopy(previous['explicit_budget_extension'])
    for key,arg in [('wall_seconds','add_wall_seconds'),('gpu_attempts','add_gpu_attempts')]:
        delta=options.get(arg) or 0
        if type(delta) is not int or delta<0:raise ValueError('Supplemental allocation must be nonnegative')
        allocation[key]+=delta
    m=dict(schema='llm_scene_design_to_patch_v3',host_pins=host_pins,policies={'global':str(OUT/'global_v15.json'),'stable':str(OUT/'global_v15.json')},
        explicit_budget_extension=allocation,allocation_event=dict(additions=options,previous_allocation=previous['explicit_budget_extension'],no_timer_or_counter_reset=True),
        source_state=asset(OUT/'source_state.json'),source_registry=asset(OUT/'source_registry.json'),global_patch=asset(OUT/'global_parent_snapshot.json'),new_edits=[],scene_action_cycle=dict(enabled=True,validation_scope='counterexample_cross_scene',different_scene_after_confirmation=True),scenario_design=dict(enabled=True,max_batches=2,max_requests_per_batch=3,max_edits_per_batch=4),
        discovery_gpu_carried=ledger['gpu_attempts'],budget_lineage=dict(previous_ledger=asset(OLD/'ledger.json'),previous_agent=prior,counters_reset=False),baseline='fixed global v15; model authors scene proposals; host measures all outcomes',automatic_handoff=True,global_patch_validated=False)
    write(OUT/'ledger.json',ledger);write(OUT/'mission.json',m);return m


def merge_design_usage(base,usage):
    result=copy.deepcopy(base)
    for k in ('api_requests','charged_tokens','reported_tokens','prompt_tokens','completion_tokens'):result[k]=result.get(k,0)+usage.get(k,0)
    return result
