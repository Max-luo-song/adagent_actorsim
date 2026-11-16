import torch

MODES = ('history','bev_flag_sync','bev_reset',
         'empty_track_queries','new_scene_each_frame')


def apply_policy(model,mode):
    core = getattr(model,'module',model)
    if mode in ('history','new_scene_each_frame'):
        return
    for name in ('prev_bev','prev_frame_info','test_track_instances'):
        if not hasattr(core,name):
            raise ValueError('missing state: '+name)
    if mode=='bev_flag_sync':
        core.prev_frame_info['prev_bev'] = core.prev_bev
    elif mode=='bev_reset':
        core.prev_bev = None
        core.prev_frame_info['prev_bev'] = None
    elif mode=='empty_track_queries':
        with torch.no_grad():
            core.test_track_instances = core._generate_empty_tracks()
    else:
        raise ValueError('unknown replay policy: '+mode)


def instrument(tail):
    marker = 'with torch.no_grad():\n    _results = model'
    if tail.count(marker)!=1:
        raise ValueError('inference-call structure differs')
    hook = (
        "from five_ablation_policy import apply_policy\n"
        "apply_policy(model,_os.environ['ABLATION_MODE'])\n"
    )
    return tail.replace(marker,hook+marker)


def build(source):
    changes = [
        ("for mode in ('history','new_scene_each_frame'):",
         'for mode in '+repr(MODES)+':'),
        ("token = 'replay_history_v1' if mode=='history' else f'replay_fresh_{i}'",
         "token = f'ablation_fresh_{i}' if mode=='new_scene_each_frame' else 'ablation_'+mode\n"
         "            os.environ['ABLATION_MODE'] = mode"),
        ("    code = compile(tail,'recorded_observation_replay','exec')",
         "    from five_ablation_policy import instrument\n"
         "    tail = instrument(tail)\n"
         "    code = compile(tail,'recorded_observation_replay','exec')"),
        ("    (job/'replay_report.json').write_text(json.dumps(report,indent=2))",
         "    report['ablation_comparison'] = [dict(step=i,**{m:rows[i]['last_forward_m'] for m,rows in modes.items()}) for i in range(len(packets))]\n"
         "    report['interventions'] = {'bev_flag_sync':'sync outer BEV-presence flag to actual cached BEV','bev_reset':'clear BEV and flag, retain track instances','empty_track_queries':'regenerate track queries; retain BEV; not a full tracker reset'}\n"
         "    print('ABLATION TABLE:',json.dumps(report['ablation_comparison'],indent=2))\n"
         "    (job/'replay_report.json').write_text(json.dumps(report,indent=2))"),
        ('five_temporal_replay_','five_history_ablation_'),
    ]
    for old,new in changes:
        if source.count(old)!=1:
            raise ValueError('replay source differs: '+old)
        source = source.replace(old,new)
    compile(source,'five_ablation.py','exec')
    return source
