"""Versioned pure JSON contracts shared by service and external Agent clients."""
PREVIEWS={"camera":"preview","actor_trajectory":"preview_actor_trajectory","ego_trajectory":"preview_ego_trajectory"}
TASK_SCHEMA={
 '$schema':'https://json-schema.org/draft/2020-12/schema',
 'type':'object','additionalProperties':False,'required':['schema','task_id','scene_id'],
 'properties':{
  'schema':{'const':'scene_task_v1'},'task_id':{'type':'string','pattern':'^[A-Za-z][A-Za-z0-9_-]{0,63}$'},
  'scene_id':{'type':'string','minLength':1},'description':{'type':'string'},
  'weather':{'enum':['clear','rainy'],'default':'clear'},
  'attitude_policy':{'enum':['gravity_level','initial_ego','recorded_up'],'default':'gravity_level'},
  'edits':{'type':'array','maxItems':32,'items':{'type':'object'}},
  'previews':{'type':'array','maxItems':8,'items':{'type':'object','required':['kind','args'],'additionalProperties':False,'properties':{'kind':{'enum':list(PREVIEWS)},'args':{'type':'object'}}}},
  'ego_execution':{'type':'object','required':['driver','trajectory'],'properties':{'driver':{'const':'reference_tracking'},'trajectory':{'type':'object'}}},
  'verification_frames':{'type':'array','maxItems':8,'uniqueItems':True,'items':{'type':'integer','minimum':0}},
  'reset_after':{'type':'boolean','default':True}
 }}
RESULT_SCHEMA={'$schema':'https://json-schema.org/draft/2020-12/schema','type':'object',
 'required':['schema','task_id','status','context','execution','artifacts','acceptance','error'],
 'properties':{'schema':{'const':'scene_task_result_v1'},'task_id':{'type':'string'},'status':{'enum':['completed','failed']},
  'context':{'type':'object'},'execution':{'type':'object'},'artifacts':{'type':'array'},'acceptance':{'type':['object','null']},'error':{'type':['object','null']}}}


# Portable edit vocabulary. Geometric, scene-clock and target-existence checks
# remain authoritative in the existing backend validator.
_VECTOR={'type':'array','minItems':3,'maxItems':3,'items':{'type':'number'}}
_POSE={'type':'array','minItems':4,'maxItems':4,'items':{'type':'array','minItems':4,'maxItems':4,'items':{'type':'number'}}}
_STATE={'enum':['red','yellow','green','off','original']}
_EDIT_FIELDS={
 'translate':{'offset_world_m':_VECTOR},'rotate':{'yaw_degrees':{'type':'number','minimum':-45,'maximum':45}},
 'insert':{'asset_id':{'type':'string'},'placement_id':{'type':'string'},'asset_to_world':_POSE},
 'set_pose':{'placement_id':{'type':'string'},'asset_to_world':_POSE},'remove':{},
 'set_signal':{'state':_STATE},'set_signal_program':{'outside_state':_STATE,'phases':{'type':'array','minItems':1,'maxItems':128,'items':{'type':'object','additionalProperties':False,'required':['start_s','end_s','state'],'properties':{'start_s':{'type':'number','minimum':0},'end_s':{'type':'number','exclusiveMinimum':0},'state':_STATE}}}},
 'restore_signal':{},'set_actor_trajectory':{'trajectory':{'type':'object','description':'actor_trajectory_v1; see ACTOR_TRAJECTORIES.md and ACTOR_MOTION_SMOOTHING.md'}},'restore_actor_trajectory':{}}
_EDIT_SCHEMAS=[]
for _operation,_fields in _EDIT_FIELDS.items():
    _required=['operation','object_id']+list(_fields)
    if _operation in ('insert','set_pose'):_required=[k for k in _required if k not in ('placement_id','asset_to_world')]
    _schema={'type':'object','additionalProperties':False,'required':_required,'properties':{'operation':{'const':_operation},'object_id':{'type':'string'},**_fields}}
    if _operation in ('insert','set_pose'):_schema['oneOf']=[{'required':['placement_id']},{'required':['asset_to_world']}]
    _EDIT_SCHEMAS.append(_schema)
TASK_SCHEMA['properties']['edits']['items']={'oneOf':_EDIT_SCHEMAS}
TASK_SCHEMA['properties']['ego_execution'].update(additionalProperties=False)
TASK_SCHEMA['properties']['ego_execution']['properties'].update({k:{'type':'number'} for k in ('initial_speed_mps','initial_steer_rad','max_tracking_error_m','max_tracking_heading_error_degrees')})
