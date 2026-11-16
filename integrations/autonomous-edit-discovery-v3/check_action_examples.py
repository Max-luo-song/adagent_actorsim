"""Actual CPU compilation/audit against frozen scene assets; never runs controls."""
import copy,sys,time
from common_flow import *
from action_program import PROGRAM,validate_phases

def run():
 sys.path.insert(0,str(DISCOVERY));from episode import bind
 policy=read(P26/'global_patch.json')['policy'];bind(policy)
 import action_scenario
 from feasibility import check
 def p(action,duration,speed,lane='keep'):return dict(action=action,duration_s=duration,target_speed_ratio=speed,lane=lane)
 examples=dict(merge_brake_accelerate=[p('lane_change',3.2,1.,'ego_lane'),p('decelerate',4.5,.3),p('accelerate',4.5,1.)],
  merge_stop_restart=[p('lane_change',3.2,1.,'ego_lane'),p('decelerate',6.,0.),p('stop_hold',1.5,0.),p('accelerate',5.,.8)],
  merge_yield_return=[p('decelerate',3.2,.6,'ego_lane'),p('cruise',1.5,.6),p('accelerate',3.2,1.,'original_lane')])
 results=[];baseline=Path(policy['matrix'])/'cases/original551/on';root=OUT/'checks/action_examples'/('attempt_'+str(time.time_ns()));root.mkdir(parents=True,exist_ok=False)
 for name,phases in examples.items():
  folder=root/name;original=copy.deepcopy(action_scenario.CONFIG);config=copy.deepcopy(original)
  for k in ('lane_change_duration_s','actor_deceleration_duration_s','target_lane_center_source_y_m','final_actor_speed_mps'):config.pop(k,None)
  config.update(phases=validate_phases(phases),action_program=PROGRAM,interpretation='multi-stage host action example, not a measured failure',direction='declared lane intents')
  action_scenario.CONFIG=config
  try:action_scenario.generate(folder,baseline)
  except ValueError:
   if not (folder/'scenario.json').exists():raise
  finally:action_scenario.CONFIG=original
  audit=read(folder/'scenario.json');task=dict(edit_assets=str(folder),edit_receipts={n:asset(folder/n) for n in ('scenario.json','target_track.npz')});write(folder/'task.json',task)
  feasible=check(folder/'task.json',policy['metrics']['cutin551']['report']['path']);write(folder/'feasibility.json',feasible)
  result=dict(name=name,phases=phases,passed=audit['passed'],checks=audit['checks'],motion=audit['metrics'],other_actor_gap_m=audit['minimum_other_actor_gap_m'],offline_feasible_path=feasible['passed'],witness_gap_m=feasible['minimum_edited_actor_gap_m'],track=asset(folder/'target_track.npz'));results.append(result)
  print(json.dumps(result,ensure_ascii=False),flush=True)
 report=dict(schema='multistage_actual_CPU_audit_v1',global_default='v15',results=results,API_called=False,GPU_executed=False,model_authored=False,failed_scene_claim=False)
 write(root/'report.json',report);write(OUT/'checks/action_examples/latest.json',asset(root/'report.json'))
 if not all(x['passed'] and x['offline_feasible_path'] for x in results):raise ValueError('An example needs repair or rejection before GPU')
 return report
if __name__=='__main__':run()
