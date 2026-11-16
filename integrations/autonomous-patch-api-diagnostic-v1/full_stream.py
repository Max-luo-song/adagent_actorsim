"""Replay the saved formal model request once, streamed; never apply its output."""

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

import json,os,socket,time,urllib.request,urllib.error,urllib.parse
pass  # Path is provided by the portable boundary above
from probe import ROOT,APP,classify
CAMPAIGN=ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/autonomous_patch_agent_v2/campaigns/patch_auto_v2'

def run():
 c=_hugsim_expand(json.loads((CAMPAIGN/'config.json').read_text()))['api']
 missing=[c[k] for k in ('base_url_env','model_env','api_key_env') if not os.environ.get(c[k])]
 if missing:return {'status':'configuration_missing','missing':missing}
 if not (CAMPAIGN/'STOP').exists():return {'status':'stop_campaign_before_probe'}
 state=_hugsim_expand(json.loads((CAMPAIGN/'state.json').read_text()));r=state.get('active_round') or {};folder=CAMPAIGN/'rounds'/r['id'];records=sorted((folder/'model_calls').glob('*.request.json'))
 saved=_hugsim_expand(json.loads(records[-1].read_text()));messages=saved['messages'];url=os.environ[c['base_url_env']].rstrip('/');key=os.environ[c['api_key_env']]
 parts=urllib.parse.urlparse(url)
 if parts.username or parts.password or parts.query or parts.fragment or parts.scheme!='https':return {'status':'invalid_base_url'}
 if not url.endswith('/chat/completions'):url+='/chat/completions'
 payload={'model':os.environ[c['model_env']],'messages':messages,c['token_parameter']:c['max_output_tokens'],'stream':True}
 if c['response_format']=='json_object':payload['response_format']={'type':'json_object'}
 elif c['response_format']=='json_schema':payload['response_format']={'type':'json_schema','json_schema':{'name':'patch_decision','strict':True,'schema':_hugsim_expand(json.loads((APP/'schemas/decision.schema.json').read_text()))}}
 data=json.dumps(payload,ensure_ascii=False).encode()
 result={'status':'running','mode':'full_saved_request_stream','payload_bytes':len(data),'max_output_tokens':c['max_output_tokens'],'idle_timeout_seconds':180,'total_limit_seconds':600,'campaign_resumed':False,'patch_applied':False,'supplier_usage':None}
 print('Starting one full streamed request; campaign stays stopped.',flush=True)
 class NoRedirect(urllib.request.HTTPRedirectHandler):
  def redirect_request(self,*args,**kwargs):return None
 opener=urllib.request.build_opener(NoRedirect());req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json',c['auth_header']:c['auth_prefix']+key},method='POST')
 start=time.monotonic();content=[];events=0;size=0;last=0
 try:
  with opener.open(req,timeout=180) as response:
   result['http_status']=response.status;result['headers_seconds']=round(time.monotonic()-start,2);print('HTTP response received after %.2fs'%result['headers_seconds'],flush=True)
   result['content_type']=response.headers.get('Content-Type','')
   if 'text/event-stream' not in result['content_type'].lower():
    result['status']='endpoint_did_not_stream';return result
   while True:
    elapsed=time.monotonic()-start
    if elapsed>600:raise TimeoutError('Full diagnostic exceeded 600 seconds')
    line=response.readline(65537);size+=len(line)
    if len(line)>65536 or size>2097152:raise ValueError('Stream exceeded diagnostic limits')
    if not line:break
    if not line.startswith(b'data:'):continue
    text=line[5:].strip()
    if text==b'[DONE]':break
    item=_hugsim_expand(json.loads(text));events+=1
    if events==1:result['first_event_seconds']=round(elapsed,2);print('First stream event after %.2fs'%elapsed,flush=True)
    if isinstance(item.get('usage'),dict):result['supplier_usage']={k:v for k,v in item['usage'].items() if k in ('prompt_tokens','completion_tokens','total_tokens') and isinstance(v,int)}
    if item.get('error'):result['status']='provider_stream_error';break
    for choice in item.get('choices',[]):
     delta=choice.get('delta',{});piece=delta.get('content')
     if isinstance(piece,str):content.append(piece)
     if choice.get('finish_reason'):result['finish_reason']=choice['finish_reason']
    if elapsed-last>=15:print('Receiving stream: %d events, %d content chars, %.0fs'%(events,sum(map(len,content)),elapsed),flush=True);last=elapsed
   else:pass
   if result['status']=='running':result['status']='full_stream_succeeded' if result.get('finish_reason')=='stop' and content else 'incomplete_stream'
 except urllib.error.HTTPError as error:result.update(status='http_error',http_status=error.code)
 except (urllib.error.URLError,TimeoutError,OSError) as error:
  reason=getattr(error,'reason',error);result.update(status=classify(error),error_type=type(error).__name__,reason_type=type(reason).__name__,reason=str(reason).replace(key,'[REDACTED]')[:300])
 except (ValueError,UnicodeError) as error:result.update(status='malformed_stream',error_type=type(error).__name__)
 result.update(seconds=round(time.monotonic()-start,2),stream_events=events,content_characters=sum(map(len,content)))
 if result['status']=='full_stream_succeeded':
  try:
   value=_hugsim_expand(json.loads(''.join(content)));result['decision_json_valid']=isinstance(value,dict);result['decision']=value.get('decision') if isinstance(value,dict) else None
  except ValueError:result['decision_json_valid']=False
 return result

if __name__=='__main__':
 result=run();print(json.dumps(result,indent=2,ensure_ascii=True))
