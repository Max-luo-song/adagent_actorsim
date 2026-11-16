"""Bounded streamed model transport; known usage reconciles even rejected responses."""
import json,os,time,urllib.request,urllib.error,urllib.parse
from core import AgentError,Deferred,BudgetExceeded,decode,digest,write,utc
class TransientAPI(Deferred):pass
class CompletedResponse(AgentError):
 def __init__(self,message,metadata):super().__init__(message);self.metadata=metadata
class HTTPProvider:
 def __init__(self,config):self.config=config
 def readiness(self):
  return [self.config[k] for k in ('base_url_env','model_env','api_key_env') if not os.environ.get(self.config[k])]
 def complete(self,messages,schema,before_request,record):
  missing=self.readiness()
  if missing:raise Deferred('API configuration pending: '+', '.join(missing))
  c=self.config;key=os.environ[c['api_key_env']];url=os.environ[c['base_url_env']].rstrip('/');parts=urllib.parse.urlparse(url)
  if parts.username or parts.password or parts.query or parts.fragment:raise AgentError('API base URL must not contain credentials/query/fragment')
  if parts.scheme!='https' and not (parts.scheme=='http' and parts.hostname in ('127.0.0.1','localhost','::1')):raise AgentError('Use HTTPS or loopback HTTP')
  if not url.endswith('/chat/completions'):url+='/chat/completions'
  payload={'model':os.environ[c['model_env']],'messages':messages,c['token_parameter']:c['max_output_tokens'],'stream':bool(c.get('stream',False))}
  if c['response_format']=='json_schema':payload['response_format']={'type':'json_schema','json_schema':{'name':'patch_decision','strict':True,'schema':schema}}
  elif c['response_format']=='json_object':payload['response_format']={'type':'json_object'}
  elif c['response_format']!='text':raise AgentError('Unknown response format')
  data=json.dumps(payload,ensure_ascii=False).encode();reservation=len(data)+512+c['max_output_tokens'];before_request(reservation)
  req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json',c['auth_header']:c['auth_prefix']+key},method='POST')
  class NoRedirect(urllib.request.HTTPRedirectHandler):
   def redirect_request(self,*args,**kwargs):return None
  handlers=[NoRedirect()]
  if parts.hostname in ('127.0.0.1','localhost','::1'):handlers.append(urllib.request.ProxyHandler({}))
  opener=urllib.request.build_opener(*handlers);start=time.monotonic();deadline=start+c['timeout_seconds'];idle=min(c.get('idle_timeout_seconds',c['timeout_seconds']),c['timeout_seconds'])
  progress=record.with_suffix('.progress.json');parts_out=[];usage={};events=0;reasoning_chars=0;finish=None;request_id=None;refusal=False;stream_error=False;last_progress=0;content_chars=0;response=None
  def emit(force=False):
   nonlocal last_progress
   now=time.monotonic()
   if force or now-last_progress>=5:
    write(progress,dict(time=utc(),elapsed_seconds=round(now-start,2),events=events,content_characters=content_chars,reasoning_characters=reasoning_chars,finish_reason=finish,usage_known=isinstance(usage.get('total_tokens'),int),completed=force));last_progress=now
  def metadata():
   tokens=usage.get('total_tokens');known=isinstance(tokens,int) and not isinstance(tokens,bool) and tokens>=0
   fields={k:v for k,v in usage.items() if k in ('prompt_tokens','completion_tokens','total_tokens') and isinstance(v,int) and not isinstance(v,bool) and v>=0}
   return dict(time=utc(),request_id=request_id,usage_tokens=tokens if known else reservation,usage_known=known,usage=fields,reserved_tokens=reservation,model=payload['model'],finish_reason=finish,stream_events=events,reasoning_characters=reasoning_chars,content_characters=content_chars,seconds=round(time.monotonic()-start,2),external_API_called=True)
  def persist(meta,content=None,error=None):
   out=dict(metadata=meta)
   if content is not None:out['content']=content.replace(key,'[REDACTED]')
   if error:out['error']=error
   write(record,out)
  def bounded_readline():
   left=deadline-time.monotonic()
   if left<=0:raise TimeoutError('API total deadline exceeded')
   # urllib HTTPResponse exposes the live socket through its buffered reader.
   sock=getattr(getattr(getattr(response,'fp',None),'raw',None),'_sock',None)
   if sock is not None:sock.settimeout(min(idle,left))
   return response.readline(c['max_response_bytes']+1)
  try:
   emit()
   with opener.open(req,timeout=idle) as response:
    if payload['stream'] and 'text/event-stream' in response.headers.get('Content-Type','').lower():
     size=0;buffer=[]
     def consume(lines):
      nonlocal events,usage,finish,request_id,refusal,reasoning_chars,content_chars,stream_error
      if not lines:return False
      text='\n'.join(lines)
      if text=='[DONE]':return True
      item=decode(text);events+=1;request_id=item.get('id',request_id)
      if isinstance(item.get('usage'),dict):usage=item['usage']
      if item.get('error'):stream_error=True;return True
      for choice in item.get('choices',[]):
       if choice.get('index',0)!=0:continue
       delta=choice.get('delta',{});piece=delta.get('content')
       if isinstance(piece,str):parts_out.append(piece);content_chars+=len(piece)
       for field in ('reasoning_content','reasoning'):
        if isinstance(delta.get(field),str):reasoning_chars+=len(delta[field])
       refusal=refusal or bool(delta.get('refusal'))
       if choice.get('finish_reason'):finish=choice['finish_reason']
      emit();return False
     while True:
      line=bounded_readline();size+=len(line)
      if size>c['max_response_bytes']:raise AgentError('Model response exceeds configured byte limit')
      if not line:
       consume(buffer);break
      text=line.decode('utf-8').rstrip('\r\n')
      if not text:
       done=consume(buffer);buffer=[]
       if done:break
      elif text.startswith('data:'):buffer.append(text[5:].lstrip(' '))
    else:
     raw=response.read(c['max_response_bytes']+1)
     if len(raw)>c['max_response_bytes']:raise AgentError('Model response exceeds configured byte limit')
     envelope=decode(raw.decode('utf-8'));request_id=envelope.get('id');usage=envelope.get('usage') or {};choice=envelope['choices'][0];finish=choice.get('finish_reason');msg=choice['message'];refusal=bool(msg.get('refusal'));content=msg.get('content')
     if isinstance(content,str):parts_out=[content];content_chars=len(content)
     reasoning_chars=len(msg.get('reasoning_content') or '')
   meta=metadata();emit(True);content=''.join(parts_out);persist(meta,content)
   if finish!='stop' or refusal or not content or stream_error:raise CompletedResponse('No complete final JSON; finish_reason=%s, content_characters=%s'%(finish,content_chars),meta)
   try:value=decode(content)
   except (ValueError,AgentError) as error:raise CompletedResponse('Final response is not valid JSON',meta) from error
   return value,meta
  except urllib.error.HTTPError as error:
   write(record,dict(time=utc(),http_status=error.code,reserved_tokens=reservation,external_API_called=True));emit(True)
   if error.code in (401,403):raise Deferred('API authentication rejected; update configured environment variable')
   if error.code==429 or error.code>=500:raise TransientAPI('API temporarily unavailable: HTTP '+str(error.code))
   raise CompletedResponse('API request rejected: HTTP '+str(error.code),dict(usage_tokens=reservation,reserved_tokens=reservation,usage_known=False))
  except (urllib.error.URLError,TimeoutError,OSError) as error:
   reason=getattr(error,'reason',error);safe=str(reason).replace(key,'[REDACTED]')[:300];meta=metadata();persist(meta,error=dict(type=type(error).__name__,reason_type=type(reason).__name__,message=safe));emit(True)
   # A received usage receipt remains authoritative even if transport later broke.
   if meta['usage_known']:raise CompletedResponse('API stream interrupted after reported usage',meta) from error
   raise TransientAPI('API transport unavailable (%s): %s'%(type(reason).__name__,safe)) from error
class FixtureProvider:
 """Test-only; the live CLI never selects it from user configuration."""
 def __init__(self,callback):self.callback=callback;self.calls=0
 def readiness(self):return []
 def complete(self,messages,schema,before_request,record):
  before_request(32);self.calls+=1;value=self.callback(messages,self.calls)
  meta={'usage_tokens':32,'reserved_tokens':32,'external_API_called':False,'mock_model':True}
  write(record,dict(metadata=meta,content=value));return value,meta
