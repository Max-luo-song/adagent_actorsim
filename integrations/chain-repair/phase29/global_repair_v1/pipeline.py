"""One auditable evidence contract shared by worker, driver and snapshots."""
import copy,hashlib,json,math,pickle
import numpy as np
from .evidence import EvidenceHistory
from .admission import apply
SCHEMA='global_evidence_pipeline_v1'

def signature(array):
    value=np.ascontiguousarray(np.asarray(array,dtype=np.float64))
    if not np.isfinite(value).all():raise ValueError('Nonfinite evidence array')
    return hashlib.sha256(json.dumps(list(value.shape)).encode()+value.tobytes()).hexdigest()

def config_signature(config):return hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False).encode()).hexdigest()

class EvidencePipeline:
    def __init__(self,config,enabled=True):
        self.config=copy.deepcopy(config);self.config_hash=config_signature(config)
        for key in ('max_gap_seconds','confirmation_seconds','confirmation_observations','obstacle_margin_m','caution_speed_mps'):
            if not math.isfinite(float(config[key])) or float(config[key])<=0:raise ValueError('Invalid shared evidence config: '+key)
        self.enabled=bool(enabled);self.observer=EvidenceHistory(config);self.count=0
    def process(self,report,raw_boxes,geometry_boxes,scores,labels,ids,obs,info):
        timestamp=float(info['timestamp']);token=info['scene_token']
        if not isinstance(token,str) or not token or not math.isfinite(timestamp) or timestamp<0:raise ValueError('Invalid frame context')
        if report.get('raw_scores_modified',False):raise ValueError('Upstream neural scores have been modified')
        raw_boxes=np.asarray(raw_boxes);geometry_boxes=np.asarray(geometry_boxes);scores=np.asarray(scores);labels=np.asarray(labels);ids=np.asarray(ids)
        if raw_boxes.ndim!=2 or raw_boxes.shape[1]!=9 or geometry_boxes.shape!=raw_boxes.shape or any(x.shape!=(len(raw_boxes),) for x in (scores,labels,ids)):raise ValueError('Unaligned evidence arrays')
        if not np.issubdtype(ids.dtype,np.integer) or not np.issubdtype(labels.dtype,np.integer):raise ValueError('IDs and labels must be integer vectors')
        if (raw_boxes[:,3:6]<=0).any() or (geometry_boxes[:,3:6]<=0).any():raise ValueError('Invalid evidence box dimensions')
        if (ids==-2).any():raise ValueError('Special ego query leaked into external candidates')
        for value in (raw_boxes,geometry_boxes,scores,labels):signature(value)
        if config_signature(report['config'])!=self.config_hash:raise ValueError('Evidence configuration changed')
        if self.observer.scene==token and self.observer.time is not None:
            delta=timestamp-self.observer.time
            if delta<=0:raise ValueError('Repeated/nonmonotonic frame requires explicit episode reset')
            if delta>self.config['max_gap_seconds']:raise ValueError('Temporal gap requires a complete model/driver episode reset')
        new_episode=self.observer.scene is not None and self.observer.scene!=token
        observer=EvidenceHistory(self.config) if new_episode else copy.deepcopy(self.observer)
        count=0 if new_episode else self.count
        evidence=observer.observe(raw_boxes,ids,labels,obs,info)
        # Untracked fresh queries receive per-frame observation checks only.
        # They may share a contradicted duplicate's geometry status, but never
        # create independent temporal confirmation or enter the ID history.
        fresh=np.flatnonzero((ids<0)&(scores>=self.config.get('candidate_floor',.05)))
        if len(fresh):
            temporary=EvidenceHistory(self.config).observe(raw_boxes[fresh],np.arange(len(fresh)),labels[fresh],obs,info)
            for row in temporary['queries']:
                index=int(fresh[row['index']]);row.update(index=index,neural_id=int(ids[index]),history_observations=0,history_seconds=0.,motion_innovation_m=0.,untrusted_geometry=False)
                evidence['queries'].append(row)
        evidence.update(timestamp=timestamp,scene_token=token)
        # If all duplicates lack observation protection, geometry invalidity
        # belongs to the shared same-frame hypothesis, not only its winning ID.
        # No duplicate creates additional temporal observations.
        evidence=self._share_geometry_status(evidence,np.asarray(raw_boxes),np.asarray(labels))
        output=apply(report,evidence,ids,info) if self.enabled else copy.deepcopy(report)
        self.observer=observer;self.count=count+1
        receipt=dict(schema=SCHEMA,stage='before_driver_memory',episode=token,timestamp=timestamp,frame_count=self.count,
            config_sha256=self.config_hash,geometry_sha256=signature(geometry_boxes),scores_sha256=signature(scores),labels_sha256=signature(labels),
            neural_state_modified=False,raw_scores_modified=False)
        output['global_evidence_receipt']=receipt
        output['global_evidence']=evidence
        return output
    def _share_geometry_status(self,evidence,boxes,labels):
        evidence=copy.deepcopy(evidence);rows=evidence['queries'];parent=list(range(len(rows)))
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        for i,a in enumerate(rows):
            ai=a['index']
            for j in range(i):
                bi=rows[j]['index'];ratio=boxes[ai,3:6]/boxes[bi,3:6]
                if labels[ai]==labels[bi] and (.5<=ratio).all() and (ratio<=2).all() and np.linalg.norm(boxes[ai,:2]-boxes[bi,:2])<min(1.,min(boxes[ai,3:5])/2):parent[find(i)]=find(j)
        groups={}
        for i in range(len(rows)):groups.setdefault(find(i),[]).append(i)
        for group in groups.values():
            protected=any(rows[i]['protected_observation'] for i in group)
            sources=[rows[i]['neural_id'] for i in group if rows[i]['untrusted_geometry']]
            if sources and not protected:
                for i in group:
                    rows[i]['untrusted_geometry']=True;rows[i]['shared_geometry_evidence_sources']=sources
        return evidence
    def reset(self):
        self.observer=EvidenceHistory(self.config);self.count=0
    def capture(self):
        state=dict(schema=SCHEMA,config_sha256=self.config_hash,enabled=self.enabled,count=self.count,observer=copy.deepcopy(self.observer.__dict__))
        blob=pickle.dumps(state,protocol=4)
        return dict(blob=blob,sha256=hashlib.sha256(blob).hexdigest())
    def validate_state(self,package):
        blob=package['blob']
        if hashlib.sha256(blob).hexdigest()!=package['sha256']:raise ValueError('Evidence snapshot hash mismatch')
        state=pickle.loads(blob)
        if state['schema']!=SCHEMA or state['config_sha256']!=self.config_hash or state['enabled']!=self.enabled:raise ValueError('Incompatible evidence snapshot')
        if isinstance(state['count'],bool) or not isinstance(state['count'],int) or state['count']<0:raise ValueError('Invalid evidence frame count')
        observer=state['observer']
        if observer['config']!=self.config or set(observer)!=set(self.observer.__dict__):raise ValueError('Incomplete evidence observer snapshot')
        if observer['records'] and (observer['time'] is None or not observer['scene']):raise ValueError('Evidence snapshot lacks scene/clock context')
        if observer['time'] is not None and not math.isfinite(observer['time']):raise ValueError('Invalid evidence clock')
        for identity,record in observer['records'].items():
            if not isinstance(identity,int) or identity<0:raise ValueError('Invalid evidence identity')
            if record['last_seen']>observer['time'] or observer['time']-record['last_seen']>self.config['max_gap_seconds']+1e-6:raise ValueError('Stale/future evidence record')
            previous=-math.inf
            for timestamp,world,velocity in record['history']:
                if not math.isfinite(timestamp) or timestamp<=previous or np.shape(world)!=(3,) or np.shape(velocity)!=(3,) or not np.isfinite(world).all() or not np.isfinite(velocity).all():raise ValueError('Invalid evidence history')
                previous=timestamp
        return state
    def restore(self,package):
        state=self.validate_state(package)
        observer=EvidenceHistory(self.config);observer.__dict__.update(copy.deepcopy(state['observer']))
        self.observer=observer;self.count=state['count']

def validate_memory_input(report,boxes,scores,labels,info):
    receipt=report.get('global_evidence_receipt')
    if receipt is None or receipt.get('schema')!=SCHEMA or receipt.get('stage')!='before_driver_memory':raise ValueError('Missing pre-memory evidence receipt')
    if receipt['episode']!=info['scene_token'] or receipt['timestamp']!=float(info['timestamp']):raise ValueError('Stale evidence receipt')
    if receipt['config_sha256']!=config_signature(report['config']):raise ValueError('Driver evidence config mismatch')
    for key,value in [('geometry_sha256',boxes),('scores_sha256',scores),('labels_sha256',labels)]:
        if receipt[key]!=signature(value):raise ValueError('Evidence-to-memory array mapping changed: '+key)
    if any(o['candidate_index']<0 for o in report['obstacles']):raise ValueError('Obstacle memory already applied')
    return receipt
