"""Enforce worker-to-memory ordering without weakening existing geometry."""
from .pipeline import validate_memory_input

def memory_class(original):
    class ContractObstacleMemory(original):
        def update(self,report,boxes,scores,labels,info,current_occupancy=None):
            receipt=validate_memory_input(report,boxes,scores,labels,info)
            result=super().update(report,boxes,scores,labels,info,current_occupancy=current_occupancy)
            result['global_evidence_receipt']=dict(receipt,stage='after_driver_memory')
            return result
    ContractObstacleMemory.__name__='ContractObstacleMemory'
    return ContractObstacleMemory

# Stable module-level class: driver snapshots must be pickle-restorable.
from planning_guard import ObstacleMemory as BaseObstacleMemory
class ContractObstacleMemory(BaseObstacleMemory):
    def update(self,report,boxes,scores,labels,info,current_occupancy=None):
        receipt=validate_memory_input(report,boxes,scores,labels,info)
        result=super().update(report,boxes,scores,labels,info,current_occupancy=current_occupancy)
        result['global_evidence_receipt']=dict(receipt,stage='after_driver_memory')
        return result

def validate_guard_input(report,info):
    receipt=report.get('global_evidence_receipt') or {}
    if receipt.get('stage')!='after_driver_memory' or receipt.get('episode')!=info['scene_token'] or receipt.get('timestamp')!=float(info['timestamp']):
        raise ValueError('Guard input did not traverse the current evidence/memory contract')
