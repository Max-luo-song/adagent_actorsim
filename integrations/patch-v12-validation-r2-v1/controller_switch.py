"""Switch the complete external patch only. Original NN/iLQR stay shared."""
import ast
import numpy as np
from joint_safety import JointReference as FrozenPatch
MODE=None
class NativeReference(FrozenPatch):
    def process(self,raw,arrays,obs,info,motion):
        raw=np.asarray(raw,float)
        if raw.shape!=(6,2) or not np.isfinite(raw).all():raise ValueError('Invalid native reference')
        self.observed(obs,info,arrays);self.selected_states=None;self.selected=None
        return raw.copy(),dict(source='native_uniad_with_shared_ilqr',mode='full_external_patch_off',reference_changed=False,
            reference_certificate=self.certificate(self.states(raw)),audit_only=True)
    def admit(self,motion,nominal):
        return nominal.copy(),dict(selected='native_tracker_patch_off',override=False,external_patch_executed=False,
            common_environment_collision_and_map_stop_retained=True)
def JointReference(vehicle,mapping):
    if MODE=='on':return FrozenPatch(vehicle,mapping)
    if MODE=='off':return NativeReference(vehicle,mapping)
    raise ValueError('Explicit off/on selection required')
def rewrite_runtime(source):
    tree=ast.parse(source);found=0
    for n in ast.walk(tree):
        if isinstance(n,ast.ImportFrom) and n.module=='joint_safety' and [(a.name,a.asname) for a in n.names]==[('JointReference',None)]:
            n.module='controller_switch';found+=1
    if found!=1:raise ValueError('Frozen runtime controller import contract changed')
    return tree
