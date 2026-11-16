"""Scene-independent admission rules; scene cases carry data, never thresholds."""
from dataclasses import asdict

from n1_cross_scene_audit import CrossSceneProtocol
from edit_trust import EditPolicy, EditTrustError, digest
from scene_reliability import ReliabilityConfig
from runtime_configuration import load_vehicle_config, model_input_profile, configured_reliability


def shared_rules(planning_bev="history", model_profile="legacy"):
    if planning_bev not in ("history", "current"):
        raise ValueError("Unsupported global planning BEV policy")
    # The implementation hash separately binds procedural MPC/adapter rules.
    protocol = asdict(CrossSceneProtocol())
    protocol["planning_bev"] = planning_bev
    model_input = model_input_profile()
    if model_profile != "legacy":
        from model_profiles import resolve_model_profile
        _, model_input = resolve_model_profile(dict(model_input_profile=model_profile))
    model_input["planning_bev"] = planning_bev
    return dict(schema_version=1, protocol=protocol,
                edit_policy=asdict(EditPolicy()),
                reliability=asdict(configured_reliability()),
                model_input=model_input,
                vehicle=asdict(load_vehicle_config()))


def validate_case_rules(case, domain=None, *, planning_bev="history", model_profile="legacy"):
    allowed = {"source_report", "protocol_hash", "asset_report", "edit_trust_state",
               "scene_edits", "rules_hash"}
    unexpected = set(case)-allowed
    if unexpected:
        raise ValueError("Scene case contains unsupported fields/overrides: "+str(sorted(unexpected)))
    rules = shared_rules(planning_bev, model_profile)
    if case.get("protocol_hash") != digest(rules["protocol"]):
        raise ValueError("Case does not use the locked shared cross-scene protocol")
    rules_hash = digest(rules)
    if "rules_hash" in case and case["rules_hash"] != rules_hash:
        raise ValueError("Case does not use the shared edit/reliability rules")
    if domain is not None and asdict(domain.policy) != rules["edit_policy"]:
        raise EditTrustError("N1 domain policy differs from shared edit policy")
    return dict(rules=rules, rules_hash=rules_hash,
                case_rules_hash_explicit="rules_hash" in case)
