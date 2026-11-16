"""Fail-closed contracts for edit admission and counterfactual experiments.

The model may propose JSON, but it never supplies trust measurements or grants
itself authority to execute an edit. This module intentionally has no CUDA or
LLM dependency.
"""
from __future__ import annotations

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand


from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _exact(value: dict, names: set[str], what: str) -> None:
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError(f"{what}: expected exactly {sorted(names)}")


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{what}: expected finite number")
    return float(value)


@dataclass(frozen=True)
class EditCommand:
    operation: str
    target_id: str
    arguments: dict

    @classmethod
    def parse(cls, value: dict) -> "EditCommand":
        _exact(value, {"operation", "target_id", "arguments"}, "edit command")
        op, target, args = value["operation"], value["target_id"], value["arguments"]
        signatures = {
            "insert_vehicle": {"asset_id", "x_m", "y_m", "yaw_rad"},
            "remove_vehicle": set(),
            "modify_lane_marking": {"marking"},
            "add_occluder": {"asset_id", "x_m", "y_m", "yaw_rad"},
        }
        if op not in signatures or not isinstance(target, str) or not target or len(target) > 64:
            raise ValueError("unsupported operation or target")
        _exact(args, signatures[op], "edit arguments")
        if op in ("insert_vehicle", "add_occluder"):
            if not isinstance(args["asset_id"], str) or not args["asset_id"]:
                raise ValueError("asset_id must be a catalogue key")
            for key in ("x_m", "y_m", "yaw_rad"):
                _number(args[key], key)
        if op == "modify_lane_marking" and args["marking"] not in ("solid", "dashed"):
            raise ValueError("unsupported marking")
        return cls(op, target, args)


@dataclass(frozen=True)
class EditPlan:
    schema: str
    commands: tuple[EditCommand, ...]

    @classmethod
    def parse(cls, text_or_value: str | dict) -> "EditPlan":
        data = _hugsim_expand(json.loads(text_or_value)) if isinstance(text_or_value, str) else text_or_value
        _exact(data, {"schema", "commands"}, "edit plan")
        if data["schema"] != "causal_edit_plan_v1" or not isinstance(data["commands"], list) or not 1 <= len(data["commands"]) <= 4:
            raise ValueError("unsupported edit plan")
        commands = tuple(EditCommand.parse(c) for c in data["commands"])
        if len({c.target_id for c in commands}) != len(commands):
            raise ValueError("duplicate target in one edit plan")
        return cls(data["schema"], commands)


@dataclass(frozen=True)
class CapabilityCatalog:
    """Reality of the connected backend; do not infer support from a schema."""
    managed_ids: frozenset[str]
    asset_ids: frozenset[str]
    native_vehicle_removal: bool = False
    lane_marking_edit: bool = False
    occluder_insertion: bool = False
    calibrated_vehicle_insertion: bool = False
    formal_evaluation_admitted: bool = False

    def check(self, command: EditCommand) -> None:
        if command.operation == "insert_vehicle":
            if not self.calibrated_vehicle_insertion or command.arguments["asset_id"] not in self.asset_ids:
                raise ValueError("vehicle insertion unsupported or asset not calibrated")
        elif command.operation == "remove_vehicle":
            if command.target_id not in self.managed_ids and not self.native_vehicle_removal:
                raise ValueError("native vehicle removal unsupported")
        elif command.operation == "modify_lane_marking" and not self.lane_marking_edit:
            raise ValueError("lane marking edit unsupported")
        elif command.operation == "add_occluder":
            if not self.occluder_insertion or command.arguments["asset_id"] not in self.asset_ids:
                raise ValueError("occluder insertion unsupported or asset not calibrated")


@dataclass(frozen=True)
class GeometryCertificate:
    """Evidence produced by a verifier, never accepted from model output."""
    source_sha256: str
    independent_source_sha256: str
    measured_footprint_clearance_m: float
    measured_hover_m: float
    measured_penetration_m: float
    multiview_max_reprojection_px: float
    multiview_cameras: tuple[str, ...]
    observed_revision: int
    physical_skin_checked: bool
    independent_modality: str = "unverified"

    def check(self, revision: int, *, require_independent: bool) -> None:
        for name in ("source_sha256", "independent_source_sha256"):
            digest = getattr(self, name)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid evidence hash")
        if self.observed_revision != revision or not self.physical_skin_checked:
            raise ValueError("stale or missing physical skin certificate")
        if require_independent and self.source_sha256 == self.independent_source_sha256:
            raise ValueError("edited renderer cannot independently certify its own pixels")
        if require_independent and self.independent_modality not in (
                "source_lidar", "surveyed_mesh", "real_multiview_depth", "synthetic_oracle"):
            raise ValueError("independent geometry modality not certified")
        if len(set(self.multiview_cameras)) < 2:
            raise ValueError("at least two independent camera views required")
        values = (self.measured_footprint_clearance_m, self.measured_hover_m,
            self.measured_penetration_m, self.multiview_max_reprojection_px)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("non-finite geometry evidence")
        if self.measured_footprint_clearance_m < 0.5 or self.measured_hover_m > 0.08 or self.measured_penetration_m > 0.05:
            raise ValueError("collision, hover or ground penetration")
        if self.multiview_max_reprojection_px > 2.0:
            raise ValueError("inconsistent camera projections")


@dataclass
class TrustGrid:
    x0_m: float
    y0_m: float
    resolution_m: float
    cells: list[list[dict]]
    revision: int = 0
    quarantined: set[tuple[int, int]] = field(default_factory=set)

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        if self.resolution_m <= 0 or not math.isfinite(self.resolution_m):
            raise ValueError("invalid trust grid resolution")
        i, j = math.floor((y-self.y0_m)/self.resolution_m), math.floor((x-self.x0_m)/self.resolution_m)
        if i < 0 or i >= len(self.cells) or j < 0 or j >= len(self.cells[i]):
            raise ValueError("edit outside measured trust grid")
        return i, j

    def footprint(self, *, x_m: float, y_m: float, width_m: float, length_m: float, yaw_rad: float) -> set[tuple[int, int]]:
        """Sample a whole rotated footprint at <= quarter-cell spacing."""
        if width_m <= 0 or length_m <= 0 or not all(map(math.isfinite, (x_m,y_m,width_m,length_m,yaw_rad))):
            raise ValueError("invalid footprint")
        count_x = max(2, math.ceil(width_m/(self.resolution_m/4)))
        count_y = max(2, math.ceil(length_m/(self.resolution_m/4)))
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        touched = set()
        for a in range(count_x+1):
            lateral = width_m*(a/count_x-.5)
            for b in range(count_y+1):
                forward = length_m*(b/count_y-.5)
                touched.add(self._cell(x_m+c*lateral-s*forward,y_m+s*lateral+c*forward))
        return touched

    def preflight(self, touched: set[tuple[int,int]], certificate: GeometryCertificate) -> dict:
        certificate.check(self.revision, require_independent=True)
        if not touched:raise ValueError("empty edit footprint")
        for i,j in touched:
            cell=self.cells[i][j]
            if (i,j) in self.quarantined or cell.get("confidence",0) < .95 or cell.get("skin_normal_z",0) < math.cos(math.radians(15)) or cell.get("camera_count",0) < 2:
                raise ValueError("edit footprint not in certified trusted domain")
            if abs(cell.get("skin_z_m",float("inf"))-cell.get("ground_z_m",0)) > .08:
                raise ValueError("geometric skin inconsistent with ground")
        return dict(admitted=True,checked_cells=len(touched),pre_edit_revision=self.revision,
            evidence_sha256=canonical_hash(certificate.__dict__))

    def apply_quarantine(self, touched: set[tuple[int,int]]) -> int:
        if not touched:raise ValueError("empty quarantine")
        self.quarantined.update(touched)
        self.revision += 1
        return self.revision

    def reverify(self, touched: set[tuple[int,int]], certificate: GeometryCertificate) -> dict:
        if not touched or not touched <= self.quarantined:
            raise ValueError("reverification must address quarantined cells")
        certificate.check(self.revision, require_independent=True)
        self.quarantined.difference_update(touched)
        return dict(restored_cells=len(touched),revision=self.revision,
            independent_evidence_sha256=canonical_hash(certificate.__dict__))
