"""Frozen legacy Waymo PBR inference compatibility.

Lighting architecture and material formula are ported from Omniscenegraph
2026-04-18 scene code. Checkpoint names/shapes and historical lighting time
normalization are preserved. This module imports only Torch; it does not train.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class XYZ_Encoder(nn.Module):
    encoder_type = "XYZ_Encoder"
    """Encode XYZ coordinates or directions to a vector."""

    def __init__(self, n_input_dims):
        super().__init__()
        self.n_input_dims = n_input_dims

    @property
    def n_output_dims(self) -> int:
        raise NotImplementedError

class SinusoidalEncoder(XYZ_Encoder):
    encoder_type = "SinusoidalEncoder"
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(
        self,
        n_input_dims: int = 3,
        min_deg: int = 0,
        max_deg: int = 10,
        enable_identity: bool = True,
    ):
        super().__init__(n_input_dims)
        self.n_input_dims = n_input_dims
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.enable_identity = enable_identity
        self.register_buffer(
            "scales", Tensor([2**i for i in range(min_deg, max_deg + 1)])
        )

    @property
    def n_output_dims(self) -> int:
        return (
            int(self.enable_identity) + (self.max_deg - self.min_deg + 1) * 2
        ) * self.n_input_dims

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [..., n_input_dims]
        Returns:
            encoded: [..., n_output_dims]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[..., None, :] * self.scales[:, None]),
            list(x.shape[:-1])
            + [(self.max_deg - self.min_deg + 1) * self.n_input_dims],
        )
        encoded = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
        if self.enable_identity:
            encoded = torch.cat([x] + [encoded], dim=-1)
        return encoded

class MLP(nn.Module):
    """A simple MLP with skip connections."""

    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        num_layers: int = 3,
        hidden_dims: Optional[int] = 256,
        skip_connections: Optional[Tuple[int]] = [0],
    ) -> None:
        super().__init__()
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.n_output_dims = out_dims
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(in_dims, out_dims))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    layers.append(nn.Linear(in_dims, hidden_dims))
                elif i in skip_connections:
                    layers.append(nn.Linear(in_dims + hidden_dims, hidden_dims))
                else:
                    layers.append(nn.Linear(hidden_dims, hidden_dims))
            layers.append(nn.Linear(hidden_dims, out_dims))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        input = x
        for i, layer in enumerate(self.layers):
            if i in self.skip_connections:
                x = torch.cat([x, input], -1)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.functional.relu(x)
        return x

class GlobalLightingModule(nn.Module):
    def __init__(
        self,
        class_name: str,
        n: int,
        sh_degree: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 4,
        appearance_embedding_dim: int = 16,
        device: torch.device = torch.device("cuda")
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.device = device
        self.sh_degree = sh_degree
        self.num_sh_coeffs = (self.sh_degree + 1) ** 2
        self.output_dim = 3 * self.num_sh_coeffs
        self.time_encoding = SinusoidalEncoder(n_input_dims=1, min_deg=0, max_deg=4)
        self.appearance_embedding_dim = appearance_embedding_dim
        self.appearance_embedding = nn.Embedding(n, appearance_embedding_dim)
        in_dims = self.time_encoding.n_output_dims + self.appearance_embedding_dim
        self.mlp = MLP(
            in_dims=in_dims,
            out_dims=self.output_dim,
            num_layers=num_layers,
            hidden_dims=hidden_dim,
            skip_connections=[max(1, num_layers // 2)],
        )

    def forward(self, normalized_time: Tensor, camera_idx: Tensor):
        time_emb = self.time_encoding(normalized_time)
        cam_emb = self.appearance_embedding(camera_idx)
        if cam_emb.dim() > time_emb.dim():
            cam_emb = cam_emb.squeeze(1)
        x = torch.cat([time_emb, cam_emb], dim=-1)
        sh_coeffs = self.mlp(x)
        return sh_coeffs.reshape(-1, self.num_sh_coeffs, 3)

    def get_param_groups(self):
        return {
            self.class_prefix+"all": self.parameters(),
        }

MATERIAL_SHAPES = {
    "_roughness": (1,), "_metallic": (1,), "_normals": (3,),
    "_asg_amplitude": (4, 3), "_asg_sharpness": (4, 2), "_asg_dir": (4, 3),
}


def init_material(model):
    model.use_pbr = bool(model.ctrl_cfg.get("use_pbr", False))
    if model.use_pbr:
        model.num_asg_lobes = 4
        for name, shape in MATERIAL_SHAPES.items():
            setattr(model, name, torch.zeros((1,) + shape, device=model.device))


def load_material(model, state):
    if not model.use_pbr:
        return
    count = state["_means"].shape[0]
    # All six fields are mandatory, including the two unused by the old formula.
    # No random/zero replacements for missing pretrained material values.
    for name, shape in MATERIAL_SHAPES.items():
        if name not in state or tuple(state[name].shape) != (count,) + shape:
            raise ValueError(f"Missing or invalid PBR checkpoint tensor: {name}")
    for name in MATERIAL_SHAPES:
        setattr(model, name, nn.Parameter(torch.empty_like(state[name], device=model.device)))


def lighting_for_frame(module, frame, image_id, num_full_images, device):
    # Historical training uses num_full_images, NOT number of timesteps.
    # Retain this convention for frozen checkpoints.
    time = torch.as_tensor(frame, device=device).float().reshape(1, 1) / max(1, num_full_images - 1)
    index = torch.as_tensor(image_id, device=device).long().reshape(1)
    if index.item() < 0 or index.item() >= module.appearance_embedding.num_embeddings:
        raise ValueError("Lighting image index is outside the checkpoint embeddings")
    return module(time, index).squeeze(0)


def compute_pbr_color(model, cam, global_sh_coeffs, spherical_harmonics):
    albedo = torch.sigmoid(model._features_dc)
    normals = F.normalize(model._normals, dim=-1)
    viewdirs = -F.normalize(model._means.detach() - cam.camtoworlds.data[..., :3, 3], dim=-1)
    if global_sh_coeffs is not None:
        coeffs = global_sh_coeffs.unsqueeze(0).expand(model.num_points, -1, -1)
        degree = int(global_sh_coeffs.shape[-2] ** .5) - 1
        if (degree + 1) ** 2 != global_sh_coeffs.shape[-2]:
            raise ValueError("Lighting SH coefficient count must be square")
        diffuse = albedo * spherical_harmonics(degree, normals, coeffs).clamp_min(0)
    else:
        diffuse = albedo * .1
    specular = torch.zeros_like(diffuse)
    amplitude = torch.sigmoid(model._asg_amplitude)
    directions = F.normalize(model._asg_dir, dim=-1)
    shininess = 1.0 / (torch.sigmoid(model._roughness) + 1e-6)
    for i in range(model.num_asg_lobes):
        half = F.normalize(viewdirs + directions[:, i, :], dim=-1)
        ndoth = (normals * half).sum(-1, keepdim=True).clamp(0., 1.)
        specular += amplitude[:, i, :] * torch.pow(ndoth, shininess * 10.)
    hdr = diffuse + specular
    return hdr / (1. + hdr)
