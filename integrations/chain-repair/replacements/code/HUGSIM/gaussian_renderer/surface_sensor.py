"""Sparse opacity-quantile observations. Unknown foreground stays occluding."""
import math
import torch


@torch.no_grad()
def front_surface(meta, scales, alpha_image, stride=4, uncertainty_m=.5, max_range=50.):
    if int(meta['n_cameras']) != 1:
        raise ValueError('Surface sensor expects one camera')
    height, width = int(meta['height']), int(meta['width'])
    tile = int(meta['tile_size'])
    if stride <= 0 or tile % stride or uncertainty_m <= 0 or max_range <= .1:
        raise ValueError('Invalid surface sampling parameters')
    if tuple(alpha_image.shape) != (height, width):
        raise ValueError('Invalid alpha image shape')
    device = scales.device
    ids = meta['flatten_ids'].long()
    tile_ids = (meta['isect_ids'] >> 32).long()
    z = meta['depths'][0]
    allowed = (scales.max(-1).values <= uncertainty_m) & (z > .1) & (z < max_range)
    # Do not discard untrusted primitives before compositing. They can occlude
    # a trusted surface; removing them would turn unknown space into free space.
    # Explicit stable ordering also makes the contract independent of a caller's
    # intersection ordering, preserving near-to-far order within each tile.
    order = torch.argsort(z[ids], stable=True)
    ids, tile_ids = ids[order], tile_ids[order]
    order = torch.argsort(tile_ids, stable=True)
    ids, tile_ids = ids[order], tile_ids[order]
    count_tiles = int(meta['tile_height']) * int(meta['tile_width'])
    counts = torch.bincount(tile_ids, minlength=count_tiles)
    offsets = torch.cumsum(counts, 0) - counts
    counts_cpu = counts.cpu().tolist()
    active = [i for i in torch.argsort(counts).cpu().tolist() if counts_cpu[i]]
    step = tile // stride
    yy, xx = torch.meshgrid(torch.arange(step, device=device) * stride,
                            torch.arange(step, device=device) * stride, indexing='ij')
    relative = torch.stack([xx.flatten(), yy.flatten()], -1)
    pixels = len(relative)
    out = torch.zeros((height, width), device=device)
    valid_image = torch.zeros((height, width), device=device, dtype=torch.bool)
    means, conics, opacity = meta['means2d'][0], meta['conics'][0], meta['opacities'][0]
    cursor = 0
    while cursor < len(active):
        batch = active[cursor:cursor + 16]
        maximum = max(counts_cpu[i] for i in batch)
        batch = batch[:max(1, 2000000 // max(1, maximum * pixels))]
        maximum = max(counts_cpu[i] for i in batch)
        cursor += len(batch)
        tiles = torch.tensor(batch, device=device)
        j = torch.arange(maximum, device=device)
        in_tile = j[None] < counts[tiles, None]
        selected = ids[(offsets[tiles, None] + j[None]).clamp(max=len(ids) - 1)]
        tile_xy = torch.stack([tiles % int(meta['tile_width']),
                               tiles // int(meta['tile_width'])], -1) * tile
        xy = tile_xy[:, None] + relative[None]
        delta = xy[:, None].to(means.dtype) + .5 - means[selected][:, :, None]
        c = conics[selected]
        power = .5 * (c[:, :, None, 0] * delta[..., 0] ** 2
                      + c[:, :, None, 2] * delta[..., 1] ** 2)
        power += c[:, :, None, 1] * delta[..., 0] * delta[..., 1]
        a = (opacity[selected][:, :, None] * torch.exp(-power)).clamp(0, .999)
        a = torch.where(in_tile[:, :, None] & (power >= 0) & (a >= 1 / 255.), a, 0.)
        cumulative = (-torch.log1p(-a)).cumsum(1)
        untrusted_prefix = ((~allowed[selected])[:, :, None] & (a > 0)).cumsum(1)
        depths = z[selected][:, :, None].expand(-1, -1, pixels)
        quantiles, hits, trusted = [], [], []
        for q in (.05, .1):
            reached = cumulative >= -math.log1p(-q)
            index = reached.to(torch.int8).argmax(1)
            quantiles.append(torch.gather(depths, 1, index[:, None]).squeeze(1))
            hits.append(reached.any(1))
            trusted.append(torch.gather(untrusted_prefix, 1, index[:, None]).squeeze(1) == 0)
        inside = (xy[..., 0] < width) & (xy[..., 1] < height)
        alpha = alpha_image[xy[..., 1].clamp(max=height - 1), xy[..., 0].clamp(max=width - 1)]
        valid = (inside & torch.isfinite(alpha) & (alpha >= .9) & hits[0] & hits[1]
                 & trusted[0] & trusted[1]
                 & ((quantiles[1] - quantiles[0]).abs() <= uncertainty_m))
        y, x = xy[..., 1][inside], xy[..., 0][inside]
        out[y, x] = torch.where(valid, quantiles[1], 0.)[inside]
        valid_image[y, x] = valid[inside]
    return out, valid_image
