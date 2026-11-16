"""Sparse front-surface depth from sorted Gaussian opacity, not averaged depth."""
import math
import torch

@torch.no_grad()
def front_surface(meta,scales,alpha_image,stride=4,uncertainty_m=.5,max_range=50.):
    if int(meta['n_cameras'])!=1:raise ValueError('Surface sensor expects one camera')
    height,width=int(meta['height']),int(meta['width']);tile=int(meta['tile_size'])
    if tile%stride:raise ValueError('Surface stride must divide raster tile size')
    count_tiles=int(meta['tile_height'])*int(meta['tile_width']);device=scales.device
    ids=meta['flatten_ids'].long();tile_ids=(meta['isect_ids']>>32).long()
    z=meta['depths'][0];allowed=(scales.max(-1).values<=uncertainty_m)&(z>.1)&(z<max_range)
    good=allowed[ids];ids=ids[good];tile_ids=tile_ids[good]
    counts=torch.bincount(tile_ids,minlength=count_tiles);offsets=torch.cumsum(counts,0)-counts
    order=torch.argsort(counts);counts_cpu=counts.cpu().tolist();order_cpu=order.cpu().tolist()
    step=tile//stride;yy,xx=torch.meshgrid(torch.arange(step,device=device)*stride,torch.arange(step,device=device)*stride,indexing='ij')
    relative=torch.stack([xx.flatten(),yy.flatten()],-1);pixels=len(relative)
    out=torch.zeros((height,width),device=device);valid_image=torch.zeros((height,width),device=device,dtype=torch.bool)
    means=meta['means2d'][0];conics=meta['conics'][0];opacity=meta['opacities'][0]
    active=[index for index in order_cpu if counts_cpu[index]>0];cursor=0
    while cursor<len(active):
        batch=active[cursor:cursor+16];maximum=max(counts_cpu[index] for index in batch)
        if maximum*pixels*len(batch)>2000000:
            batch=batch[:max(1,2000000//max(1,maximum*pixels))];maximum=max(counts_cpu[index] for index in batch)
        cursor+=len(batch);tiles=torch.tensor(batch,device=device);j=torch.arange(maximum,device=device)
        in_tile=j[None]<counts[tiles,None];positions=offsets[tiles,None]+j[None]
        selected=ids[positions.clamp(max=max(0,len(ids)-1))]
        tile_xy=torch.stack([tiles%int(meta['tile_width']),tiles//int(meta['tile_width'])],-1)*tile
        xy=tile_xy[:,None,:]+relative[None];delta=xy[:,None].to(means.dtype)+.5-means[selected][:,:,None]
        conic=conics[selected];power=.5*(conic[:,:,None,0]*delta[...,0]**2+conic[:,:,None,2]*delta[...,1]**2)+conic[:,:,None,1]*delta[...,0]*delta[...,1]
        a=(opacity[selected][:,:,None]*torch.exp(-power)).clamp(0,.999)
        a=torch.where(in_tile[:,:,None]&(power>=0)&(a>=1/255.),a,torch.zeros_like(a))
        cumulative=(-torch.log1p(-a)).cumsum(1)
        depths=z[selected][:,:,None].expand(-1,-1,pixels);quantiles=[];hits=[]
        for q in (.05,.1):
            reached=cumulative>=-math.log1p(-q);index=reached.to(torch.int8).argmax(1)
            quantiles.append(torch.gather(depths,1,index[:,None]).squeeze(1));hits.append(reached.any(1))
        inside=(xy[...,0]<width)&(xy[...,1]<height)
        opaque=alpha_image[xy[...,1].clamp(max=height-1),xy[...,0].clamp(max=width-1)]>=.9
        valid=inside&opaque&hits[0]&hits[1]&((quantiles[1]-quantiles[0]).abs()<=uncertainty_m)
        y=xy[...,1][inside];x=xy[...,0][inside];out[y,x]=torch.where(valid,quantiles[1],torch.zeros_like(quantiles[1]))[inside];valid_image[y,x]=valid[inside]
    return out,valid_image
