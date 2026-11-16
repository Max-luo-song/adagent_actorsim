"""Current-reference boundary correction; no world/route history or scene IDs."""
import numpy as np

def join_reference(raw,y,origin,length,curvature):
    raw=np.asarray(raw,float);y=np.asarray(y,float);origin=np.asarray(origin,float)
    if raw.ndim!=1 or raw.shape!=y.shape or len(y)<2 or origin.shape!=(2,) or not np.isfinite(raw).all() or not np.isfinite(y).all() or not np.isfinite(origin).all() or not np.isfinite([length,curvature]).all() or length<=0 or np.any(np.diff(y)<=0):raise ValueError('Invalid current-reference join')
    distance=y-origin[1];u=np.clip(distance/length,0.,1.);blend=u*u*u*(10+u*(-15+6*u))
    offset=raw[0]-origin[0];slope=(raw[1]-raw[0])/(y[1]-y[0])
    # Correct boundary mismatch, not the entire current trajectory shape.
    return raw-(offset+slope*distance)*(1-blend)+.5*curvature*distance**2*(1-u)**3
