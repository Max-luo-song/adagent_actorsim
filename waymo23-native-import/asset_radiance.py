"""Rotate library-asset view-dependent radiance with its physical geometry."""
import ast
from functools import lru_cache
from pathlib import Path
import numpy as np


@lru_cache(maxsize=1)
def evaluator():
    # Use the renderer's actual SH constants/polynomials without importing CUDA.
    path=Path(__file__).resolve().parents[1]/'code/HUGSIM/utils/sh_utils.py'
    tree=ast.parse(path.read_text())
    tree.body=[n for n in tree.body if isinstance(n,(ast.Assign,ast.FunctionDef))]
    scope={}
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),scope)
    return scope['eval_sh']


def basis(directions, degree):
    count=(degree+1)**2
    values=evaluator()(degree,np.eye(count),np.asarray(directions))
    return np.broadcast_to(values,(len(directions),count))


def sh_rotation(local_to_world, degree=3):
    """C_world = M @ C_local; Y(d_world) M = Y(d_world @ R).

    Solve each finite-dimensional SH degree separately on a fixed sphere.
    Rotation preserves degree; no DC/high-order mixing or fitting to imagery.
    """
    rotation=np.asarray(local_to_world,dtype=float)
    if (rotation.shape!=(3,3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6)
            or not np.isclose(np.linalg.det(rotation),1.,atol=1e-6)):
        raise ValueError('Radiance rotation must be a proper rigid rotation')
    if degree not in (0,1,2,3):
        raise ValueError('Supported radiance degrees are 0..3')
    i=np.arange(128)+.5;z=1-2*i/len(i);angle=i*np.pi*(3-np.sqrt(5))
    directions=np.column_stack([np.sqrt(1-z*z)*np.cos(angle),np.sqrt(1-z*z)*np.sin(angle),z])
    world=basis(directions,degree);local=basis(directions@rotation,degree)
    result=np.zeros(((degree+1)**2,)*2)
    for level in range(degree+1):
        sl=slice(level**2,(level+1)**2)
        result[sl,sl]=np.linalg.lstsq(world[:,sl],local[:,sl],rcond=None)[0]
    result[0,0]=1.
    return result
