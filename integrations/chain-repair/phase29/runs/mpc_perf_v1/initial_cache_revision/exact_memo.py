"""Bounded exact-value memoization; each optimizer call owns fresh caches."""
from collections import OrderedDict
import functools
import numpy as np

def key(value):
    a=np.asarray(value)
    if a.dtype.hasobject:raise TypeError('Only numeric arguments may be memoized')
    return (a.dtype.str,a.shape,np.ascontiguousarray(a).tobytes())

def clone(value):
    if isinstance(value,np.ndarray):return value.copy()
    if isinstance(value,tuple):return tuple(clone(x) for x in value)
    return value

class Memo:
    def __init__(self,capacity=64):self.capacity=capacity;self.metrics={}
    def wrap(self,fn):
        cache=OrderedDict();metric=dict(hits=0,misses=0,evictions=0);self.metrics[fn.__name__]=metric
        @functools.wraps(fn)
        def call(*args,**kwargs):
            k=(tuple(key(x) for x in args),tuple((n,key(v)) for n,v in sorted(kwargs.items())))
            if k in cache:
                metric['hits']+=1;cache.move_to_end(k);return clone(cache[k])
            result=fn(*args,**kwargs);metric['misses']+=1
            cache[k]=clone(result)
            if len(cache)>self.capacity:cache.popitem(last=False);metric['evictions']+=1
            return result
        return call
    def stats(self):return {name:dict(value) for name,value in self.metrics.items()}
