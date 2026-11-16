"""Original geometry-only partition of a frozen foreground mask."""
import numpy as np

def partition(mask):
    coords=np.argwhere(mask)
    assert len(coords)>=8
    def split(idx,depth):
        if depth==0:return [idx]
        c=coords[idx];span=np.ptp(c,axis=0);axis=int(np.argmax(span))
        order=np.lexsort((c[:,1-axis],c[:,axis]));ordered=idx[order];cut=len(idx)//2
        return split(ordered[:cut],depth-1)+split(ordered[cut:],depth-1)
    groups=split(np.arange(len(coords)),3)
    assert np.array_equal(np.sort(np.concatenate(groups)),np.arange(len(coords)))
    return coords,groups
