"""Original sampler and atomic checkpoint writer, without unrelated model imports."""
import os
import torch
from torch.utils.data import Sampler

def save(state,path):
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    temporary.replace(path)
    directory=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)

class EpochSampler(Sampler):
    def __init__(self,length,seed):self.length=length;self.seed=seed;self.epoch=0;self.start=0
    def __iter__(self):
        order=torch.randperm(self.length,generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()
        return iter(order[self.start:])
    def __len__(self):return self.length-self.start
