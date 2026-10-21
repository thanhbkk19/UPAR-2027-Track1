"""Score fold logits with the v11 decision rule: logit(mean prob) - 0.4*logit(prior) + bias > 0.

Usage: fold_eval.py --members wise2_pe:alpha_0.8 wise2_eva:alpha_0.7 [--bias .25]
Each member is <npz prefix under artifacts/loso>:<key>; the fold name is appended as _<fold>.npz.
"""
import argparse
import numpy as np
from data import read_annotations
from metrics import score_dataset,sigmoid

FOLDS=['PETA','Market1501','PA100k']


def fold_prior(fold):
    _,images,labels=read_annotations('train',f'artifacts/loso/split_{fold}/train.csv')
    dom=np.array([p.split('/')[0] for p in images])
    return np.mean([labels[dom==d].mean(0) for d in np.unique(dom)],0).clip(1e-4,1-1e-4)


def logit(p):return np.log(p)-np.log1p(-p)


def decide(logits_list,prior,bias=.25,k=.4):
    p=np.mean([sigmoid(z) for z in logits_list],0).clip(1e-7,1-1e-7)
    return sigmoid(logit(p)-k*logit(prior)+bias)


def evaluate(members,bias=.25,k=.4,verbose=True):
    out=[]
    for f in FOLDS:
        zs=[];labels=None
        for m in members:
            prefix,key=m.split(':');d=np.load(f'artifacts/loso/{prefix}_{f}.npz')
            zs.append(d[key]);labels=d['labels'] if labels is None else labels
        r=score_dataset(labels,decide(zs,fold_prior(f),bias,k));out.append(r)
        if verbose:print(f,{k2:round(r[k2],4) for k2 in ['Challenge_Avg','mA','Instance_F1']})
    mean=np.mean([r['Challenge_Avg'] for r in out]);print('MEAN',round(mean,4));return mean


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--members',nargs='+',required=True)
    p.add_argument('--bias',type=float,default=.25);p.add_argument('--k',type=float,default=.4)
    a=p.parse_args();evaluate(a.members,a.bias,a.k)
