"""Fit global and regional attribute heads on a shared frozen CLIP feature cache."""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from clip_large import RegionHead
from metrics import score,sigmoid
from attribute_losses import bce_polarity_weights,soft_domain_f1_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=25);p.add_argument('--batch-size',type=int,default=2048)
    p.add_argument('--selection-metric',choices=['Challenge_Avg','Instance_F1'],default='Challenge_Avg')
    p.add_argument('--domain-power',type=float,default=0.)
    p.add_argument('--balance-power',type=float,default=1.)
    p.add_argument('--inverse-balance-power',type=float,default=0.)
    p.add_argument('--initialize-head',help='Initialize from the head of a fine-tuned EMA checkpoint')
    p.add_argument('--head-lr',type=float,default=.003)
    p.add_argument('--f1-loss-weight',type=float,default=0.)
    p.add_argument('--regional-only',action='store_true')
    args=p.parse_args();torch.set_num_threads(4);torch.manual_seed(2092);np.random.seed(2092)
    cache=Path(args.cache);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((cache/'manifest.json').read_text())
    x=torch.from_numpy(np.load(cache/'train_features.npy')).cuda()
    vx=torch.from_numpy(np.load(cache/'tune_features.npy')).cuda()
    train=np.load(cache/'train_labels.npz');tune=np.load(cache/'tune_labels.npz')
    y=torch.from_numpy(train['labels']).cuda();vy=tune['labels']
    domains=np.array([p.split('/')[0] for p in tune['paths']])
    train_domains=np.array([p.split('/')[0] for p in train['paths']])
    counts={d:int((train_domains==d).sum()) for d in np.unique(train_domains)}
    dw=np.array([counts[d]**-args.domain_power for d in train_domains],dtype=np.float32);dw/=dw.mean()
    domain_weights=torch.from_numpy(dw).cuda()
    domain_ids=torch.tensor([sorted(counts).index(d) for d in train_domains],device='cuda')
    rate=(y*domain_weights[:,None]).sum(0)/domain_weights.sum()
    positive_weight,negative_weight=bce_polarity_weights(rate,args.balance_power,args.inverse_balance_power)
    (out/'config.json').write_text(json.dumps(dict(args=vars(args),cache_manifest=meta,attributes=train['names'].tolist()),indent=2))
    summaries={}
    for kind in (['regional'] if args.regional_only else ['global','regional']):
        torch.manual_seed(2092)
        head=RegionHead(regional=kind=='regional',dim=x.shape[-1]).cuda()
        with torch.no_grad():head.global_head.bias.copy_(torch.logit(rate.clamp(.001,.999)))
        if args.initialize_head:
            parent=torch.load(args.initialize_head,map_location='cpu',weights_only=False)
            initial={k.removeprefix('head.'):v for k,v in parent['ema'].items() if k.startswith('head.')}
            head.load_state_dict(initial,strict=True);del parent,initial
        optimizer=torch.optim.AdamW(head.parameters(),lr=args.head_lr,weight_decay=.01)
        best=-1;started=time.monotonic()
        for epoch in range(args.epochs):
            head.train();order=torch.randperm(len(x),device='cuda');total=0
            for group in optimizer.param_groups:group['lr']=args.head_lr*(.1+.9*.5*(1+math.cos(math.pi*epoch/args.epochs)))
            for start in range(0,len(x),args.batch_size):
                ii=order[start:start+args.batch_size];target=y[ii]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=head(x[ii]);weight=target*positive_weight+(1-target)*negative_weight
                    loss=((F.binary_cross_entropy_with_logits(logits.float(),target,reduction='none')*weight).mean(1)*domain_weights[ii]).mean()
                    if args.f1_loss_weight:
                        loss=loss+args.f1_loss_weight*soft_domain_f1_loss(logits,target,domain_ids[ii],len(counts))
                loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5);optimizer.step();total+=float(loss)
            head.eval()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                pred=torch.cat([head(vx[start:start+1024]).float() for start in range(0,len(vx),1024)]).cpu().numpy()
            metrics=score(vy,sigmoid(pred),domains);value=metrics['mean'][args.selection_metric]
            record=dict(kind=kind,epoch=epoch+1,loss=total/math.ceil(len(x)/args.batch_size),seconds=time.monotonic()-started,**metrics)
            with (out/(kind+'_history.jsonl')).open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(dict(kind=kind,epoch=epoch+1,score=value,seconds=record['seconds'])),flush=True)
            if value>best:
                best=value;torch.save(dict(head=head.state_dict(),regional=kind=='regional',epoch=epoch+1,cache_manifest=meta),out/(kind+'_best.pt'))
                np.savez_compressed(out/(kind+'_tune.npz'),paths=tune['paths'],labels=vy,logits=pred)
                (out/(kind+'_best_metrics.json')).write_text(json.dumps(record,indent=2))
                summaries[kind]=dict(epoch=epoch+1,score=value)
        del head,optimizer
    (out/'summary.json').write_text(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
