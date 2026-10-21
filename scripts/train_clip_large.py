"""Fine-tune a pretrained CLIP-L model after controlled frozen-feature head fits."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from clip_large import ClipLarge
from data import UPAR,read_annotations
from metrics import score,sigmoid
from training_utils import EpochSampler,save
from attribute_losses import bce_polarity_weights


@torch.inference_mode()
def evaluate(model,loader):
    model.eval();pred=np.empty_like(loader.dataset.labels)
    for x,_,ii in loader:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(x.cuda(non_blocking=True))
        pred[ii.numpy()]=logits.float().cpu().numpy()
    return pred,score(loader.dataset.labels,sigmoid(pred),loader.dataset.domains)


def selection_value(metrics,metric,logits,dataset,folds=None):
    if metric=='Joint_Min':return min(metrics['mean'][k] for k in ['Instance_F1','Challenge_Avg'])
    if metric=='Calibrated_Joint':
        from calibrate_f1 import fit
        prediction=np.empty_like(logits)
        for fold in range(3):
            train=folds!=fold
            offset=fit(logits[train].astype(float),dataset.labels[train],dataset.domains[train],objective='joint_min')*.75
            prediction[~train]=sigmoid(logits[~train]-offset)
        metrics['calibration_cv']=score(dataset.labels,prediction,dataset.domains)
        return min(metrics['calibration_cv']['mean'][k] for k in ['Instance_F1','Challenge_Avg'])
    return metrics['mean'][metric]


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--head-checkpoint',required=True)
    p.add_argument('--tag',default='apple_mclip2_dfndr2b');p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=6);p.add_argument('--last-blocks',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--encoder-lr',type=float,default=6e-6);p.add_argument('--head-lr',type=float,default=2e-4)
    p.add_argument('--seed',type=int,default=2093);p.add_argument('--resume');p.add_argument('--split-root',default='data/splits/train90_v1')
    p.add_argument('--architecture',choices=['clip','dino','siglip','solider'],default='clip')
    p.add_argument('--dino-variant',choices=['base','large','large_v3'],default='base')
    p.add_argument('--height',type=int,default=224);p.add_argument('--width',type=int,default=224)
    p.add_argument('--domain-power',type=float,default=.5)
    p.add_argument('--initialize-checkpoint',help='Start a new fine-tuning run from a selected EMA checkpoint')
    p.add_argument('--balance-power',type=float,default=1.)
    p.add_argument('--inverse-balance-power',type=float,default=0.)
    p.add_argument('--f1-loss-weight',type=float,default=0.)
    p.add_argument('--selection-metric',choices=['Challenge_Avg','Instance_F1','Joint_Min','Calibrated_Joint'],default='Challenge_Avg')
    p.add_argument('--mild-augmentation',action='store_true');p.add_argument('--shift-augmentation',action='store_true');p.add_argument('--timm-name')
    p.add_argument('--attribute-attention',action='store_true')
    p.add_argument('--query-lr-mult',type=float,default=1.)
    p.add_argument('--drop-path',type=float,default=0.,help='Stochastic depth rate (CLIP architecture only)')
    p.add_argument('--source-prior',action='store_true',help='Source-prior factorization: per-source attribute bias during training, averaged into the head for inference')
    p.add_argument('--tune-in-train',action='store_true',help='Final all-data runs: tune rows are part of training, so tune scores are in-sample')
    p.add_argument('--save-epochs',action='store_true',help='Also keep every epoch EMA as epoch<N>.pt')
    p.add_argument('--lowres-probability',type=float,help='Override the downscale probability of the default training degradation')
    p.add_argument('--feature-anchor',type=float,default=0.,help='Weight of squared distance between normalized fine-tuned and initial features (CLIP architecture only)')
    p.add_argument('--semantic-weight',type=float,default=.2,help='SOLIDER semantic controller value (solider architecture only)')
    p.add_argument('--source-heads',action='store_true',help='One head per training source (each row supervised by its own source head); inference averages their probabilities')
    p.add_argument('--layer-decay',type=float,default=1.,help='Layer-wise encoder learning-rate decay; 1 keeps one encoder rate')
    p.add_argument('--mixup-alpha',type=float,default=0.,help='Input mixup Beta(alpha,alpha) on images, targets and domain weights; 0 disables')
    p.add_argument('--perspective-probability',type=float,default=0.,help='Probability of the ElevationWarp viewpoint augmentation per training image')
    p.add_argument('--perspective-strength',type=float,default=.25,help='Maximum ElevationWarp convergence strength')
    p.add_argument('--micro-batches',type=int,default=1,help='Split each optimizer batch into this many sequential forward/backward chunks (same maths, less activation memory)')
    p.add_argument('--mixup-probability',type=float,default=1.,help='Fraction of training batches that are mixed when mixup is enabled')
    args=p.parse_args()
    if args.architecture=='dino':args.tag='dinov3_lvd1689m' if args.dino_variant=='large_v3' else 'dinov2_lvd142m'
    if args.architecture=='siglip':
        args.tag='siglip2_so400m_webli'
        if args.attribute_attention:raise ValueError('SigLIP attribute-query variant is not implemented')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists() and not args.resume:raise ValueError('Run already exists')
    head_state=torch.load(args.head_checkpoint,map_location='cpu',weights_only=False)
    if args.architecture=='dino':
        from dino_attributes import DinoAttributes
        model=DinoAttributes(args.weights,(args.height,args.width),head_state['regional'],attribute_attention=args.attribute_attention,variant=args.dino_variant)
    elif args.architecture=='siglip':
        from siglip_attributes import SiglipAttributes
        model=SiglipAttributes(args.weights,(args.height,args.width),head_state['regional'])
    elif args.architecture=='solider':
        from solider_attributes import SoliderAttributes
        args.tag='solider_swin_base_lup'
        model=SoliderAttributes(args.weights,(args.height,args.width),head_state['regional'],semantic_weight=args.semantic_weight,drop_path=args.drop_path)
    else:
        n_sources=len(set(p.split('/')[0] for p in read_annotations('train',Path(args.split_root)/'train.csv')[1])) if args.source_heads else 0
        model=ClipLarge(args.tag,args.weights,head_state['regional'],size=(args.height,args.width),name=args.timm_name,drop_path=args.drop_path,source_heads=n_sources)
    missing=model.head.load_state_dict(head_state['head'],strict=not args.attribute_attention)
    if args.attribute_attention:
        assert missing.missing_keys==['queries'] and not missing.unexpected_keys
    initialization=None
    if args.initialize_checkpoint:
        parent=torch.load(args.initialize_checkpoint,map_location='cpu',weights_only=False)
        source=parent['ema'];old_args=parent['config']['args']
        old_size=(old_args.get('height',224),old_args.get('width',224))
        if old_size!=(args.height,args.width):
            if args.architecture!='dino':raise ValueError('Resolution transfer currently supports DINO only')
            if args.dino_variant=='large_v3':raise ValueError('DINOv3 resolution transfer needs explicit rotary-buffer handling')
            from timm.layers import resample_abs_pos_embed
            grid=(args.height//14,args.width//14)
            prefixes=model.backbone.pos_embed.shape[1]-grid[0]*grid[1]
            source['backbone.pos_embed']=resample_abs_pos_embed(source['backbone.pos_embed'],list(grid),
                old_size=[old_size[0]//14,old_size[1]//14],num_prefix_tokens=prefixes)
            source['pool_weights']=model.pool_weights
        missing=model.load_state_dict(source,strict=not args.attribute_attention)
        if args.attribute_attention:
            assert missing.missing_keys in [[],['head.queries']] and not missing.unexpected_keys
        with open(args.initialize_checkpoint,'rb') as f:parent_sha=hashlib.file_digest(f,'sha256').hexdigest()
        initialization=dict(checkpoint=args.initialize_checkpoint,sha256=parent_sha,parent_epoch=parent['epoch']+1,
                            original_size=old_size,source_config=parent['config'])
        del source,parent
    model.backbone.requires_grad_(False)
    if not 1<=args.last_blocks<=len(model.backbone.blocks):raise ValueError('Invalid block count')
    for block in model.backbone.blocks[-args.last_blocks:]:block.requires_grad_(True)
    model.backbone.norm.requires_grad_(True)
    if args.source_heads:
        for h in model.source_heads:h.load_state_dict(model.head.state_dict())
    model.cuda();ema=copy.deepcopy(model).eval();ema.requires_grad_(False)
    anchor=None
    if args.feature_anchor:
        if args.architecture!='clip':raise ValueError('Feature anchoring is implemented for the CLIP architecture only')
        anchor=copy.deepcopy(model).eval();anchor.requires_grad_(False)
    root=Path(args.split_root)
    train=UPAR('train','data',(args.height,args.width),train=True,annotations=root/'train.csv')
    if args.perspective_probability>0:
        from data import ElevationWarp
        train.transform.transforms.insert(0,ElevationWarp(args.perspective_probability,args.perspective_strength))
    if args.lowres_probability is not None:
        from data import SurveillanceDegradation
        for i,op in enumerate(train.transform.transforms):
            if isinstance(op,SurveillanceDegradation):train.transform.transforms[i]=SurveillanceDegradation(args.lowres_probability,.25)
    if args.mild_augmentation:
        from torchvision import transforms as T
        from data import SurveillanceDegradation
        train.transform=T.Compose([T.Resize((args.height,args.width),interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(),T.RandomApply([T.ColorJitter(brightness=.1,contrast=.1,saturation=.05)],p=.5),
            SurveillanceDegradation(),T.ToTensor(),T.Normalize([.5]*3,[.5]*3),
            T.RandomErasing(p=.1,scale=(.015,.06),ratio=(.5,2),value=0)])
    if args.shift_augmentation:
        from torchvision import transforms as T
        from data import SurveillanceDegradation
        train.transform=T.Compose([T.Resize((args.height,args.width),interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.RandomAffine(degrees=5,translate=(.06,.04),scale=(.88,1.12),interpolation=T.InterpolationMode.BILINEAR,fill=127)],p=.6),
            T.RandomApply([T.ColorJitter(brightness=.35,contrast=.35,saturation=.15)],p=.7),
            SurveillanceDegradation(),T.ToTensor(),T.Normalize([.5]*3,[.5]*3),
            T.RandomErasing(p=.2,scale=(.015,.1),ratio=(.3,3.3),value=0)])
    val=UPAR('tune','data',(args.height,args.width),annotations=root/'tune.csv')
    calibration_folds=None
    if args.selection_metric=='Calibrated_Joint':
        groups=json.loads((root/'groups.json').read_text())
        calibration_folds=np.array([int(hashlib.sha256(('joint-grouped-2123/'+groups[path]).encode()).hexdigest()[:8],16)%3 for path in val.images])
    assert train.names==val.names and (args.tune_in_train or not(set(train.images)&set(val.images)))
    sampler=EpochSampler(len(train),args.seed)
    tl=DataLoader(train,batch_size=args.batch_size,sampler=sampler,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
    steps_per_epoch=math.ceil(len(train)/args.batch_size)
    vl=DataLoader(val,batch_size=args.batch_size,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
    counts={d:int((train.domains==d).sum()) for d in np.unique(train.domains)}
    dw=np.array([counts[d]**-args.domain_power for d in train.domains],dtype=np.float32);dw/=dw.mean()
    rate=torch.tensor(np.average(train.labels,axis=0,weights=dw),device='cuda',dtype=torch.float32)
    positive_weight,negative_weight=bce_polarity_weights(rate,args.balance_power,args.inverse_balance_power)
    domain_weights=torch.from_numpy(dw).cuda()
    unique_domains=sorted(counts)
    domain_ids=torch.tensor([unique_domains.index(d) for d in train.domains],device='cuda')
    source_bias=None
    if args.source_prior:
        # Initialize each source's bias at its prevalence log-odds relative to the pooled (domain-weighted) rate.
        labels=torch.from_numpy(train.labels).cuda().clamp(0,1)
        pooled=torch.logit(rate.clamp(.002,.998))
        init=torch.stack([torch.logit(labels[domain_ids==k].mean(0).clamp(.002,.998))-pooled for k in range(len(unique_domains))])
        source_bias=torch.nn.Parameter(init-init.mean(0,keepdim=True))
    head=[v for n,v in model.head.named_parameters() if n!='queries']
    if args.source_heads:head=list(model.source_heads.parameters())  # the shared head is unused when training per-source heads
    depth=len(model.backbone.blocks);parameter_groups=[]
    if args.layer_decay==1:
        encoder=[p for p in model.backbone.parameters() if p.requires_grad]
        parameter_groups.append(dict(params=encoder,lr=args.encoder_lr,base_lr=args.encoder_lr))
    else:
        # Block i gets encoder_lr*decay**(depth-1-i); final norm uses the top rate.
        layers={}
        for name,v in model.backbone.named_parameters():
            if not v.requires_grad:continue
            i=int(name.split('.')[1]) if name.startswith('blocks.') else depth-1
            layers.setdefault(i,[]).append(v)
        for i,params in sorted(layers.items()):
            lr=args.encoder_lr*args.layer_decay**(depth-1-i)
            parameter_groups.append(dict(params=params,lr=lr,base_lr=lr))
    parameter_groups.append(dict(params=head,lr=args.head_lr,base_lr=args.head_lr))
    if args.attribute_attention:
        lr=args.head_lr*args.query_lr_mult
        parameter_groups.append(dict(params=[model.head.queries],lr=lr,base_lr=lr))
    if source_bias is not None:parameter_groups.append(dict(params=[source_bias],lr=1e-3,base_lr=1e-3,weight_decay=0.))
    optim=torch.optim.AdamW(parameter_groups,weight_decay=.01)

    def ema_state():
        # Inference uses the shared score plus the mean source bias, folded into the global head bias.
        state={k:v.clone() for k,v in ema.state_dict().items()}
        if source_bias is not None:state['head.global_head.bias']+=source_bias.detach().mean(0)
        return state

    def evaluate_ema():
        if source_bias is None:return evaluate(ema,vl)
        with torch.no_grad():ema.head.global_head.bias+=source_bias.mean(0)
        try:return evaluate(ema,vl)
        finally:
            with torch.no_grad():ema.head.global_head.bias-=source_bias.mean(0)
    config=dict(args=vars(args),attributes=train.names,regional=head_state['regional'],
                train_images=len(train),tune_images=len(val),domain_counts=counts,
                domain_loss_weighting=f'inverse source frequency to power {args.domain_power}; every row sampled once per epoch',
                pretrained=head_state['cache_manifest'],initialization=initialization,
                loss=dict(class_balance_power=args.balance_power,inverse_balance_power=args.inverse_balance_power,soft_domain_f1_weight=args.f1_loss_weight,mixup_alpha=args.mixup_alpha,mixup_probability=args.mixup_probability,perspective_probability=args.perspective_probability,perspective_strength=args.perspective_strength),
                selection_metric=args.selection_metric,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
                epoch_calibration=('Three known-group tuning folds, joint-min threshold objective, fixed 0.75 shrinkage; audit excluded.' if calibration_folds is not None else None),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                total_parameters=sum(p.numel() for p in model.parameters()),
                annotation_sha256={s:hashlib.sha256((root/(s+'.csv')).read_bytes()).hexdigest() for s in ['train','tune']})
    (out/'config.json').write_text(json.dumps(config,indent=2));print(json.dumps(config),flush=True)
    best=-1;global_step=0;start_epoch=0;start_batch=0;partial_loss=0.;partial_seconds=0.
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);ema.load_state_dict(state['ema']);optim.load_state_dict(state['optimizer'])
        if source_bias is not None:
            with torch.no_grad():
                source_bias.copy_(state['source_bias'].cuda());ema.head.global_head.bias-=source_bias.mean(0)
        best=state['best'];global_step=state['global_step'];start_epoch=state['epoch']+1
        start_batch=state.get('next_batch',0);partial_loss=state.get('partial_loss',0.);partial_seconds=state.get('partial_seconds',0.)
    else:
        logits,metrics=evaluate_ema();best=selection_value(metrics,args.selection_metric,logits,val,calibration_folds)
        (out/'initial_metrics.json').write_text(json.dumps(metrics,indent=2))
        save(dict(model=model.state_dict(),ema=ema_state(),**({'source_bias':source_bias.detach().cpu()} if source_bias is not None else {}),optimizer=optim.state_dict(),epoch=-1,best=best,global_step=0,config=config),out/'best.pt')
        np.savez_compressed(out/'best_tune.npz',paths=np.array(val.images),labels=val.labels,logits=logits)
        print(json.dumps(dict(epoch=0,score=best)),flush=True)
    total=args.epochs*steps_per_epoch
    for epoch in range(start_epoch,args.epochs):
        started=time.monotonic();model.train();loss_sum=partial_loss
        sampler.epoch=epoch;sampler.start=start_batch*args.batch_size
        for block in model.backbone.blocks[:-args.last_blocks]:block.eval()
        for step,(x,y,ii) in enumerate(tl,start=start_batch):
            warmup=min(1.,(global_step+1)/(steps_per_epoch*.5))
            progress=max(0.,(global_step-steps_per_epoch*.5)/max(total-steps_per_epoch*.5,1))
            scale=warmup*(.05+.95*.5*(1+math.cos(math.pi*progress)))
            for group in optim.param_groups:group['lr']=group['base_lr']*scale
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);ii=ii.cuda(non_blocking=True)
            batch_domain_weights=domain_weights[ii]
            if args.mixup_alpha>0 and random.random()<args.mixup_probability:
                lam=float(np.random.beta(args.mixup_alpha,args.mixup_alpha));perm=torch.randperm(len(x),device='cuda')
                x=lam*x+(1-lam)*x[perm];y=lam*y+(1-lam)*y[perm];batch_domain_weights=lam*batch_domain_weights+(1-lam)*batch_domain_weights[perm]
            optim.zero_grad(set_to_none=True)
            full=(x,y,ii,batch_domain_weights);chunk=math.ceil(len(x)/args.micro_batches);loss_value=0.
            for c0 in range(0,len(full[0]),chunk):
                x,y,ii,batch_domain_weights=(t[c0:c0+chunk] for t in full);share=len(x)/len(full[0])
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    if anchor is not None:
                        features=model.features(x);logits=model.head(features)
                        with torch.no_grad():reference=anchor.features(x)
                        drift=((features-reference)**2).sum(-1).mean()
                    elif args.source_heads:
                        features=model.features(x);every=torch.stack([h(features) for h in model.source_heads])
                        logits=every[domain_ids[ii],torch.arange(len(ii),device='cuda')]
                    else:logits=model(x)
                    if source_bias is not None:logits=logits+source_bias[domain_ids[ii]]
                    weights=y*positive_weight+(1-y)*negative_weight
                    row_loss=(F.binary_cross_entropy_with_logits(logits.float(),y,reduction='none')*weights).mean(1)
                    loss=(row_loss*batch_domain_weights).mean()
                    if anchor is not None:loss=loss+args.feature_anchor*drift
                    if args.f1_loss_weight:
                        probabilities=logits.float().sigmoid();intersection=(probabilities*y).sum(1)
                        precision=intersection/probabilities.sum(1).clamp_min(1)
                        recall=intersection/y.sum(1).clamp_min(1)
                        ids=domain_ids[ii];n=torch.bincount(ids,minlength=len(unique_domains)).float()
                        p=torch.zeros_like(n).scatter_add_(0,ids,precision)/n.clamp_min(1)
                        r=torch.zeros_like(n).scatter_add_(0,ids,recall)/n.clamp_min(1)
                        f1=2*p*r/(p+r).clamp_min(1e-6);present=(n>0).float()
                        loss=loss+args.f1_loss_weight*((1-f1)*present).sum()/present.sum().clamp_min(1)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                (loss*share).backward();loss_value+=loss.item()*share
            loss=torch.tensor(loss_value);torch.nn.utils.clip_grad_norm_(model.parameters(),5);optim.step()
            with torch.no_grad():
                decay=min(.999,(1+global_step)/(10+global_step))
                for ep,mp in zip(ema.parameters(),model.parameters()):
                    if mp.requires_grad:ep.lerp_(mp,1-decay)
            global_step+=1;loss_sum+=loss.item()
            if step%100==0:print(json.dumps(dict(epoch=epoch+1,step=step,steps=steps_per_epoch,loss=loss.item(),drift=float(drift) if anchor is not None else None,seconds=partial_seconds+time.monotonic()-started)),flush=True)
            if (step+1)%200==0:
                save(dict(model=model.state_dict(),ema=ema_state(),**({'source_bias':source_bias.detach().cpu()} if source_bias is not None else {}),optimizer=optim.state_dict(),epoch=epoch-1,best=best,
                    global_step=global_step,config=config,next_batch=step+1,partial_loss=loss_sum,partial_seconds=partial_seconds+time.monotonic()-started),out/'resume.pt')
        logits,metrics=evaluate_ema();value=selection_value(metrics,args.selection_metric,logits,val,calibration_folds);improved=value>best;best=max(best,value)
        result=dict(epoch=epoch+1,loss=loss_sum/steps_per_epoch,seconds=partial_seconds+time.monotonic()-started,**metrics)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        print(json.dumps(dict(epoch=epoch+1,score=value,seconds=result['seconds'])),flush=True)
        state=dict(model=model.state_dict(),ema=ema_state(),**({'source_bias':source_bias.detach().cpu()} if source_bias is not None else {}),optimizer=optim.state_dict(),epoch=epoch,best=best,global_step=global_step,config=config)
        save(state,out/'last.pt')
        if args.save_epochs:save(dict(ema=ema_state(),**({'source_bias':source_bias.detach().cpu()} if source_bias is not None else {}),epoch=epoch,config=config),out/f'epoch{epoch+1}.pt')
        save(state,out/'resume.pt')
        if improved:
            save(state,out/'best.pt');np.savez_compressed(out/'best_tune.npz',paths=np.array(val.images),labels=val.labels,logits=logits)
            (out/'best_metrics.json').write_text(json.dumps(result,indent=2))
        start_batch=0;partial_loss=0.;partial_seconds=0.
    (out/'complete.json').write_text(json.dumps(dict(epochs=args.epochs,best=best),indent=2))


if __name__=='__main__':main()
