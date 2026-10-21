"""Cache only train/tuning embeddings for cheap, controlled classifier trials."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from clip_large import ClipLarge
from data import UPAR


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True)
    p.add_argument('--tag',default='apple_mclip2_dfndr2b');p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--architecture',choices=['clip','dino','siglip','solider'],default='clip')
    p.add_argument('--dino-variant',choices=['base','large','large_v3'],default='base')
    p.add_argument('--height',type=int,default=224);p.add_argument('--width',type=int,default=224)
    p.add_argument('--name');p.add_argument('--checkpoint',help='Cache features of a fine-tuned EMA model instead of generic weights')
    p.add_argument('--split-root',default='data/splits/train90_v1',help='Directory containing train.csv and tune.csv')
    p.add_argument('--inference-size',type=int,nargs=2,metavar=('HEIGHT','WIDTH'),help='Resize a loaded CLIP backbone and its regional pooling for feature extraction')
    p.add_argument('--pure-half',action='store_true',help='Use the deployment FP16 backbone instead of BF16 autocast')
    args=p.parse_args()
    if args.architecture=='dino':args.tag='dinov3_lvd1689m' if args.dino_variant=='large_v3' else 'dinov2_lvd142m'
    if args.architecture=='siglip':args.tag='siglip2_so400m_webli'
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    if args.architecture=='dino':
        from dino_attributes import DinoAttributes
        model=DinoAttributes(args.weights,(args.height,args.width),variant=args.dino_variant)
    elif args.architecture=='siglip':
        from siglip_attributes import SiglipAttributes
        model=SiglipAttributes(args.weights,(args.height,args.width))
    elif args.architecture=='solider':
        from solider_attributes import SoliderAttributes
        args.tag='solider_swin_base_lup'
        model=SoliderAttributes(args.weights,(args.height,args.width))
    else:model=ClipLarge(args.tag,args.weights,size=(args.height,args.width),name=args.name)
    checkpoint_provenance=None
    if args.checkpoint:
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        a=state['config']['args']
        assert a.get('architecture','clip')==args.architecture
        assert (a.get('height',224),a.get('width',224))==(args.height,args.width)
        if args.architecture=='dino':assert a.get('dino_variant','base')==args.dino_variant
        if a.get('attribute_attention',False):raise ValueError('Attention pooling has no five-region cache representation')
        model.load_state_dict(state['ema'],strict=True)
        with open(args.checkpoint,'rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        checkpoint_provenance=dict(path=args.checkpoint,sha256=digest,selected_epoch=state['epoch']+1,training_recipe=state['config'])
        del state
    size=tuple(args.inference_size or (args.height,args.width))
    if args.inference_size:
        if args.architecture!='clip':raise ValueError('Inference-size override currently supports CLIP backbones only')
        from resolution_probe import resize_model
        resize_model(model,size)
    model=model.cuda().eval()
    if args.pure_half:
        model.backbone.half();model.mean=model.mean.half();model.std=model.std.half()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    manifest=dict(args=vars(args),parameters=sum(p.numel() for p in model.parameters()),splits={})
    if checkpoint_provenance:manifest['fine_tuned_checkpoint']=checkpoint_provenance
    with open(args.weights,'rb') as f:manifest['pretrained_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    for split in ['train','tune']:
        annotation=Path(args.split_root)/(split+'.csv')
        ds=UPAR(split,'data',size,annotations=annotation)
        file=out/(split+'_features.npy')
        if file.exists():raise ValueError('Refusing to overwrite an existing cache')
        features=np.lib.format.open_memmap(file,mode='w+',dtype=np.float16,shape=(len(ds),5,model.head.dim))
        loader=DataLoader(ds,batch_size=args.batch_size,num_workers=args.workers,pin_memory=True)
        started=time.monotonic()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=not args.pure_half):
            for step,(x,_,ii) in enumerate(loader):
                x=x.cuda(non_blocking=True)
                result=model.features(x.half() if args.pure_half else x).cpu().numpy()
                if not np.isfinite(result).all():raise ValueError('Nonfinite cached features')
                features[ii.numpy()]=result
                if step%50==0:print(json.dumps(dict(split=split,images=int(ii[-1])+1,total=len(ds),seconds=time.monotonic()-started)),flush=True)
        features.flush();del features
        np.savez_compressed(out/(split+'_labels.npz'),paths=np.array(ds.images),labels=ds.labels,names=ds.names)
        manifest['splits'][split]=dict(images=len(ds),seconds=time.monotonic()-started,annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest())
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
