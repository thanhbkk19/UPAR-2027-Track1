"""Zero-shot yes/no attribute scores from a general VLM (Qwen3-VL, not trained for PAR) on organizer images.

Score = logit("Yes") - logit("No") for the first answer token. The image prefix is encoded once per image and the
questions are evaluated as a batch on a copy of its KV cache. Used only on organizer training/validation images.

Usage: vlm_attributes.py --csv split.csv --output scores.npz [--limit N] [--attributes conflict|all]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

QUESTIONS={
 'Age-Young':'Is this person a child or a teenager (younger than 18)?',
 'Age-Adult':'Is this person an adult between 18 and 60 years old?',
 'Age-Old':'Is this person elderly (older than 60)?',
 'Gender-Female':'Is this person a woman or a girl?',
 'Hair-Length-Long':'Does this person have long hair (reaching the shoulders or longer)?',
 'UpperBody-Length-Short':'Does this person wear short sleeves or no sleeves on the upper body?',
 'UpperBody-Color-Grey':'Is the upper-body clothing mainly grey?',
 'UpperBody-Color-Other':'Is the main colour of the upper-body clothing something other than black, blue, brown, green, grey, orange, pink, purple, red, white or yellow (for example beige, khaki or a multicoloured pattern)?',
 'LowerBody-Length-Short':'Does this person wear shorts or a short skirt that ends above the knee?',
 'LowerBody-Color-Grey':'Is the lower-body clothing mainly grey?',
 'LowerBody-Color-Other':'Is the main colour of the lower-body clothing something other than black, blue, brown, green, grey, orange, pink, purple, red, white or yellow (for example beige, khaki or a pattern)?',
 'LowerBody-Type-Skirt&Dress':'Is this person wearing a skirt or a dress?',
 'Accessory-Hat':'Is this person wearing a hat or a cap?',
 'Accessory-Bag':'Is this person carrying a handbag, shoulder bag or tote bag (not a backpack)?',
 'Accessory-Backpack':'Is this person carrying a backpack?',
 'Accessory-Glasses-Sun':'Is this person wearing sunglasses?',
}
CONFLICT=['Age-Young','Age-Adult','Age-Old','UpperBody-Length-Short','UpperBody-Color-Grey','UpperBody-Color-Other','LowerBody-Length-Short',
          'LowerBody-Color-Grey','LowerBody-Color-Other','LowerBody-Type-Skirt&Dress','Accessory-Hat','Accessory-Bag']


def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',default='artifacts/pretrained/qwen3_vl_8b');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--attributes',choices=['conflict','all'],default='conflict');p.add_argument('--seed',type=int,default=0);p.add_argument('--images-per-batch',type=int,default=1)
    p.add_argument('--min-side',type=int,default=224,help='Upscale small crops so the shorter side is at least this many pixels')
    a=p.parse_args()
    import sys;sys.path.insert(0,'scripts')
    from data import read_annotations,resolve_image
    from transformers import AutoProcessor,AutoModelForImageTextToText
    names,images,labels=read_annotations('x',a.csv)
    order=np.arange(len(images))
    if a.limit:order=np.random.default_rng(a.seed).permutation(len(images))[:a.limit]
    attrs=CONFLICT if a.attributes=='conflict' else list(QUESTIONS)
    proc=AutoProcessor.from_pretrained(a.model)
    model=AutoModelForImageTextToText.from_pretrained(a.model,dtype=torch.bfloat16,device_map='cuda').eval()
    tok=proc.tokenizer;yes=tok.encode('Yes',add_special_tokens=False)[0];no=tok.encode('No',add_special_tokens=False)[0]
    scores=np.zeros((len(order),len(attrs)),dtype=np.float32);started=time.monotonic()
    def load(i):
        with Image.open(resolve_image(images[i],'data')) as im:
            im=im.convert('RGB');s=a.min_side/min(im.size)
            return im.resize((round(im.size[0]*s),round(im.size[1]*s)),Image.Resampling.BICUBIC) if s>1 else im
    texts=[proc.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':QUESTIONS[q]+' Answer Yes or No.'}]}],
                                    tokenize=False,add_generation_prompt=True) for q in attrs]
    for start in range(0,len(order),a.images_per_batch):
        chunk=order[start:start+a.images_per_batch];ims=[load(i) for i in chunk]
        batch=proc(text=texts*len(chunk),images=[im for im in ims for _ in attrs],return_tensors='pt',padding=True,padding_side='left').to('cuda')
        with torch.inference_mode():
            logits=model(**batch).logits[:,-1].float()
        scores[start:start+len(chunk)]=(logits[:,yes]-logits[:,no]).view(len(chunk),len(attrs)).cpu().numpy()
        if start%(50*a.images_per_batch)==0:print(json.dumps(dict(done=start+len(chunk),total=len(order),seconds=round(time.monotonic()-started,1))),flush=True)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    np.savez(a.output,scores=scores,attributes=np.array(attrs),paths=np.array([images[i] for i in order]),
             labels=labels[order][:,[names.index(q) for q in attrs]],names=np.array(names))
    print(json.dumps(dict(output=a.output,images=len(order),seconds=round(time.monotonic()-started,1))),flush=True)


if __name__=='__main__':main()
