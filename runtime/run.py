"""Offline, globally weighted image ensemble for UPAR Track 1."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path

BATCH_SIZE = 512
HERE = Path(__file__).resolve().parent
_CONFIG = None
_MODELS = []


def _size(value):return (value,value) if isinstance(value,int) else tuple(value)


def load_model():
    global _CONFIG, _MODELS, _POOL, _DEVICE, _DTYPE, _CALIBRATION
    import torch
    _CONFIG = json.loads((HERE/'assets/config.json').read_text())
    assert len(_CONFIG['attribute_names']) == len(set(_CONFIG['attribute_names'])) == 40
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    _DEVICE = torch.device(os.environ.get('UPAR_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
    _DTYPE = torch.float16 if _DEVICE.type == 'cuda' else torch.float32
    _MODELS = []
    for c in _CONFIG['models']:
        if c.get('half',False):
            # Traced with a pure FP16 backbone; stored dtypes are kept as exported.
            if _DEVICE.type != 'cuda':raise RuntimeError('FP16-traced models require CUDA')
            model = torch.jit.load(str(HERE/c['file']), map_location=_DEVICE).eval()
            if c.get('freeze',False):model=torch.jit.freeze(model,optimize_numerics=False)
            _MODELS.append((model,torch.tensor(c['logit_offset'],device=_DEVICE,dtype=torch.float32),c))
            continue
        dtype = torch.float32 if c.get('amp',False) else _DTYPE
        model = torch.jit.load(str(HERE/c['file']), map_location=_DEVICE).eval().to(dtype=dtype)
        # Freeze only after storage tensors have been converted for the target
        # device. Disable numerical rewrites such as convolution/BN folding.
        if c.get('freeze',False):model=torch.jit.freeze(model,optimize_numerics=False)
        offset = torch.tensor(c['logit_offset'],device=_DEVICE,dtype=torch.float32)
        _MODELS.append((model,offset,c))
    assert all(c['weight'] >= 0 for _,_,c in _MODELS)
    assert abs(sum(c['weight'] for _,_,c in _MODELS)-1) < 1e-6
    _CALIBRATION = torch.tensor(_CONFIG.get('logit_offset',[0.]*40),device=_DEVICE,dtype=torch.float32)
    _POOL = ThreadPoolExecutor(max_workers=min(12,os.cpu_count() or 1))


def _read(sample):
    import numpy as np
    from PIL import Image
    with Image.open(sample['image_path']) as image:
        image = image.convert('RGB')
        sizes=dict.fromkeys(_size(c['image_size']) for _,_,c in _MODELS)
        return {size:np.array(image.resize((size[1],size[0]),Image.Resampling.BILINEAR),dtype=np.uint8) for size in sizes}


def predict_batch(samples):
    if not samples:return []
    if _CONFIG is None:load_model()
    import numpy as np
    import torch
    names = _CONFIG['attribute_names']; known = set(names)
    for sample in samples:
        if len(sample['attribute_names']) != 40 or set(sample['attribute_names']) != known:
            raise ValueError('Unknown or duplicate attribute names')
    rows=[]
    internal=int(_CONFIG.get('gpu_microbatch',128)) if _DEVICE.type=='cuda' else 8
    if not 1<=internal<=512:raise ValueError('GPU microbatch must be between 1 and 512')
    # Profiling-based JIT fusion changes fp16 rounding after warmup in these
    # supplied traces. Disable it to keep predictions stable across calls.
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        starts=list(range(0,len(samples),internal))
        # Decode the next chunk on CPU threads while the GPU runs the current one.
        pending=[_POOL.submit(_read,s) for s in samples[:internal]]
        for number,start in enumerate(starts):
            batch=samples[start:start+internal]
            images=[f.result() for f in pending]
            if number+1<len(starts):pending=[_POOL.submit(_read,s) for s in samples[starts[number+1]:starts[number+1]+internal]]
            probability=torch.zeros(len(batch),40,device=_DEVICE,dtype=torch.float32)
            inputs={}
            for model,offset,c in _MODELS:
                dtype = torch.float16 if c.get('half',False) else torch.float32 if c.get('amp',False) else _DTYPE
                key=(_size(c['image_size']),dtype)
                if key not in inputs:
                    array=np.stack([im[key[0]] for im in images])
                    # Preserve calibration's NCHW layout and CPU arithmetic.
                    # These graphs do not mutate their input tensors, so models
                    # with identical input geometry/precision can share pixels.
                    x=torch.from_numpy(array).permute(0,3,1,2).contiguous().float()
                    inputs[key]=x.div_(255).sub_(.5).div_(.5).to(_DEVICE,dtype=dtype)
                x=inputs[key]
                with torch.autocast(_DEVICE.type,dtype=torch.float16,enabled=_DEVICE.type=='cuda' and c.get('amp',False) and not c.get('half',False)):
                    logits=model(x).float()
                    if c.get('flip_tta',False):logits=(logits+model(x.flip(-1)).float())*.5
                probability.add_(torch.sigmoid(logits-offset),alpha=c['weight'])
            probability=probability.clamp(1e-7,1-1e-7)
            logits=torch.log(probability)-torch.log1p(-probability)
            outputs=torch.sigmoid(logits-_CALIBRATION).cpu().tolist()
            for sample,row in zip(batch,outputs):
                if len(row)!=40 or not all(math.isfinite(v) and 0<=v<=1 for v in row):
                    raise ValueError('Invalid model output')
                lookup=dict(zip(names,row))
                rows.append([lookup[n] for n in sample['attribute_names']])
    return rows


def predict_image(sample):
    return predict_batch([sample])[0]
