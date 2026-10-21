"""Validate the actual extracted ZIP API and optionally evaluate public images."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import zipfile

import numpy as np
from PIL import Image

from metrics import score


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--data')
    p.add_argument('--annotations',default='starterkit/data/annotations/task1/val/gt.csv')
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    import torch
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='upar_validate_') as tmp:
        root=Path(tmp)
        with zipfile.ZipFile(args.archive) as z:
            if 'run.py' not in z.namelist() or 'metadata.yaml' not in z.namelist():
                raise ValueError('Entry files must be at the ZIP root')
            for name in z.namelist():
                dest=(root/name).resolve()
                if not dest.is_relative_to(root.resolve()):raise ValueError('Unsafe archive path')
            bad=z.testzip()
            if bad:raise ValueError(f'Corrupt member: {bad}')
            z.extractall(root)
        config=json.loads((root/'assets/config.json').read_text());names=config['attribute_names']
        if (root/'manifest.json').is_file():
            for name,expected in json.loads((root/'manifest.json').read_text()).items():
                with (root/name).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
                if actual!=expected:raise ValueError(f'Manifest mismatch: {name}')
        sys.path.insert(0,str(root))
        spec=importlib.util.spec_from_file_location('upar_submission_runtime',root/'run.py')
        runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
        runtime.load_model()
        rng=np.random.default_rng(2027)
        picture=root/'contract.png';Image.fromarray(rng.integers(0,256,(273,111,3),dtype=np.uint8)).save(picture)
        sample=dict(image_path=str(picture),attribute_names=names,index=0,image='irrelevant/path.png')
        single=np.asarray(runtime.predict_image(sample))
        batch=np.asarray(runtime.predict_batch([sample]*3))
        assert single.shape==(40,) and batch.shape==(3,40)
        assert np.isfinite(batch).all() and batch.min()>=0 and batch.max()<=1
        assert runtime.predict_batch([])==[]
        # Cross an internal batch boundary with genuinely different images.
        boundary = runtime.BATCH_SIZE+1 if torch.cuda.is_available() else 9
        other = root/'contract_other.png'
        Image.fromarray(rng.integers(0,256,(181,91,3),dtype=np.uint8)).save(other)
        other_sample=dict(sample,image_path=str(other))
        mixed=[sample if i%2 else other_sample for i in range(boundary)]
        mixed_output=np.asarray(runtime.predict_batch(mixed))
        other_single=np.asarray(runtime.predict_image(other_sample))
        expected=np.stack([single if i%2 else other_single for i in range(boundary)])
        np.testing.assert_allclose(mixed_output,expected,rtol=.02,atol=.005)
        reordered=np.asarray(runtime.predict_image(dict(sample,attribute_names=names[::-1])))
        np.testing.assert_allclose(reordered,single[::-1],rtol=0.01,atol=0.003)
        np.testing.assert_allclose(batch,np.tile(single,(3,1)),rtol=0.01,atol=0.003)
        # Path metadata and row index must have no effect on predictions.
        renamed=np.asarray(runtime.predict_image(dict(sample,image='different/domain.png',index=999)))
        # JIT warmup may change fusion kernels between early calls. Permit only
        # small FP16 numerical noise, not a domain-specific prediction change.
        np.testing.assert_allclose(renamed,single,rtol=5e-4,atol=1e-4)
        result=dict(status='PASS',torch=torch.__version__,cuda=torch.cuda.is_available(),
                    max_batch_difference=float(np.abs(batch-single).max()),
                    max_metadata_difference=float(np.abs(renamed-single).max()),
                    contract_seconds=time.monotonic()-started,
                    note='Interface check only; local GPU timing does not establish A10 runtime.')
        if args.data:
            import csv
            with open(args.annotations) as f:
                reader=csv.reader(f);header=next(reader);rows=list(reader)
            if header[1:]!=names:raise ValueError('Annotation and model label order differ')
            if args.limit:rows=rows[:args.limit]
            predictions=[];eval_started=time.monotonic()
            for start in range(0,len(rows),runtime.BATCH_SIZE):
                samples=[]
                for index,row in enumerate(rows[start:start+runtime.BATCH_SIZE],start):
                    path=Path(args.data)/row[0]
                    if not path.is_file() and row[0].startswith('PA100k/'):
                        for prefix in ['PA100k/release_data/release_data','PA100k/release_data','PA100k/images','PA100k']:
                            candidate=Path(args.data)/prefix/Path(row[0]).name
                            if candidate.is_file():path=candidate;break
                    samples.append(dict(index=index,image=row[0],image_path=str(path.resolve()),attribute_names=names))
                predictions.extend(runtime.predict_batch(samples))
                if start%4096==0:print(f'validated {len(predictions)}/{len(rows)}',flush=True)
            elapsed=time.monotonic()-eval_started
            labels=np.array([r[1:] for r in rows],dtype=np.float32)
            paths=np.array([r[0] for r in rows]);probs=np.asarray(predictions)
            result.update(score(labels,probs,np.array([p.split('/')[0] for p in paths])))
            result.update(images=len(rows),evaluation_seconds=elapsed,images_per_second=len(rows)/elapsed)
            np.savez_compressed(Path(args.output).with_suffix('.npz'),paths=paths,labels=labels,probabilities=probs)
        with open(args.archive,'rb') as f:result['archive_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
        if torch.cuda.is_available():
            result['peak_gpu_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
            result['peak_gpu_reserved_mib']=torch.cuda.max_memory_reserved()/1024**2
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
