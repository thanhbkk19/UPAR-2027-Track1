"""Load a submission ZIP through the organizer API (load_model / predict_batch) and score it."""
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

from data import ROOT, read_annotations, resolve_image
from metrics import score


def api_checks(runtime, names, root):
    """Batch/single consistency, empty input, attribute reordering and metadata independence."""
    rng = np.random.default_rng(2027)
    pictures = []
    for i, shape in enumerate([(273, 111, 3), (181, 91, 3)]):
        pictures.append(root/f'contract{i}.png')
        Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8)).save(pictures[-1])
    a, b = (dict(image_path=str(p), attribute_names=names, index=0, image='x.png') for p in pictures)
    single = np.asarray(runtime.predict_image(a))
    assert single.shape == (40,) and runtime.predict_batch([]) == []
    mixed = np.asarray(runtime.predict_batch([a if i % 2 else b for i in range(runtime.BATCH_SIZE+1)]))
    expected = np.stack([single if i % 2 else np.asarray(runtime.predict_image(b)) for i in range(runtime.BATCH_SIZE+1)])
    np.testing.assert_allclose(mixed, expected, rtol=.02, atol=.005)
    np.testing.assert_allclose(runtime.predict_image(dict(a, attribute_names=names[::-1])), single[::-1], rtol=.01, atol=.003)
    np.testing.assert_allclose(runtime.predict_image(dict(a, image='other/domain.png', index=999)), single, rtol=5e-4, atol=1e-4)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True)
    p.add_argument('--annotations', default=str(ROOT/'data/splits/release/audit.csv'))
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int, default=0, help='Score only the first N images (smoke test)')
    args = p.parse_args()
    import torch
    with tempfile.TemporaryDirectory(prefix='upar_eval_') as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(args.archive) as z:
            if any(not (root/n).resolve().is_relative_to(root.resolve()) for n in z.namelist()):
                raise ValueError('Unsafe archive path')
            z.extractall(root)
        for name, expected in json.loads((root/'manifest.json').read_text()).items():
            with (root/name).open('rb') as f:
                if hashlib.file_digest(f, 'sha256').hexdigest() != expected:
                    raise ValueError(f'Manifest mismatch: {name}')
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location('submission_runtime', root/'run.py')
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        runtime.load_model()
        names = json.loads((root/'assets/config.json').read_text())['attribute_names']
        api_checks(runtime, names, root)
        header, images, labels = read_annotations(args.annotations)
        if header != names:
            raise ValueError('Annotation and model attribute order differ')
        if args.limit:
            images, labels = images[:args.limit], labels[:args.limit]
        torch.cuda.reset_peak_memory_stats()
        started, predictions = time.monotonic(), []
        for start in range(0, len(images), runtime.BATCH_SIZE):
            predictions += runtime.predict_batch([dict(index=i, image=im, attribute_names=names,
                                                       image_path=str(resolve_image(im, ROOT/'data').resolve()))
                                                  for i, im in enumerate(images[start:start+runtime.BATCH_SIZE], start)])
        elapsed = time.monotonic()-started
    probs = np.asarray(predictions)
    result = dict(api_checks='PASS', **score(labels, probs, np.array([q.split('/')[0] for q in images])),
                  images=len(images), images_per_second=len(images)/elapsed,
                  peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    with open(args.archive, 'rb') as f:
        result['archive_sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    np.savez_compressed(Path(args.output).with_suffix('.npz'), paths=np.array(images), labels=labels, probabilities=probs)
    print(json.dumps(dict(archive_sha256=result['archive_sha256'], images=len(images),
                          **{k: round(result['mean'][k], 6) for k in ['mA', 'Instance_F1', 'Challenge_Avg']})))


if __name__ == '__main__':
    main()
