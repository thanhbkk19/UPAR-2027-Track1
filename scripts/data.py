"""Read only the organizer's explicitly enumerated train/validation images."""
import csv
import io
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]


def read_annotations(split, annotations=None):
    path = Path(annotations) if annotations else ROOT / 'starterkit/data/annotations/task1' / split / 'gt.csv'
    with path.open() as f:
        reader = csv.reader(f)
        names = next(reader)[1:]
        rows = list(reader)
    return names, [r[0] for r in rows], np.array([r[1:] for r in rows], dtype=np.float32)


def resolve_image(name, root):
    path = Path(root) / name
    if path.is_file():
        return path
    # PA100K downloads differ only in their directory nesting, not image content.
    if name.startswith('PA100k/'):
        for prefix in ['PA100k/release_data/release_data', 'PA100k/release_data',
                       'PA100k/images', 'PA100k']:
            path = Path(root) / prefix / Path(name).name
            if path.is_file():
                return path
    raise FileNotFoundError(f'Missing organizer image: {name} under {root}')


class SurveillanceDegradation:
    """Label-preserving resolution and compression variations, no hue shifts."""
    def __init__(self, downscale_probability=0.25, minimum_scale=0.4):
        self.downscale_probability = downscale_probability
        self.minimum_scale = minimum_scale

    def __call__(self, image):
        if random.random() < self.downscale_probability:
            size = image.size
            scale = random.uniform(self.minimum_scale, 0.85)
            image = image.resize((max(8, int(size[0]*scale)), max(16, int(size[1]*scale))),
                                 Image.Resampling.BILINEAR).resize(size, Image.Resampling.BILINEAR)
        if random.random() < 0.12:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 1.0)))
        if random.random() < 0.12:
            buf = io.BytesIO()
            image.save(buf, 'JPEG', quality=random.randint(35, 90))
            buf.seek(0)
            image = Image.open(buf).convert('RGB')
        return image


class UPAR(Dataset):
    def __init__(self, split, root, size=(256, 192), train=False, mean=(0.5,)*3,
                 std=(0.5,)*3, indices=None, annotations=None):
        self.names, self.images, self.labels = read_annotations(split, annotations)
        if indices is not None:
            self.images = [self.images[i] for i in indices]
            self.labels = self.labels[indices]
        self.domains = np.array([p.split('/')[0] for p in self.images])
        self.paths = [resolve_image(p, root) for p in self.images]
        ops = [T.Resize(size, interpolation=T.InterpolationMode.BILINEAR)]
        if train:
            ops += [T.RandomHorizontalFlip(),
                    T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.12)], p=0.7),
                    SurveillanceDegradation()]
        ops += [T.ToTensor(), T.Normalize(mean, std)]
        if train:
            ops += [T.RandomErasing(p=0.2, scale=(0.015, 0.12), ratio=(0.3, 3.3), value=0)]
        self.transform = T.Compose(ops)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            pixels = self.transform(image.convert('RGB'))
        return pixels, torch.from_numpy(self.labels[index]), index
