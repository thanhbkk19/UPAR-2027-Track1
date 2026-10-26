"""Organizer annotations, image loading and training augmentation."""
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
SOURCES = ('Market1501', 'PA100k', 'PETA')


def read_annotations(path):
    with Path(path).open() as f:
        reader = csv.reader(f)
        names = next(reader)[1:]
        rows = list(reader)
    return names, [r[0] for r in rows], np.array([r[1:] for r in rows], dtype=np.float32)


def source_of(path):
    return str(path).split('/')[0]


def source_mean_prior(path):
    """Training prior used by the decision rule: per-source prevalence averaged over sources."""
    _, images, labels = read_annotations(path)
    domains = np.array([source_of(p) for p in images])
    return np.mean([labels[domains == d].mean(0) for d in np.unique(domains)], 0).clip(1e-4, 1-1e-4)


def resolve_image(name, root):
    path = Path(root)/name
    if path.is_file():
        return path
    # PA100k releases differ only in directory nesting.
    if name.startswith('PA100k/'):
        for prefix in ['PA100k/release_data/release_data', 'PA100k/release_data', 'PA100k/images', 'PA100k']:
            path = Path(root)/prefix/Path(name).name
            if path.is_file():
                return path
    raise FileNotFoundError(f'Missing organizer image: {name} under {root}')


class SurveillanceDegradation:
    """Label-preserving low resolution, blur and JPEG artefacts."""
    def __call__(self, image):
        if random.random() < .25:
            size = image.size
            scale = random.uniform(.4, .85)
            image = image.resize((max(8, int(size[0]*scale)), max(16, int(size[1]*scale))),
                                 Image.Resampling.BILINEAR).resize(size, Image.Resampling.BILINEAR)
        if random.random() < .12:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(.1, 1.)))
        if random.random() < .12:
            buf = io.BytesIO()
            image.save(buf, 'JPEG', quality=random.randint(35, 90))
            buf.seek(0)
            image = Image.open(buf).convert('RGB')
        return image


class UPAR(Dataset):
    """Images from an annotation CSV, resized to (height, width) and scaled to [-1, 1]."""
    def __init__(self, annotations, size, train=False, root=ROOT/'data'):
        self.names, self.images, self.labels = read_annotations(annotations)
        self.domains = np.array([source_of(p) for p in self.images])
        self.paths = [resolve_image(p, root) for p in self.images]
        ops = [T.Resize(size, interpolation=T.InterpolationMode.BILINEAR)]
        if train:
            ops += [T.RandomHorizontalFlip(),
                    T.RandomApply([T.ColorJitter(brightness=.2, contrast=.2, saturation=.12)], p=.7),
                    SurveillanceDegradation()]
        ops += [T.ToTensor(), T.Normalize((.5,)*3, (.5,)*3)]
        if train:
            ops += [T.RandomErasing(p=.2, scale=(.015, .12), ratio=(.3, 3.3), value=0)]
        self.transform = T.Compose(ops)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            pixels = self.transform(image.convert('RGB'))
        return pixels, torch.from_numpy(self.labels[index]), index


class DeployedImages(Dataset):
    """Same preprocessing as runtime/run.py: PIL bilinear resize of uint8 RGB, then [-1, 1]."""
    def __init__(self, paths, size, root=ROOT/'data'):
        self.paths = [resolve_image(str(p), root) for p in paths]
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            a = np.array(image.convert('RGB').resize((self.size[1], self.size[0]), Image.Resampling.BILINEAR))
        return torch.from_numpy(a).permute(2, 0, 1).contiguous().float().div_(255).sub_(.5).div_(.5), index
