"""CPU checks for training inputs, release recipe and readout behavior."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tools'))
from clip_large import RegionHead
from robust_head_release import RobustCommittee
from common import check_hash, read_json


class FeatureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = RegionHead(True, 8)

    def features(self, x):
        return x


class ReproductionTests(unittest.TestCase):
    def test_teacher_is_exact_and_training_only(self):
        expected = read_json(ROOT / 'reference/input_hashes.json')['teacher']['sha256']
        check_hash(ROOT / 'teacher/targets.npz', expected)
        with gzip.open(ROOT / 'splits/train.txt.gz', 'rt') as f:
            training = set(f.read().splitlines())
        with gzip.open(ROOT / 'splits/teacher.txt.gz', 'rt') as f:
            ordered = f.read().splitlines()
        with np.load(ROOT / 'teacher/targets.npz') as d:
            self.assertEqual(d['paths'].tolist(), ordered)
            self.assertEqual(d['scores'].shape, (24000,16))
            self.assertTrue(np.isfinite(d['scores']).all())
            self.assertTrue(set(ordered) <= training)

    def test_splits_are_disjoint_and_grouped(self):
        sets = {}
        with gzip.open(ROOT / 'splits/groups.json.gz', 'rt') as f:
            groups = json.load(f)
        for split,count in [('train',117970),('tune',6554),('audit',6552)]:
            with gzip.open(ROOT / f'splits/{split}.txt.gz', 'rt') as f:
                rows = f.read().splitlines()
            self.assertEqual(len(rows), count)
            self.assertEqual(len(set(rows)), count)
            sets[split] = {groups[r] for r in rows}
        for a,b in [('train','tune'),('train','audit'),('tune','audit')]:
            self.assertFalse(sets[a] & sets[b])

    def test_committee_matches_explicit_probability_and_logit_mixtures(self):
        torch.manual_seed(52)
        base = FeatureModel()
        heads = [RegionHead(True,8) for _ in range(3)]
        teacher = RegionHead(True,8)
        columns = torch.tensor([1,5,12])
        network = RobustCommittee(base,heads,teacher,columns,teacher_weight=.2,original_weight=.25).eval()
        x = torch.randn(7,5,8)
        with torch.inference_mode():
            source = torch.stack([h(x).sigmoid() for h in heads]).mean(0)
            z = torch.logit(source)
            z[:,columns] = .8*z[:,columns] + .2*teacher(x)[:,columns]
            expected = .25*base.head(x).sigmoid() + .75*z.sigmoid()
            torch.testing.assert_close(network(x).sigmoid(), expected, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(network(x[:1]),network(x)[:1],rtol=1e-5,atol=1e-6)

    def test_runtime_and_recipe_match_reference(self):
        digest = read_json(ROOT / 'reference/submission_manifest.json')['run.py']
        check_hash(ROOT / 'runtime/run.py', digest)
        recipe = read_json(ROOT / 'configs/balanced.json')
        stages = {s['name']:s for s in recipe['stages']}
        self.assertEqual(len(stages), len(recipe['stages']))
        self.assertNotIn('--pure-half', stages['readout-cache-pe']['command'])
        self.assertIn('--pure-half', stages['readout-cache-eva']['command'])
        for member in ['pe','eva']:
            self.assertIn('--original-weight', stages['export-'+member]['command'])
        self.assertEqual(recipe['teacher_weight'], .2)
        self.assertEqual(recipe['original_weight'], .25)
        self.assertEqual(recipe['prior_power'], .4)

    def test_checksum_guard_rejects_changed_input(self):
        with self.assertRaises(ValueError):
            check_hash(ROOT / 'runtime/run.py', '0'*64)


if __name__ == '__main__':
    unittest.main()
