"""CLIP-style ViT with global + body-region readouts and the per-source committee."""
import math

import torch
from torch import nn
from torch.nn import functional as F
import timm

# Gaussian body regions (vertical centre/width, image-height fractions): head, upper, lower, whole body.
REGION_CENTERS = (.13, .40, .75, .5)
REGION_WIDTHS = (.15, .24, .25, .60)
# Region read by each of the 40 attributes (0 head, 1 upper, 2 lower, 3 whole body).
ATTRIBUTE_REGION = [3]*4 + [0]*3 + [1]*13 + [2]*15 + [3]*2 + [0]*3


def region_pool_weights(grid):
    y = (torch.arange(grid[0]) + .5) / grid[0]
    x = (torch.arange(grid[1]) + .5) / grid[1]
    centers, widths = torch.tensor(REGION_CENTERS), torch.tensor(REGION_WIDTHS)
    prior = -.5*((y[None, :, None]-centers[:, None, None])/widths[:, None, None])**2 - .5*((x[None, None, :]-.5)/.5)**2
    return prior.flatten(1).softmax(-1)


class RegionHead(nn.Module):
    """Linear readout: global token for every attribute plus the attribute's body-region token."""
    def __init__(self, dim=1024):
        super().__init__()
        self.dim = dim
        self.global_head = nn.Linear(dim, 40)
        self.local_weight = nn.Parameter(torch.zeros(40, dim))
        self.register_buffer('region_index', torch.tensor(ATTRIBUTE_REGION))
        self.dropout = nn.Dropout(.15)
        nn.init.normal_(self.global_head.weight, std=.005)
        nn.init.zeros_(self.global_head.bias)

    def forward(self, features):
        x = self.dropout(features)*math.sqrt(self.dim)
        local = x[:, 1:].index_select(1, self.region_index)
        return self.global_head(x[:, 0]) + (local*self.local_weight).sum(-1)


class SourceHead(nn.Module):
    """Per-source readout (wrapper keeps the released checkpoint key names)."""
    def __init__(self, dim=1024):
        super().__init__()
        self.base = RegionHead(dim)

    def forward(self, x):
        return self.base(x)


class ClipLarge(nn.Module):
    """Encoder + regional head. Inputs are RGB in [-1, 1]; CLIP normalisation is internal."""
    def __init__(self, name, size, weights=None):
        super().__init__()
        self.name = name
        self.backbone = timm.create_model(name, pretrained=False, img_size=tuple(size))
        patch = self.backbone.patch_embed.patch_size[0]
        self.grid = (size[0]//patch, size[1]//patch)
        if weights:
            from safetensors.torch import load_file
            from timm.layers import resample_abs_pos_embed
            state = load_file(str(weights))
            native = self.backbone.pretrained_cfg.get('input_size', (3, 224, 224))[1:]
            old_grid = [native[0]//patch, native[1]//patch]
            if list(self.grid) != old_grid:
                state['pos_embed'] = resample_abs_pos_embed(state['pos_embed'], list(self.grid), old_size=old_grid,
                                                            num_prefix_tokens=self.backbone.num_prefix_tokens)
            self.backbone.load_state_dict(state, strict=True)
        self.head = RegionHead(self.backbone.num_features)
        cfg = self.backbone.pretrained_cfg
        self.register_buffer('mean', torch.tensor(cfg['mean']).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(cfg['std']).view(1, 3, 1, 1))
        self.register_buffer('pool_weights', region_pool_weights(self.grid))

    def set_resolution(self, size):
        """Run at another input size: resample positional embeddings and rebuild region pooling."""
        patch = self.backbone.patch_embed.patch_size[0]
        if any(d <= 0 or d % patch for d in size):
            raise ValueError(f'Image dimensions must be positive multiples of {patch}')
        self.backbone.set_input_size(img_size=tuple(size))
        self.grid = (size[0]//patch, size[1]//patch)
        self.pool_weights = region_pool_weights(self.grid).to(self.pool_weights)

    def features(self, pixels):
        """(B, 5, D) L2-normalised tokens: [global, head, upper, lower, whole body]."""
        x = ((pixels+1)*.5-self.mean)/self.std
        tokens = self.backbone.forward_features(x)
        prefix = self.backbone.num_prefix_tokens
        local = torch.einsum('rn,bnd->brd', self.pool_weights.to(tokens.dtype), tokens[:, prefix:])
        return F.normalize(torch.cat([tokens[:, :1], local], dim=1).float(), dim=-1)

    def forward(self, pixels):
        return self.head(self.features(pixels))


def logit(p):
    return torch.log(p)-torch.log1p(-p)


class Committee(nn.Module):
    """Deployed predictor for one encoder; returns attribute logits.

    1. average the per-source heads in probability space;
    2. move the teacher-supervised columns 20% towards the teacher head (logit space);
    3. mix 25% of the original fine-tuned head back in (probability space).
    """
    def __init__(self, model, heads, teacher_head, columns, teacher_weight=.2, original_weight=.25):
        super().__init__()
        self.model = model
        self.heads = nn.ModuleList(heads)
        self.teacher_head = teacher_head
        self.register_buffer('columns', columns)
        self.teacher_weight = teacher_weight
        self.original_weight = original_weight

    def forward(self, pixels):
        x = self.model.features(pixels)
        z = logit(torch.stack([h(x).float().sigmoid() for h in self.heads]).mean(0).clamp(1e-7, 1-1e-7))
        zt = logit(self.teacher_head(x).float().sigmoid().clamp(1e-7, 1-1e-7))
        delta = (zt.index_select(1, self.columns)-z.index_select(1, self.columns))*self.teacher_weight
        z = z + torch.zeros_like(z).index_copy(1, self.columns, delta)
        original = self.model.head(x).float().sigmoid()
        p = ((1-self.original_weight)*z.sigmoid()+self.original_weight*original).clamp(1e-7, 1-1e-7)
        return logit(p)


def load_encoder(checkpoint):
    """Rebuild a ClipLarge from a fine-tuned/interpolated checkpoint (EMA weights)."""
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    args = state['config']['args']
    model = ClipLarge(args['timm_name'], (args['height'], args['width']))
    model.load_state_dict(state['ema'], strict=True)
    return model, state
