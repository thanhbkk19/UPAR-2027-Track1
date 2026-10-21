"""Generic CLIP ViT-L/14 with global and body-region attribute classifiers.

Inputs use the submission convention [-1,1]. CLIP normalization is internal.
The spatial pooling is our lightweight adaptation, not a paper reproduction.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
import timm


class RegionHead(nn.Module):
    def __init__(self, regional=True, dim=1024):
        super().__init__()
        self.regional = regional
        self.dim = dim
        self.global_head = nn.Linear(dim,40)
        self.local_weight = nn.Parameter(torch.zeros(40,dim))
        self.register_buffer('region_index',torch.tensor([3]*4+[0]*3+[1]*13+[2]*15+[3]*2+[0]*3))
        self.dropout = nn.Dropout(.15)
        nn.init.normal_(self.global_head.weight,std=.005)
        nn.init.zeros_(self.global_head.bias)

    def forward(self, features):
        x = self.dropout(features)*math.sqrt(self.dim)
        result = self.global_head(x[:,0])
        if self.regional:
            local = x[:,1:].index_select(1,self.region_index)
            result = result+(local*self.local_weight).sum(-1)
        return result


class ClipLarge(nn.Module):
    def __init__(self, tag='apple_mclip2_dfndr2b', weights=None, regional=True, size=(224,224), name=None, drop_path=0., source_heads=0):
        super().__init__()
        self.tag = tag
        size = tuple(size)
        self.name = name or 'vit_large_patch14_clip_224.'+tag
        kwargs = {} if (size == (224,224) and name is None) else dict(img_size=size)
        if drop_path:kwargs['drop_path_rate'] = drop_path
        self.backbone = timm.create_model(self.name,pretrained=False,**kwargs)
        patch = self.backbone.patch_embed.patch_size[0]
        self.grid = (size[0]//patch,size[1]//patch)
        native = self.backbone.pretrained_cfg.get('input_size',(3,224,224))[1:]
        old_grid = [native[0]//patch,native[1]//patch]
        if weights:
            from safetensors.torch import load_file
            state = load_file(str(weights))
            # Weights converted at the target grid (e.g. Sapiens) need no resampling.
            at_target = state['pos_embed'].shape[1]-self.backbone.num_prefix_tokens == self.grid[0]*self.grid[1]
            if list(self.grid) != old_grid and not at_target:
                from timm.layers import resample_abs_pos_embed
                state['pos_embed'] = resample_abs_pos_embed(state['pos_embed'],list(self.grid),old_size=old_grid,num_prefix_tokens=self.backbone.num_prefix_tokens)
            self.backbone.load_state_dict(state,strict=True)
        self.head = RegionHead(regional,dim=self.backbone.num_features)
        # Optional per-source heads (one annotation convention each); inference averages their probabilities.
        self.source_heads = nn.ModuleList([RegionHead(regional,dim=self.backbone.num_features) for _ in range(source_heads)]) if source_heads else None
        cfg = self.backbone.pretrained_cfg
        self.register_buffer('mean',torch.tensor(cfg['mean']).view(1,3,1,1))
        self.register_buffer('std',torch.tensor(cfg['std']).view(1,3,1,1))
        y=(torch.arange(self.grid[0])+.5)/self.grid[0]
        x=(torch.arange(self.grid[1])+.5)/self.grid[1]
        centers=torch.tensor([.13,.40,.75,.5])
        widths=torch.tensor([.15,.24,.25,.60])
        prior=-.5*((y[None,:,None]-centers[:,None,None])/widths[:,None,None])**2-.5*((x[None,None,:]-.5)/.5)**2
        self.register_buffer('pool_weights',prior.flatten(1).softmax(-1))

    def features(self,pixels):
        x=((pixels+1)*.5-self.mean)/self.std
        tokens=self.backbone.forward_features(x)
        prefix=self.backbone.num_prefix_tokens
        # Models without a class token (e.g. PE-core-G) use their pretrained attention pool as the global token.
        first=tokens[:,:1] if prefix else self.backbone.pool(tokens)[:,None]
        local=torch.einsum('rn,bnd->brd',self.pool_weights.to(tokens.dtype),tokens[:,prefix:])
        features=torch.cat([first,local],dim=1)
        return F.normalize(features.float(),dim=-1)

    def forward(self,pixels):
        features=self.features(pixels)
        if self.source_heads is None:return self.head(features)
        p=torch.stack([h(features).float().sigmoid() for h in self.source_heads]).mean(0).clamp(1e-7,1-1e-7)
        return torch.log(p)-torch.log1p(-p)
