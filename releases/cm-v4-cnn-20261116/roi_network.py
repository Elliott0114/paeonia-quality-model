"""Shared archived EfficientNet1D encoder, with bag-level HPLC supervision."""
from common import old,torch
from torch import nn

class RegionNet(nn.Module):
    def __init__(self,mode):
        super().__init__();assert mode in ['mil_mean','mil_attention'];self.mode=mode
        self.backbone=old.build_1d_backbone_model('efficientnet1d',6)
        if mode=='mil_attention':
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(20261106)
                self.attention=nn.Sequential(nn.Linear(48,16),nn.Tanh(),nn.Linear(16,1))
                nn.init.zeros_(self.attention[-1].weight);nn.init.zeros_(self.attention[-1].bias)

    def encode(self,x):
        n,k,l=x.shape
        h=self.backbone.stem(x.reshape(n*k,l).unsqueeze(1))
        h=self.backbone.features(h);h=self.backbone.pool(h).squeeze(-1)
        return h.reshape(n,k,48)

    def forward(self,x,weights,return_attention=False):
        h=self.encode(x)
        if self.mode=='mil_attention':
            w=(self.attention(h).squeeze(-1)+weights.log()).softmax(dim=1)
        else:w=weights
        pooled=(h*w.unsqueeze(-1)).sum(1)
        pred=self.backbone.head(pooled.unsqueeze(-1))
        return (pred,w) if return_attention else pred

def build(mode):return RegionNet(mode)
