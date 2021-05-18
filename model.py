import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class TransposeLast(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.transpose(-1, -2)


class MixerLayer(nn.Module):
    def __init__(self, n_samp, n_dim):
        super().__init__()
        self.tokenmix = nn.Sequential(
            nn.LayerNorm(n_dim),
            TransposeLast(),
            nn.Linear(n_samp, n_samp * 2),
            nn.GELU(),
            nn.Linear(n_samp * 2, n_samp),
            TransposeLast(),
        )
        self.channelmix = nn.Sequential(
            nn.LayerNorm(n_dim),
            nn.Linear(n_dim, n_dim * 2),
            nn.GELU(),
            nn.Linear(n_dim * 2, n_dim)
        )

    def forward(self, x):
        x = self.tokenmix(x) + x
        x = self.channelmix(x) + x
        return x


class MlpMixer(nn.Module):
    def __init__(self, img_size=28, patch_size=7, n_dim=64, n_classes=10):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(1, n_dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c')
        )
        n_samp = (img_size // patch_size) ** 2
        self.block = nn.Sequential(
            MixerLayer(n_samp, n_dim),
            MixerLayer(n_samp, n_dim),
            MixerLayer(n_samp, n_dim),
            MixerLayer(n_samp, n_dim),
        )
        self.clf = nn.Sequential(
            nn.Linear(n_dim, n_dim * 2),
            nn.GELU(),
            nn.Linear(n_dim * 2, n_classes),
        )

    def forward(self, x):
        x = self.embed(x)
        x = self.block(x)
        return self.clf(x.mean(dim=1))


if __name__ == "__main__":
    X = torch.randn(8, 1, 28, 28)
    mm = MlpMixer(img_size=28, patch_size=7, n_dim=64, n_classes=10)
    mm(X)
