"""A compact ASC integration model and an explicit encoder/head boundary.

CompactASC is a repository integration architecture. Published CP-Mobile,
BC-ResNet, TF-SepNet and PaSST implementations can use the same split interface.
"""

from torch import nn


class SeparableBlock(nn.Sequential):
    def __init__(self, source: int, target: int):
        super().__init__(
            nn.Conv2d(source, source, 3, stride=2, padding=1, groups=source, bias=False),
            nn.BatchNorm2d(source), nn.SiLU(),
            nn.Conv2d(source, target, 1, bias=False),
            nn.BatchNorm2d(target), nn.SiLU(),
        )


class SplitASC(nn.Module):
    """Encoder -> [batch, feature_dim] -> scene head."""
    def __init__(self, encoder: nn.Module, head: nn.Module, feature_dim: int):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.feature_dim = feature_dim

    def forward(self, inputs):
        return self.head(self.encoder(inputs))


class CompactASC(SplitASC):
    def __init__(self, num_scenes: int = 10, feature_dim: int = 128):
        encoder = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24), nn.SiLU(),
            SeparableBlock(24, 48), SeparableBlock(48, 96),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, feature_dim), nn.SiLU(),
        )
        head = nn.Sequential(nn.LayerNorm(feature_dim), nn.Dropout(0.1),
                             nn.Linear(feature_dim, num_scenes))
        super().__init__(encoder, head, feature_dim)
