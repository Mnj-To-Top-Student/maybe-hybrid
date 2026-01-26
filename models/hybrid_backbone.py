import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4


'''
base = cnn + patch + vit
head = classifier
'''

class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = efficientnet_b4(weights="IMAGENET1K_V1")
        self.features = model.features[:5]  # cut early

    def forward(self, x):
        return self.features(x)

class PatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim):
        super().__init__()
        self.proj = nn.Linear(in_ch, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.proj(x)
        return x

class TinyViTBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class Classifier(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class HybridBase(nn.Module):
    def __init__(self, cnn, patch, vit):
        super().__init__()
        self.cnn = cnn
        self.patch = patch
        self.vit = vit

    def forward(self, x):
        x = self.cnn(x)
        x = self.patch(x)
        x = self.vit(x)
        x = x.mean(dim=1)   # (B, 192)
        return x

class HybridModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        backbone = efficientnet_b4(weights="IMAGENET1K_V1")
        self.cnn = backbone.features          # (B, 1792, 7, 7)
        self.token_proj = nn.Conv2d(1792, 256, kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Head exists BUT is NOT used in forward
        self.head = nn.Linear(256, num_classes)

        # For FedBABU / FedAvg
        self.base = nn.Sequential(
            self.cnn,
            self.token_proj
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.token_proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        x = x.mean(dim=1)     # (B, 256)
        return x              # ✅ FEATURES ONLY

if __name__ == "__main__":
    model = HybridModel(num_classes=8)
    x = torch.randn(10, 3, 224, 224)

    features = model(x)
    print(features.shape)      # ✅ (10, 256)

    logits = model.head(features)
    print(logits.shape)        


