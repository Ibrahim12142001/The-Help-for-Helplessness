import torch
import torch.nn as nn
from torchinfo import summary
from torch.nn import Conv2d, MaxPool2d, ReLU
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18
from torchinfo import summary
from tqdm import tqdm


# ====== Greg's Model: 3D CNN ======
# class Conv3DBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, double=False):
#         super(Conv3DBlock, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#         )
#         self.conv2 = nn.Sequential() if not double else nn.Sequential(
#             nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#         )
#         self.pool = nn.MaxPool3d(kernel_size=2)
#         self.norm = nn.BatchNorm3d(out_channels)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.pool(x)
#         x = self.norm(x)
#         return x

# class HelplessnessClassifier(nn.Module):
#     def __init__(self, input_channels=3):
#         super(HelplessnessClassifier, self).__init__()
#         self.cnn = nn.Sequential(
#             Conv3DBlock(in_channels=input_channels, out_channels=64),
#             Conv3DBlock(in_channels=64, out_channels=128),
#             Conv3DBlock(in_channels=128, out_channels=256),
#             Conv3DBlock(in_channels=256, out_channels=512),
#         )
#         self.avg_pool = nn.AdaptiveAvgPool3d(2)
#         self.fc = nn.Sequential(
#             nn.Linear(4096, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(512, 3)
#         )

#     def forward(self, x):
#         x = self.cnn(x)
#         x = self.avg_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x


# ====== Ibrahim's Model: 2D CNN + LSTM ======


# class HelplessnessClassifier(nn.Module):
#     def __init__(self, num_classes=3):
#         super(HelplessnessClassifier, self).__init__()

#         # Four convolutional blocks: 32 -> 64 -> 128 -> 256
#         self.conv_block1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2)  # 224 => 112
#         )

#         self.conv_block2 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2)  # 112 => 56
#         )

#         self.conv_block3 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2)  # 56 => 28
#         )

#         self.conv_block4 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2)  # 28 => 14
#         )

#         # Use a dummy forward pass on a single frame to figure out the flattened feature size
#         dummy_frame = torch.zeros(1, 3, 224, 224)  # shape (batch=1, channels=3, H, W)
#         out = self._forward_cnn(dummy_frame)
#         feature_dim = out.numel()
#         print(f"[DEBUG] Per-frame feature dim: {feature_dim}")

#         # Two-layer FC head
#         self.fc = nn.Sequential(
#             nn.Linear(feature_dim, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(512, num_classes)
#         )

#     def _forward_cnn(self, x):
#         """
#         Passes a single frame (N,3,H,W) through the 4 conv blocks and flattens.
#         Returns shape (N, features).
#         """
#         x = self.conv_block1(x)
#         x = self.conv_block2(x)
#         x = self.conv_block3(x)
#         x = self.conv_block4(x)
#         return x.view(x.size(0), -1)  # flatten => (N, features)

#     def forward(self, x):
#         """
#         x shape: (batch_size, seq_len, channels=3, height=224, width=224).
#           1) Merge batch & time (B*T, C, H, W)
#           2) Pass each frame through conv blocks
#           3) Reshape back to (B, T, features)
#           4) Average across time
#           5) Classify
#         """
#         B, T, C, H, W = x.shape

#         # Merge (B, T) => (B*T)
#         x = x.view(B*T, C, H, W)
#         x = self._forward_cnn(x)     # => (B*T, feature_dim)

#         # Reshape => (B, T, feature_dim)
#         x = x.view(B, T, -1)

#         # Average features across time dimension
#         x = x.mean(dim=1)  # => (B, feature_dim)

#         # Classify
#         x = self.fc(x)
#         return x


class HelplessnessClassifier(nn.Module):
    """
    2D CNN + LSTM for Grayscale frames:
      - (B, T, 1, H, W) input
      - CNN processes each frame => feature vector
      - LSTM processes the time dimension
      - Final FC for classification
    """

    def __init__(self, num_classes=3):
        super(HelplessnessClassifier, self).__init__()

        # 1) 2D CNN blocks with in_channels=1 for grayscale
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # H/2, W/2
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # H/4, W/4
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # H/8, W/8
        )

        # Dummy pass to find feature dim
        dummy = torch.zeros(1, 1, 112, 112)  # single frame, 1 channel, e.g. 112x112
        out = self._forward_cnn(dummy)
        feature_dim = out.shape[1]  # (1, feature_dim)
        print(f"[DEBUG] CNN feature_dim = {feature_dim}")

        # LSTM: transforms sequence of CNN features => final hidden
        self.lstm_hidden = 128
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=self.lstm_hidden,
                            num_layers=1, batch_first=True)

        # Classifier
        self.fc = nn.Linear(self.lstm_hidden, num_classes)

    def _forward_cnn(self, x):
        """ x: (N, 1, H, W), returns flattened features shape (N, feature_dim) """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        """
        x shape: (B, T, 1, H, W)
          1) Merge B,T => (B*T, 1, H, W)
          2) CNN => (B*T, feature_dim)
          3) Reshape => (B, T, feature_dim)
          4) LSTM => (B, T, hidden)
          5) final hidden => FC => (B, num_classes)
        """
        B, T, C, H, W = x.shape  # C=1 for grayscale
        x = x.view(B * T, C, H, W)
        feats = self._forward_cnn(x)  # => (B*T, feature_dim)

        feats = feats.view(B, T, -1)  # => (B, T, feature_dim)
        lstm_out, _ = self.lstm(feats)  # => (B, T, lstm_hidden)

        last_out = lstm_out[:, -1, :]  # => (B, lstm_hidden)
        logits = self.fc(last_out)  # => (B, num_classes)
        return logits


if __name__ == "__main__":
    model = HelplessnessClassifier(num_classes=3)
    test_input = torch.zeros(2, 10, 1, 112, 112)  # (batch=2, T=10, 1 channel, 112x112)
    out = model(test_input)
    print("Output shape:", out.shape)  # => (2, 3)
