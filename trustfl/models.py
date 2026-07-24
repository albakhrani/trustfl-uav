"""Models. Small CNN for MNIST/FEMNIST (standard in FL papers, cf. McMahan
et al. 2017), ResNet-18 adapted to 32x32 for CIFAR-100, MLP for synthetic."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import NUM_CLASSES


class MLP(nn.Module):
    def __init__(self, in_dim=32, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, x):
        return self.net(x)


class CNNMnist(nn.Module):
    """2-conv CNN (~1.2M params), the FedAvg-paper MNIST architecture."""

    def __init__(self, num_classes=10, in_ch=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        return self.fc2(F.relu(self.fc1(x)))


class CNNEuroSAT(nn.Module):
    """CNNMnist trunk for 64x64 RGB inputs (EuroSAT): in_ch=3, num_classes=10.
    After the two pooled convs (64x64 -> 16x16) an adaptive max-pool reduces
    to the same 7x7 grid the MNIST/FEMNIST trunk flattens, keeping the wire
    size (~53 Mbit) in the class the shared 5 s comm deadline was calibrated
    for. (A raw 64*16*16 flatten = 8.4M params = 270 Mbit is undeliverable
    under the unchanged channel settings: 0-2 deliveries/round, measured.)"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.adaptive_max_pool2d(x, 7)
        x = x.flatten(1)
        return self.fc2(F.relu(self.fc1(x)))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.GroupNorm(8, cout)  # GroupNorm: BN is unstable in FL (Hsieh et al. 2020)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.GroupNorm(8, cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.GroupNorm(8, cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.short(x))


class ResNet18(nn.Module):
    """ResNet-18 for 32x32 inputs, GroupNorm variant for federated training."""

    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.GroupNorm(8, 64)
        layers, cin = [], 64
        for cout, stride in [(64, 1), (64, 1), (128, 2), (128, 1),
                             (256, 2), (256, 1), (512, 2), (512, 1)]:
            layers.append(BasicBlock(cin, cout, stride))
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def build_model(model_cfg, dataset: str) -> nn.Module:
    n_cls = model_cfg.num_classes or NUM_CLASSES[dataset]
    name = model_cfg.name
    if name == "auto":
        name = {"synthetic": "mlp", "mnist": "cnn_mnist",
                "femnist": "cnn_femnist", "cifar100": "resnet18",
                "eurosat": "cnn_eurosat"}[dataset]
    if name == "mlp":
        return MLP(num_classes=n_cls)
    if name == "cnn_mnist":
        return CNNMnist(num_classes=n_cls)
    if name == "cnn_femnist":
        return CNNMnist(num_classes=n_cls)  # same trunk, 62-way head
    if name == "cnn_eurosat":
        return CNNEuroSAT(num_classes=n_cls)
    if name == "resnet18":
        return ResNet18(num_classes=n_cls)
    raise ValueError(f"unknown model {name}")
