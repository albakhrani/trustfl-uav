"""Scientific model construction: retain native initialization and parameters."""
from __future__ import annotations

import torch.nn.functional as F

from ..models import CNNEuroSAT, build_model as build_native_model
from ..data import NUM_CLASSES
from .pooling import adaptive_max_pool2d_fixed_order, operator_identity


FACTORY_ID = "scientific-model-factory-v3-poolfix-v1"


class ScientificCNNEuroSAT(CNNEuroSAT):
    """Inherit the unchanged native initializer; replace only adaptive pooling.

    No modules, parameters or buffers are added. deepcopy retains this class in
    both local-client and FLTrust root updates, on CPU and on CUDA.
    """

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = adaptive_max_pool2d_fixed_order(x)
        x = x.flatten(1)
        return self.fc2(F.relu(self.fc1(x)))


def _uses_eurosat_model(model_cfg, dataset):
    return model_cfg.name == "cnn_eurosat" or (model_cfg.name == "auto" and dataset == "eurosat")


def model_identity(model_cfg, dataset):
    """Execution identity is sealed separately from initialization/RNG recipes."""
    return {
        "factory_id": FACTORY_ID,
        "factory_module": "trustfl.scientific.models",
        "configured_name": model_cfg.name,
        "dataset": dataset,
        "class": "trustfl.scientific.models.ScientificCNNEuroSAT" if _uses_eurosat_model(model_cfg, dataset) else "unchanged_native_factory_dispatch",
        "pooling": operator_identity() if _uses_eurosat_model(model_cfg, dataset) else None,
        "initialization": "unchanged native initializer; no extra parameters/buffers or RNG draws",
    }


def build_model(model_cfg, dataset):
    if _uses_eurosat_model(model_cfg, dataset):
        return ScientificCNNEuroSAT(num_classes=model_cfg.num_classes or NUM_CLASSES[dataset])
    return build_native_model(model_cfg, dataset)
