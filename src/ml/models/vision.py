"""Image building blocks for train.py: a pretrained CNN backbone and two sklearn-style models that carry their
own preprocessing (decode, resize, normalise), so the saved model takes the same X as train.py does:

    - a DataFrame with a `path` column (what harness.load_train / load_val_inputs return for image files), or
    - a batch of decoded images, (N, H, W) or (N, H, W, C), uint8 or 0-255 floats (what the deployed
      predictor passes: one image with a batch dimension).

PatchKNNAnomaly   PatchCore-style anomaly score: a memory bank of patch features from NORMAL images; an
                  image's score is how far its most unusual patch is from any normal patch. Needs few or no
                  defect examples. binary_score.
ImageEmbeddingClassifier
                  Pooled backbone embeddings + logistic regression. Needs labelled examples of every class.
                  binary_score or class_probs.

Both run on CPU (ResNet-18 at 224 px: ~20-60 ms per image). Weights: harness.weights("resnet18.pth").
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_backbone(weights: str | None = None):
    """torchvision ResNet-18 up to layer3, in eval mode; ImageNet weights from a local state dict (None = random
    init, for tests only). layer4 and the classifier are dropped: unused, and they'd triple the saved model."""
    import torch
    import torchvision

    model = torchvision.models.resnet18(weights=None)
    if weights:
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.layer4 = model.fc = torch.nn.Identity()
    return model.eval()


def _images(X) -> list[np.ndarray]:
    """X (DataFrame with `path`, list of paths, or an array batch) -> list of HxW[xC] arrays."""
    if hasattr(X, "columns"):
        X = X["path"].tolist()
    if isinstance(X, (list, tuple)) and X and isinstance(X[0], str):
        from PIL import Image

        return [np.asarray(Image.open(p)) for p in X]
    return list(np.asarray(X))


def to_tensor(X, size: int = 224):
    """Images -> normalised float tensor (N, 3, size, size)."""
    import torch
    import torch.nn.functional as F

    out = []
    for img in _images(X):
        a = np.asarray(img, dtype=np.float32)
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=2)
        elif a.shape[2] == 1:
            a = np.repeat(a, 3, axis=2)
        a = a[..., :3]  # drop alpha
        if a.max() > 1.5:  # 0-255 (uint8, or floats from the deployed predictor)
            a = a / 255.0
        t = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None]
        out.append(F.interpolate(t, size=(size, size), mode="bilinear", antialias=True, align_corners=False))
    x = torch.cat(out)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def _batches(X, batch_size: int):
    n = len(X)
    for i in range(0, n, batch_size):
        yield X.iloc[i:i + batch_size] if hasattr(X, "iloc") else X[i:i + batch_size]


class _BackboneModel(BaseEstimator, ClassifierMixin):
    def _net(self):
        if getattr(self, "backbone_", None) is None:
            self.backbone_ = load_backbone(self.weights)
        return self.backbone_

    def _maps(self, x):
        """layer2 and layer3 feature maps for a normalised batch."""
        net = self._net()
        x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        l2 = net.layer2(net.layer1(x))
        return l2, net.layer3(l2)


class PatchKNNAnomaly(_BackboneModel):
    """fit(X, y): builds the memory bank from rows where y == normal_label (all rows when y is None).
    predict_proba(X)[:, 1] is the anomaly score in [0, 1] (0.5 = a typical held-out normal image's raw score
    plus calibration_sigmas * its spread). Rank-based metrics (AUROC) only depend on the raw distance."""

    def __init__(self, weights=None, image_size=224, memory_size=10000, normal_label=0, holdout=0.3,
                 batch_size=16, random_state=0):
        self.weights = weights
        self.image_size = image_size
        self.memory_size = memory_size
        self.normal_label = normal_label
        self.holdout = holdout
        self.batch_size = batch_size
        self.random_state = random_state

    def _patches(self, X):
        """Per image: (n_patches, dim) locally averaged layer2+layer3 features (PatchCore)."""
        import torch
        import torch.nn.functional as F

        with torch.inference_mode():
            for chunk in _batches(X, self.batch_size):
                l2, l3 = self._maps(to_tensor(chunk, self.image_size))
                l3 = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
                f = F.avg_pool2d(torch.cat([l2, l3], 1), 3, stride=1, padding=1)
                yield from f.flatten(2).transpose(1, 2)  # (N, HW, D) -> per image (HW, D)

    def _raw(self, X) -> np.ndarray:
        import torch

        bank = torch.from_numpy(self.memory_bank_)
        scores = []
        for p in self._patches(X):
            d = torch.cdist(p, bank).min(dim=1).values  # each patch's distance to the nearest normal patch
            scores.append(float(d.max()))  # the image is as abnormal as its worst patch
        return np.asarray(scores)

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.random_state)
        idx = np.arange(len(X)) if y is None else np.flatnonzero(np.asarray(y) == self.normal_label)
        if len(idx) < 2:
            raise ValueError("PatchKNNAnomaly needs at least 2 normal images")
        idx = rng.permutation(idx)
        n_cal = max(1, round(self.holdout * len(idx))) if self.holdout else 0
        cal, fit = idx[:n_cal], idx[n_cal:]
        rows = (lambda i: X.iloc[i]) if hasattr(X, "iloc") else (lambda i: np.asarray(X)[i])

        per_image = max(1, self.memory_size // len(fit))
        bank = []
        for p in self._patches(rows(fit)):
            take = rng.choice(len(p), size=min(per_image, len(p)), replace=False)
            bank.append(p[take].numpy())
        self.memory_bank_ = np.concatenate(bank).astype(np.float32)

        if n_cal:
            self.calibration_raw_ = self._raw(rows(cal))  # normal images the bank never saw
            ref = self.calibration_raw_
            self.center_, self.scale_ = float(np.median(ref)), float(ref.std() + 1e-6)
        else:
            self.calibration_raw_ = None
            self.center_, self.scale_ = 0.0, 1.0
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X) -> np.ndarray:
        return self._raw(X)

    def _squash(self, raw) -> np.ndarray:
        z = (np.asarray(raw, dtype=float) - self.center_) / self.scale_
        return 1.0 / (1.0 + np.exp(-np.clip(z - 2.0, -50, 50)))  # ~0.5 at 2 sigmas above typical normal

    def predict_proba(self, X) -> np.ndarray:
        s = self._squash(self._raw(X))
        return np.stack([1 - s, s], axis=1)

    def threshold_for_far(self, far: float) -> float:
        """Score threshold that about `far` of unseen normal images exceed, from the held-out normals.
        Needs no defect examples; pass e.g. half the false-alarm KPI to keep a margin."""
        if self.calibration_raw_ is None:
            raise ValueError("fit with holdout > 0 to calibrate a threshold")
        return float(self._squash(np.quantile(self.calibration_raw_, 1.0 - far)))

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ImageEmbeddingClassifier(_BackboneModel):
    """Global-average-pooled layer2+layer3 embeddings -> standardise -> logistic regression."""

    def __init__(self, weights=None, image_size=224, C=1.0, class_weight="balanced", batch_size=16):
        self.weights = weights
        self.image_size = image_size
        self.C = C
        self.class_weight = class_weight
        self.batch_size = batch_size

    def embed(self, X) -> np.ndarray:
        import torch

        out = []
        with torch.inference_mode():
            for chunk in _batches(X, self.batch_size):
                l2, l3 = self._maps(to_tensor(chunk, self.image_size))
                out.append(torch.cat([l2.mean((2, 3)), l3.mean((2, 3))], 1).numpy())
        return np.concatenate(out)

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.head_ = make_pipeline(StandardScaler(), LogisticRegression(
            C=self.C, class_weight=self.class_weight, max_iter=5000))
        self.head_.fit(self.embed(X), np.asarray(y))
        self.classes_ = self.head_.classes_
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.head_.predict_proba(self.embed(X))

    def predict(self, X) -> np.ndarray:
        return self.head_.predict(self.embed(X))
