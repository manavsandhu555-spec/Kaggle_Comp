"""
ffc_lib.py - reusable pipeline code for the Facial Feature Classification notebook.

Why a separate module?
    On Windows, PyTorch DataLoader workers are started with "spawn", which means
    every worker re-imports the code it needs. Classes defined *inside* a Jupyter
    notebook cannot be re-imported, so DataLoader(num_workers > 0) crashes.
    Keeping the Dataset / transforms / train loop in this file fixes that and also
    makes the pipeline reusable and easier to test.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# Some images on the internet are slightly truncated; let PIL load them anyway.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = 42) -> None:
    """Fix every random number generator we use so results are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker its own deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------- #
# File discovery helpers
# --------------------------------------------------------------------------- #
def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def file_md5(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dhash(img: Image.Image, size: int = 8) -> int:
    """Difference hash: a 64-bit perceptual fingerprint. Near-identical images
    (resized / re-compressed copies) get the same or very similar hashes."""
    g = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def laplacian_variance(img: Image.Image, max_side: int = 256) -> float:
    """Blur score = variance of the Laplacian. Low value -> blurry image."""
    g = img.convert("L")
    g.thumbnail((max_side, max_side))
    a = np.asarray(g, dtype=np.float32)
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    lap = (a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1] + a[2:, 1:-1] - 4 * a[1:-1, 1:-1])
    return float(lap.var())


def inspect_image(path: str | Path) -> dict:
    """Open one image and collect everything Task 1 needs about it."""
    info = {"filepath": str(path), "ok": True, "error": "", "width": np.nan,
            "height": np.nan, "mode": "", "format": "", "file_kb": np.nan,
            "md5": "", "dhash": np.nan, "blur": np.nan,
            "mean_brightness": np.nan}
    try:
        info["file_kb"] = os.path.getsize(path) / 1024
        info["md5"] = file_md5(path)
        with Image.open(path) as im:
            im.verify()                      # catches truncated / corrupt files
        with Image.open(path) as im:         # re-open: verify() invalidates the handle
            info["width"], info["height"] = im.size
            info["mode"] = im.mode
            info["format"] = im.format or ""
            rgb = im.convert("RGB")
            info["dhash"] = dhash(rgb)
            info["blur"] = laplacian_variance(rgb)
            info["mean_brightness"] = float(np.asarray(rgb.convert("L")).mean())
    except Exception as e:  # noqa: BLE001
        info["ok"] = False
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def hamming(a: int, b: int) -> int:
    return bin(int(a) ^ int(b)).count("1")


def near_duplicate_groups(hashes: list[int], max_dist: int = 4) -> np.ndarray:
    """Union-find over perceptual hashes. Returns a group id per image so that
    near-duplicates share an id (used later to keep them in the same split)."""
    n = len(hashes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    h = np.array([int(x) for x in hashes], dtype=np.uint64)
    popcount8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    # Vectorised pairwise Hamming distance, done in blocks to limit memory use.
    block = max(16, 20_000_000 // max(n, 1))
    for s in range(0, n, block):
        x = np.ascontiguousarray(h[s:s + block, None] ^ h[None, :])
        d = popcount8[x.view(np.uint8)].reshape(x.shape[0], n, 8).sum(-1, dtype=np.int32)
        ii, jj = np.where(d <= max_dist)
        for i, j in zip(ii + s, jj):
            if i < j:
                union(i, j)
    return np.array([find(i) for i in range(n)])


def find_dir(root: Path, names: tuple[str, ...]) -> Path | None:
    """Breadth-first search for a directory whose name is in `names` (case-insensitive)."""
    names = {n.lower() for n in names}
    for p in sorted(Path(root).rglob("*"), key=lambda q: len(q.parts)):
        if p.is_dir() and p.name.lower() in names:
            return p
    return None


def norm_id(x) -> str:
    """Normalise an image id so '00017.jpg', '00017', '17' and 17 all match."""
    stem = Path(str(x).strip()).stem
    return str(int(stem)) if stem.isdigit() else stem


def discover_dataset(data_dir: str | Path, class_names: list[str]) -> dict:
    """Locate train images + labels, test images and the sample submission.

    Supports the two usual Kaggle layouts:
      (a) train/<ClassName>/*.jpg                      (folder per class)
      (b) train/**/*.jpg + train_labels.csv(id, class)  (labels in a CSV)
    Ids are matched after normalisation (zero padding / extension ignored).
    CSVs are read as text so ids such as '00017' keep their exact spelling.
    """
    data_dir = Path(data_dir)
    lower = {c.lower(): c for c in class_names}
    train_dir = find_dir(data_dir, ("train", "training", "train_images", "train_data"))
    test_dir = find_dir(data_dir, ("test", "testing", "test_images", "test_data"))
    csvs = sorted(data_dir.rglob("*.csv"))
    sample_csv = next((c for c in csvs if "sample" in c.name.lower() or "submission" in c.name.lower()), None)
    sample_sub = pd.read_csv(sample_csv, dtype=str) if sample_csv else None

    train_df, layout = None, None
    if train_dir is not None:
        sub = [d for d in train_dir.iterdir() if d.is_dir()]
        cls_dirs = [d for d in sub if d.name.lower() in lower]
        if cls_dirs:
            layout = "folder-per-class"
            rows = [(str(p), p.name, lower[d.name.lower()]) for d in cls_dirs for p in list_images(d)]
            train_df = pd.DataFrame(rows, columns=["filepath", "image_id", "label"])
    if train_df is None:
        # look for a CSV in which one column only contains class names
        for c in csvs:
            if c == sample_csv:
                continue
            df = pd.read_csv(c, dtype=str)
            lab_col = next((col for col in df.columns
                            if df[col].str.lower().isin(lower).mean() > 0.95), None)
            if lab_col is None:
                continue
            id_col = next(col for col in df.columns if col != lab_col)
            img_root = train_dir or c.parent
            by_id = {norm_id(p.name): p for p in list_images(img_root)}
            fp = df[id_col].map(lambda s: by_id.get(norm_id(s)))
            n_missing = int(fp.isna().sum())
            train_df = pd.DataFrame({"filepath": fp, "image_id": df[id_col],
                                     "label": df[lab_col].str.lower().map(lower)})
            train_df = train_df[fp.notna().values].reset_index(drop=True)
            train_df["filepath"] = train_df.filepath.astype(str)
            layout = f"csv-labels ({c.name}: id column '{id_col}', label column '{lab_col}')"
            if n_missing:
                layout += f" - WARNING: {n_missing} ids without an image file"
            break
    if train_df is None:
        raise FileNotFoundError(f"Could not find labelled training data under {data_dir.resolve()}")

    test_df = None
    if test_dir is not None:
        files = list_images(test_dir)
        test_df = pd.DataFrame({"filepath": [str(p) for p in files],
                                "filename": [p.name for p in files],
                                "key": [norm_id(p.name) for p in files]})
    return dict(train_df=train_df, test_df=test_df, sample_sub=sample_sub,
                sample_csv=sample_csv, train_dir=train_dir, test_dir=test_dir, layout=layout)


# --------------------------------------------------------------------------- #
# Dataset + transforms
# --------------------------------------------------------------------------- #
class ImageDataset(Dataset):
    """Reads an image from disk, converts to RGB, applies the transform.

    labels may be None (test set); then the image id is returned instead.
    """

    def __init__(self, filepaths, labels=None, transform=None, ids=None):
        self.filepaths = list(map(str, filepaths))
        self.labels = None if labels is None else list(map(int, labels))
        self.transform = transform
        self.ids = list(ids) if ids is not None else [Path(p).name for p in self.filepaths]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        with Image.open(self.filepaths[idx]) as im:
            img = im.convert("RGB")          # handles greyscale / RGBA / palette images
        if self.transform is not None:
            img = self.transform(img)
        if self.labels is None:
            return img, self.ids[idx]
        return img, self.labels[idx]


def build_transforms(img_size: int = 224, augment: bool = True,
                     mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Returns (train_tf, eval_tf).

    * eval_tf is deterministic and is used for validation AND test, so both are
      preprocessed exactly the same way.
    * train_tf adds label-preserving augmentation only (a flipped / slightly
      rotated / re-lit cat eye is still a cat eye).
    """
    eval_tf = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),                        # [0,255] -> [0,1]
        T.Normalize(mean, std),              # ImageNet statistics (pretrained backbone)
    ])
    if not augment:
        return eval_tf, eval_tf
    train_tf = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.75, 1.33),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.03),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.2, scale=(0.02, 0.12)),
    ])
    return train_tf, eval_tf


class RandomDownscale:
    """Simulates the tiny / pixelated crops seen in the test set: shrink the
    image to a random small size, then let the next transform scale it back up."""

    def __init__(self, p: float = 0.35, min_side: int = 16, max_side: int = 64):
        self.p, self.min_side, self.max_side = p, min_side, max_side

    def __call__(self, img):
        if random.random() > self.p:
            return img
        target = random.randint(self.min_side, self.max_side)
        w, h = img.size
        scale = target / max(w, h)
        if scale >= 1:
            return img
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                           random.choice([Image.NEAREST, Image.BILINEAR, Image.BICUBIC]))
        return small.resize((w, h), random.choice([Image.NEAREST, Image.BILINEAR]))

    def __repr__(self):
        return f"RandomDownscale(p={self.p}, side={self.min_side}-{self.max_side})"


def build_strong_transforms(img_size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Heavier augmentation aimed at the train -> test shift: more varied crop
    shapes / scales, low-resolution simulation, blur and stronger colour changes."""
    train_tf = T.Compose([
        RandomDownscale(p=0.35),
        T.RandomResizedCrop(img_size, scale=(0.55, 1.0), ratio=(0.6, 1.67),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=20),
        T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.05),
        T.RandomGrayscale(p=0.1),
        T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.5))], p=0.2),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    _, eval_tf = build_transforms(img_size, augment=False, mean=mean, std=std)
    return train_tf, eval_tf


def make_loader(ds, batch_size, shuffle, num_workers, seed=42, drop_last=False, persistent=True):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent and num_workers > 0, worker_init_fn=seed_worker,
        generator=g,
    )


def denormalize(t: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """Tensor (C,H,W) normalised -> numpy (H,W,C) in [0,1] for plotting."""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (t.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class SimpleCNN(nn.Module):
    """Small custom CNN trained from scratch - the baseline experiment.

    4 conv blocks (Conv-BN-ReLU x2 + MaxPool), global average pooling,
    dropout and a linear layer with one output per class.
    """

    def __init__(self, num_classes: int = 8, width: int = 32, dropout: float = 0.3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        w = width
        self.features = nn.Sequential(block(3, w), block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 8 * w))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(8 * w, num_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def build_model(arch: str, num_classes: int, pretrained: bool = True,
                drop_rate: float = 0.3, drop_path_rate: float = 0.1) -> nn.Module:
    if arch == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, dropout=drop_rate)
    import timm
    kwargs = dict(pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    try:
        return timm.create_model(arch, drop_path_rate=drop_path_rate, **kwargs)
    except TypeError:  # some archs do not accept drop_path_rate
        return timm.create_model(arch, **kwargs)


def head_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Parameters of the final classification layer (timm or SimpleCNN)."""
    if hasattr(model, "get_classifier"):
        return list(model.get_classifier().parameters())
    return list(model.head.parameters())


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze / unfreeze everything except the classifier head."""
    head_ids = {id(p) for p in head_parameters(model)}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad = trainable


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def warmup_cosine(optimizer, warmup_steps: int, total_steps: int, min_factor: float = 0.01):
    """Linear warm-up then cosine decay of the learning rate (per batch)."""
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, scaler=None,
                    max_batches=None):
    model.train()
    tot_loss, correct, n = 0.0, 0, 0
    use_amp = scaler is not None
    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        tot_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return tot_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, tta: bool = False, max_batches=None):
    """Returns (loss, accuracy, probs[N,C], labels[N]).  Uses plain CE loss
    (no label smoothing) so validation loss is comparable across experiments."""
    model.eval()
    all_probs, all_y = [], []
    tot_loss, n = 0.0, 0
    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x).float()
            if tta:
                logits = (logits + model(torch.flip(x, dims=[3])).float()) / 2
        loss = criterion(logits, y)
        tot_loss += loss.item() * x.size(0)
        n += x.size(0)
        all_probs.append(torch.softmax(logits, 1).cpu())
        all_y.append(y.cpu())
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_y).numpy()
    acc = float((probs.argmax(1) == labels).mean())
    return tot_loss / max(n, 1), acc, probs, labels


@torch.no_grad()
def predict(model, loader, device, tta: bool = True):
    """Predict class probabilities for an unlabelled loader -> (ids, probs)."""
    model.eval()
    ids, probs = [], []
    for x, batch_ids in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x).float()
            if tta:
                logits = (logits + model(torch.flip(x, dims=[3])).float()) / 2
        probs.append(torch.softmax(logits, 1).cpu())
        ids.extend(list(batch_ids))
    return ids, torch.cat(probs).numpy()


@dataclass
class TrainConfig:
    name: str
    arch: str
    pretrained: bool = True
    freeze_backbone: bool = False
    epochs: int = 15
    lr: float = 3e-4               # learning rate of the head
    backbone_lr_mult: float = 0.1  # backbone LR = lr * mult (discriminative LR)
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    warmup_epochs: int = 1
    patience: int = 4              # early stopping patience (epochs)
    drop_rate: float = 0.3
    use_class_weights: bool = False
    extra: dict = field(default_factory=dict)


def fit(cfg: TrainConfig, train_loader, val_loader, num_classes, device, out_dir,
        class_weights=None, max_batches=None, verbose=True):
    """Full training run with AdamW, warm-up + cosine LR, label smoothing,
    mixed precision, best-checkpointing on validation loss and early stopping.

    Returns (model loaded with best weights, history DataFrame, summary dict).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg.arch, num_classes, pretrained=cfg.pretrained, drop_rate=cfg.drop_rate).to(device)
    if cfg.freeze_backbone:
        set_backbone_trainable(model, False)

    head_ids = {id(p) for p in head_parameters(model)}
    head = [p for p in model.parameters() if id(p) in head_ids and p.requires_grad]
    body = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
    body_lr = cfg.lr if cfg.arch == "simple_cnn" else cfg.lr * cfg.backbone_lr_mult
    groups = [{"params": head, "lr": cfg.lr}]
    if body:
        groups.append({"params": body, "lr": body_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

    steps_per_epoch = len(train_loader) if max_batches is None else min(len(train_loader), max_batches)
    scheduler = warmup_cosine(optimizer, cfg.warmup_epochs * steps_per_epoch, cfg.epochs * steps_per_epoch)

    w = None if (class_weights is None or not cfg.use_class_weights) else torch.tensor(class_weights, dtype=torch.float32, device=device)
    train_criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    total, trainable = count_params(model)
    if verbose:
        print(f"[{cfg.name}] arch={cfg.arch} params={total/1e6:.2f}M trainable={trainable/1e6:.2f}M "
              f"head_lr={cfg.lr:g} body_lr={body_lr:g}")

    history, best_loss, best_epoch, bad = [], float("inf"), -1, 0
    ckpt = out_dir / f"{cfg.name}_best.pt"
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        te = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, train_criterion, optimizer, scheduler,
                                          device, scaler, max_batches)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, eval_criterion, device, max_batches=max_batches)
        lr_now = optimizer.param_groups[0]["lr"]
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, lr=lr_now, sec=time.time() - te))
        improved = va_loss < best_loss - 1e-4
        if improved:
            best_loss, best_epoch, bad = va_loss, epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        if verbose:
            print(f"  ep {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f} | lr {lr_now:.2e} | "
                  f"{time.time() - te:.0f}s {'*' if improved else ''}")
        if bad >= cfg.patience:
            if verbose:
                print(f"  early stopping: no val-loss improvement for {cfg.patience} epochs")
            break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    hist = pd.DataFrame(history)
    hist.to_csv(out_dir / f"{cfg.name}_history.csv", index=False)
    best = hist.loc[hist.epoch == best_epoch].iloc[0]
    summary = dict(name=cfg.name, arch=cfg.arch, pretrained=cfg.pretrained,
                   freeze_backbone=cfg.freeze_backbone, params_M=round(total / 1e6, 2),
                   trainable_M=round(trainable / 1e6, 2), epochs_run=len(hist),
                   best_epoch=best_epoch, best_val_loss=float(best.val_loss),
                   best_val_acc=float(best.val_acc), train_acc_at_best=float(best.train_acc),
                   train_minutes=round((time.time() - t0) / 60, 2))
    with open(out_dir / f"{cfg.name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return model, hist, summary


# --------------------------------------------------------------------------- #
# Grad-CAM (for error analysis)
# --------------------------------------------------------------------------- #
def grad_cam(model, x: torch.Tensor, target_class: int | None = None) -> np.ndarray:
    """Grad-CAM heat-map for a timm CNN (uses forward_features / forward_head).
    x: (1,3,H,W) normalised tensor on the model's device. Returns (H,W) in [0,1]."""
    model.eval()
    x = x.clone().requires_grad_(True)   # makes sure a graph exists even if the backbone is frozen
    feats = model.forward_features(x)
    if feats.ndim != 4:
        raise ValueError("Grad-CAM needs a CNN feature map (B,C,H,W)")
    feats.retain_grad()
    logits = model.forward_head(feats)
    cls = int(logits.argmax(1)) if target_class is None else target_class
    model.zero_grad(set_to_none=True)
    logits[0, cls].backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feats).sum(1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam
