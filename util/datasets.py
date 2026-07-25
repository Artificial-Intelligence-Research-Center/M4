import hashlib
import os
import torch
from PIL import Image
from torch.utils.data import Subset
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class ImageFolderWithPath(datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path


def _resized_cache_loader(cache_dir, short_side):
    """回傳 ImageFolder loader: 從「短邊縮到 short_side 的磁碟快取」讀圖。

    高解析度資料集 (如 2576x1934 眼底照) 每次 __getitem__ 都 decode 整張大圖是
    GPU 利用率低的主因之一; 預縮快取把 decode 成本降 ~20 倍。快取 lazily 建立,
    多 worker 併發安全 (tmp 檔 + os.replace 原子替換); 快取毀損時 fallback 原圖。
    key 用 realpath 的 sha1 → symlink 指向同一實體檔的多個 fold 共用快取。
    """
    def loader(path):
        real = os.path.realpath(path)
        key = hashlib.sha1(real.encode("utf-8")).hexdigest()
        sub = os.path.join(cache_dir, key[:2])
        cpath = os.path.join(sub, f"{key}_s{short_side}.jpg")
        if os.path.isfile(cpath):
            try:
                with open(cpath, "rb") as f:
                    return Image.open(f).convert("RGB")
            except Exception:
                pass  # 快取毀損 → 重建
        with open(real, "rb") as f:
            img = Image.open(f).convert("RGB")
        w, h = img.size
        if min(w, h) > short_side:
            scale = short_side / min(w, h)
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                             Image.BICUBIC)
            try:
                os.makedirs(sub, exist_ok=True)
                tmp = cpath + f".tmp{os.getpid()}"
                img.save(tmp, format="JPEG", quality=95)
                os.replace(tmp, cpath)
            except Exception:
                pass  # 快取寫入失敗不影響訓練
        return img
    return loader


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    # 預縮圖快取 (--cache_resized N; 預設 0=關, 行為與原版完全一致)。
    # samples/path 仍是原圖路徑, 只有 loader 讀的來源換成快取小圖。
    n = int(getattr(args, "cache_resized", 0) or 0)
    if n > 0:
        cache_dir = getattr(args, "cache_dir", "./data/_resize_cache")
        loader = _resized_cache_loader(cache_dir, n)
        dataset = ImageFolderWithPath(root, transform=transform, loader=loader)
    else:
        dataset = ImageFolderWithPath(root, transform=transform)

    if is_train == 'train':
        ratio = float(getattr(args, "dataratio", 1.0))
        seed = int(getattr(args, "seed", 0))
        stratified = bool(getattr(args, "stratified", False))

        if 0.0 < ratio < 1.0:
            if stratified:
                idx = _stratified_indices(dataset.targets, ratio, seed)
            else:
                # simple uniform subsample with torch.Generator for reproducibility
                g = torch.Generator().manual_seed(seed)
                n = len(dataset)
                k = max(1, int(n * ratio))
                idx = torch.randperm(n, generator=g)[:k].tolist()
            dataset = Subset(dataset, idx)

    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train == 'train':
        return create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )

    # eval transform
    crop_pct = 224 / 256 if args.input_size <= 224 else 1.0
    size = int(args.input_size / crop_pct)
    t = [
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(t)

# ---- helpers ----

def _stratified_indices(targets, ratio: float, seed: int):
    """Maintain class proportions. Ensures at least 1 sample per class when possible."""
    t = torch.as_tensor(targets)
    classes = torch.unique(t)
    g = torch.Generator().manual_seed(seed)

    keep = []
    for c in classes.tolist():
        cls_idx = torch.nonzero(t == c, as_tuple=False).view(-1)
        if len(cls_idx) == 0:
            continue
        k = max(1, int(round(len(cls_idx) * ratio)))
        sel = cls_idx[torch.randperm(len(cls_idx), generator=g)[:k]]
        keep.extend(sel.tolist())

    # shuffle final indices (stable across seed)
    g2 = torch.Generator().manual_seed(seed + 1)
    keep = torch.tensor(keep)[torch.randperm(len(keep), generator=g2)].tolist()
    return keep

