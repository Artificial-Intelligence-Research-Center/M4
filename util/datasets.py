import os
import pandas as pd
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


class MultiLabelDataset(torch.utils.data.Dataset):
    """Image dataset backed by a CSV containing one binary column per class."""

    def __init__(self, manifest, image_root, transform, image_column="image_path", classes=None):
        self.manifest = manifest
        self.image_root = image_root
        self.transform = transform
        frame = pd.read_csv(manifest)
        if image_column not in frame.columns:
            raise ValueError(f"Missing image column '{image_column}' in {manifest}")

        self.classes = list(classes) if classes else [c for c in frame.columns if c != image_column]
        if not self.classes:
            raise ValueError(f"No label columns found in {manifest}")
        missing = [column for column in self.classes if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing label columns in {manifest}: {missing}")

        labels = frame[self.classes].apply(pd.to_numeric, errors="raise")
        values = labels.to_numpy(dtype="float32").copy()
        if not ((values == 0) | (values == 1)).all():
            raise ValueError(f"Multi-label columns in {manifest} must contain only 0 or 1")

        self.paths = frame[image_column].astype(str).tolist()
        self.targets = torch.from_numpy(values)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        if not os.path.isabs(path):
            path = os.path.join(self.image_root, path)
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, self.targets[index], path


def get_dataset_classes(dataset):
    if isinstance(dataset, Subset):
        return dataset.dataset.classes
    return dataset.classes

def resolve_multilabel_manifest(split, args):
    manifest = args.multilabel_manifest.format(split=split)
    return manifest if os.path.isabs(manifest) else os.path.join(args.data_path, manifest)


def dataset_exists(split, args):
    if args.classification_type == "multi_label":
        return os.path.isfile(resolve_multilabel_manifest(split, args))
    return os.path.isdir(os.path.join(args.data_path, split))


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    if args.classification_type == "multi_label":
        manifest = resolve_multilabel_manifest(is_train, args)
        classes = [item.strip() for item in args.multilabel_classes.split(",") if item.strip()]
        if args.multilabel_image_root:
            image_root = args.multilabel_image_root
            if not os.path.isabs(image_root):
                image_root = os.path.join(args.data_path, image_root)
        else:
            image_root = args.data_path
        dataset = MultiLabelDataset(
            manifest=manifest,
            image_root=image_root,
            transform=transform,
            image_column=args.multilabel_image_column,
            classes=classes or None,
        )
        if len(dataset.classes) != args.nb_classes:
            raise ValueError(
                f"{manifest} defines {len(dataset.classes)} classes, but --nb_classes={args.nb_classes}"
            )
    else:
        root = os.path.join(args.data_path, is_train)
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

