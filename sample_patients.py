"""
依照病患ID為單位，從 MIL fold 的 training 資料中抽取指定比例的資料。
陽性（class 1）與陰性（class 0）各自獨立隨機抽樣。
抽樣結果會複製到新的資料夾，原始資料不會被修改。

用法：
    python sample_patients.py --input_dir data/5_fold_MIL/MIL_seed42_fold0 \
                               --output_dir data/5_fold_MIL/MIL_seed42_fold0_75pct \
                               --ratio 0.75 \
                               --seed 42
"""

import argparse
import os
import random
import shutil
from collections import defaultdict
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="依病患ID抽取訓練資料")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/5_fold_MIL/MIL_seed42_fold0",
        help="來源 fold 資料夾路徑（包含 train/val/test 子資料夾）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="輸出資料夾路徑（預設為 input_dir + '_<ratio>pct'）",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.75,
        help="抽取比例，例如 0.75 代表抽取 75%% 的病患（預設：0.75）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="隨機種子（預設：42）",
    )
    parser.add_argument(
        "--train_subdir",
        type=str,
        default="train",
        help="訓練資料子資料夾名稱（預設：train）",
    )
    return parser.parse_args()


def get_patient_files(class_dir: str) -> Dict[str, List[str]]:
    """
    讀取指定類別資料夾，依病患ID分組。
    回傳 {patient_id: [filename, ...]} 的字典。
    """
    patient_map = defaultdict(list)
    for fname in sorted(os.listdir(class_dir)):
        if not os.path.isfile(os.path.join(class_dir, fname)):
            continue
        # 檔名格式：病患id_照片id.ext
        patient_id = fname.split("_")[0]
        patient_map[patient_id].append(fname)
    return dict(patient_map)


def sample_patients(patient_map: Dict, ratio: float, seed: int) -> Dict:
    """
    從 patient_map 中隨機抽取指定比例的病患，保留其所有照片。
    """
    patient_ids = sorted(patient_map.keys())
    n_sample = max(1, round(len(patient_ids) * ratio))

    rng = random.Random(seed)
    sampled_ids = rng.sample(patient_ids, n_sample)

    sampled = {pid: patient_map[pid] for pid in sampled_ids}
    return sampled


def copy_files(sampled: Dict, src_class_dir: str, dst_class_dir: str):
    """將抽樣到的檔案複製到目標資料夾。"""
    os.makedirs(dst_class_dir, exist_ok=True)
    for patient_id, files in sampled.items():
        for fname in files:
            src = os.path.join(src_class_dir, fname)
            dst = os.path.join(dst_class_dir, fname)
            shutil.copy2(src, dst)


def main():
    args = parse_args()

    if args.ratio <= 0 or args.ratio > 1:
        raise ValueError(f"--ratio 必須介於 0（不含）到 1 之間，目前為 {args.ratio}")

    # 自動設定輸出路徑
    if args.output_dir is None:
        pct = int(args.ratio * 100)
        args.output_dir = f"{args.input_dir.rstrip('/')}_{pct}pct"

    train_src = os.path.join(args.input_dir, args.train_subdir)
    train_dst = os.path.join(args.output_dir, args.train_subdir)

    if not os.path.isdir(train_src):
        raise FileNotFoundError(f"找不到訓練資料夾：{train_src}")

    class_dirs = sorted(
        d for d in os.listdir(train_src)
        if os.path.isdir(os.path.join(train_src, d))
    )

    print(f"來源資料夾  : {train_src}")
    print(f"輸出資料夾  : {train_dst}")
    print(f"抽取比例    : {args.ratio * 100:.1f}%")
    print(f"隨機種子    : {args.seed}")
    print(f"偵測到的類別: {class_dirs}\n")

    total_patients_before = 0
    total_patients_after = 0
    total_images_before = 0
    total_images_after = 0

    for cls in class_dirs:
        src_class_dir = os.path.join(train_src, cls)
        dst_class_dir = os.path.join(train_dst, cls)

        patient_map = get_patient_files(src_class_dir)
        sampled = sample_patients(patient_map, args.ratio, seed=args.seed + int(cls))

        n_patients_before = len(patient_map)
        n_patients_after = len(sampled)
        n_images_before = sum(len(v) for v in patient_map.values())
        n_images_after = sum(len(v) for v in sampled.values())

        copy_files(sampled, src_class_dir, dst_class_dir)

        print(f"  Class {cls}:")
        print(f"    病患數  : {n_patients_before} → {n_patients_after} ({n_patients_after/n_patients_before*100:.1f}%)")
        print(f"    影像數  : {n_images_before} → {n_images_after} ({n_images_after/n_images_before*100:.1f}%)")

        total_patients_before += n_patients_before
        total_patients_after += n_patients_after
        total_images_before += n_images_before
        total_images_after += n_images_after

    # 複製 val / test 資料夾（不做抽樣，直接複製）
    for subdir in os.listdir(args.input_dir):
        if subdir == args.train_subdir:
            continue
        src_sub = os.path.join(args.input_dir, subdir)
        dst_sub = os.path.join(args.output_dir, subdir)
        if os.path.isdir(src_sub):
            if os.path.exists(dst_sub):
                shutil.rmtree(dst_sub)
            shutil.copytree(src_sub, dst_sub)
            print(f"\n  已複製 {subdir}/ → {dst_sub}")

    print(f"\n完成！")
    print(f"  總病患數: {total_patients_before} → {total_patients_after}")
    print(f"  總影像數: {total_images_before} → {total_images_after}")
    print(f"  輸出位置: {args.output_dir}")


if __name__ == "__main__":
    main()
