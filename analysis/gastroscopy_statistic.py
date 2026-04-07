import os
import argparse
from collections import defaultdict

def count_detailed_patients(root_dir):
    splits = ['train', 'val', 'test']
    
    # 結構：results[split][class_name] = set(patient_ids)
    results = {split: defaultdict(set) for split in splits}

    for split in splits:
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path):
            continue

        # 列出所有類別資料夾 (例如 class_0, class_1)
        classes = [c for c in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, c))]
        
        for cls_name in classes:
            cls_path = os.path.join(split_path, cls_name)
            for file in os.listdir(cls_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # 抓取病患 ID: 003_04.jpg -> 003
                    patient_id = file.split('_')[0]
                    results[split][cls_name].add(patient_id)

    # --- 格式化列印結果 ---
    print(f"{'Split':<10} | {'Class':<15} | {'Unique Patients':<15}")
    print("-" * 45)
    
    total_summary = []
    for split, classes in results.items():
        if not classes: continue
        
        # 依照類別名稱排序顯示
        for cls_name in sorted(classes.keys()):
            num_patients = len(classes[cls_name])
            print(f"{split:<10} | {cls_name:<15} | {num_patients:<15}")
            total_summary.append({
                "split": split,
                "class": cls_name,
                "count": num_patients
            })
        print("-" * 45) # 每個 Split 結束後畫橫線

    return total_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Count unique patients in each class for each split")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory of the dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    count_detailed_patients(args.dataset_root)