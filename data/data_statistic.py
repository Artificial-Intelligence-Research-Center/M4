import os
import argparse
from tqdm import tqdm


def count_images(data_path):
    total_images = 0
    class_counts = {}

    for root, dirs, files in tqdm(os.walk(data_path), desc="Counting images"):
        if len(root.split('/')) != 3:
            continue

        dataset_name, data_split, class_name = root.split('/')
        if data_split not in ['train', 'val', 'test']:
            continue

        # Filter out non-image files
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        num_images = len(image_files)
        total_images += num_images

        # Update class counts
        if data_split not in class_counts:
            class_counts[data_split] = {}
        class_counts[data_split][class_name] = num_images

    return total_images, class_counts


def parse_args():
    parser = argparse.ArgumentParser(description="Count images in a dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    total_images, class_counts = count_images(args.data_path)

    print(f"Total images: {total_images}")
    print("Class distribution:")
    for root_class, class_dict in class_counts.items():
        for class_name, count in class_dict.items():
            print(f"{root_class}/{class_name}: {count}", end=' | ')
        print()  # New line after each root class