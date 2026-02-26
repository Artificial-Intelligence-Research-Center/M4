import os
import shutil
import random
import argparse
from tqdm import tqdm

def count_images(data_path):
    total_images = 0
    class_counts = {}

    for root, _, files in tqdm(os.walk(data_path), desc="Counting images"):
        if len(root.split('/')) != 3:
            continue

        _, data_split, class_name = root.split('/')
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


def check_file_num(class_counts1, class_counts2):
    for split in class_counts1:
        if split not in class_counts2:
            raise ValueError(f"Split {split} is missing in the second dataset")

        for class_name in class_counts1[split]:
            if class_name not in class_counts2[split]:
                raise ValueError(f"Class {class_name} is missing in split {split} of the second dataset")
            count1 = class_counts1[split][class_name]
            count2 = class_counts2[split][class_name]

            if count1 != count2:
                raise ValueError(f"Mismatch in {split} - {class_name}: {count1} vs {count2}")


def get_file_paths(data_path):
    image_paths = {}
    for root, _, files in os.walk(data_path):
        # Split image by class
        if len(root.split('/')) != 3:
            continue
        _, data_split, class_name = root.split('/')
        if data_split not in ['train', 'val', 'test']:
            continue
        if class_name not in image_paths:
            image_paths[class_name] = []
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                image_paths[class_name].append(os.path.join(root, filename))
    return image_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Count images in a dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--seed", type=int, help="Random seed for shuffling data")
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

    # create new folders for n-fold cross validation
    output_dir = os.path.join(args.data_path + f'_seed{args.seed}')
    if os.path.exists(output_dir):
        raise ValueError(f"Output directory {output_dir} already exists. Please choose a different seed or remove the existing directory.")

    # Get file paths
    image_paths = get_file_paths(args.data_path)

    # Shuffle data
    for class_name in image_paths:
        image_paths[class_name].sort()  # Ensure consistent order before shuffling
        if args.seed is not None:
            random.seed(args.seed)
            random.shuffle(image_paths[class_name])

    # Create new folders and distribute images
    training_set_dir = os.path.join(output_dir, 'train')
    validation_set_dir = os.path.join(output_dir, 'val')
    test_set_dir = os.path.join(output_dir, 'test')
    os.makedirs(training_set_dir, exist_ok=True)
    os.makedirs(validation_set_dir, exist_ok=True)
    os.makedirs(test_set_dir, exist_ok=True)


    # Distribute images according to the counts in class_counts
    for class_name in tqdm(class_counts['train'].keys()):
        class_train_dir = os.path.join(training_set_dir, class_name)
        class_val_dir = os.path.join(validation_set_dir, class_name)
        class_test_dir = os.path.join(test_set_dir, class_name)
        os.makedirs(class_train_dir, exist_ok=True)
        os.makedirs(class_val_dir, exist_ok=True)
        os.makedirs(class_test_dir, exist_ok=True)

        training_num = class_counts['train'][class_name]
        validation_num = class_counts['val'][class_name]
        test_num = class_counts['test'][class_name]

        training_data = image_paths[class_name][:training_num]
        validation_data = image_paths[class_name][training_num:training_num + validation_num]
        test_data = image_paths[class_name][training_num + validation_num:]

        for path in training_data:
            shutil.copy(path, class_train_dir)
        for path in validation_data:
            shutil.copy(path, class_val_dir)
        for path in test_data:
            shutil.copy(path, class_test_dir)

    # Verify the new dataset
    new_total_images, new_class_counts = count_images(output_dir)
    print(f"Total images in new dataset: {new_total_images}")
    print("Class distribution in new dataset:")
    for root_class, class_dict in new_class_counts.items():
        for class_name, count in class_dict.items():
            print(f"{root_class}/{class_name}: {count}", end=' | ')
        print()  # New line after each root class

    for split in class_counts:
        for class_name in class_counts[split]:
            if split not in new_class_counts or class_name not in new_class_counts[split]:
                raise ValueError(f"Class {class_name} is missing in split {split} of the new dataset")
            count1 = class_counts[split][class_name]
            count2 = new_class_counts[split][class_name]
            if count1 != count2:
                raise ValueError(f"Mismatch in {split} - {class_name}: {count1} vs {count2}")
    print("New dataset verification passed.")