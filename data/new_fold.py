import os
import shutil
import math
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
        if data_split not in ['train', 'val']:
            continue
        if class_name not in image_paths:
            image_paths[class_name] = []
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                image_paths[class_name].append(os.path.join(root, filename))
    return image_paths


def get_test_file_paths(data_path):
    image_paths = {}
    for root, _, files in os.walk(data_path):
        # Split image by class
        if len(root.split('/')) != 3:
            continue
        _, data_split, class_name = root.split('/')
        if data_split not in ['test']:
            continue
        if class_name not in image_paths:
            image_paths[class_name] = []
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                image_paths[class_name].append(os.path.join(root, filename))
    return image_paths


def copy_with_unique_name(src_path, dest_dir):
    """Copy src_path into dest_dir. If a file with the same name exists,
    append an incremental suffix to the basename until the name is unique.
    Returns the destination path used.
    """
    os.makedirs(dest_dir, exist_ok=True)
    base = os.path.basename(src_path)
    name, ext = os.path.splitext(base)
    dest_path = os.path.join(dest_dir, base)
    counter = 1
    while os.path.exists(dest_path):
        dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
        counter += 1
    shutil.copy(src_path, dest_path)
    return dest_path


def parse_args():
    parser = argparse.ArgumentParser(description="Count images in a dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--seed", type=int, help="Random seed for shuffling data")
    parser.add_argument("--fold", type=int, help="Number of folds for cross validation")
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
    test_image_paths = get_test_file_paths(args.data_path)
    # Shuffle the data for each class
    random.seed(args.seed)
    for class_name in image_paths:
        random.shuffle(image_paths[class_name])

    for fold_num in range(args.fold):
        # Create new folders and distribute images
        training_set_dir = os.path.join(f"{output_dir}_fold{fold_num}", 'train')
        validation_set_dir = os.path.join(f"{output_dir}_fold{fold_num}", 'val')
        test_set_dir = os.path.join(f"{output_dir}_fold{fold_num}", 'test')

        # Check if the training set directory already exists to avoid overwriting existing data
        if os.path.exists(training_set_dir):
            raise ValueError(f"Training set directory {training_set_dir} already exists. Please choose a different seed or remove the existing directory.")

        os.makedirs(training_set_dir, exist_ok=True)
        os.makedirs(validation_set_dir, exist_ok=True)
        os.makedirs(test_set_dir, exist_ok=True)


        for class_name in tqdm(class_counts['train'].keys()):
            class_train_dir = os.path.join(training_set_dir, class_name)
            class_val_dir = os.path.join(validation_set_dir, class_name)
            class_test_dir = os.path.join(test_set_dir, class_name)
            os.makedirs(class_train_dir, exist_ok=True)
            os.makedirs(class_val_dir, exist_ok=True)
            os.makedirs(class_test_dir, exist_ok=True)

            total_num = class_counts['train'][class_name] + class_counts['val'][class_name]
            training_num = math.floor(total_num * ((args.fold - 1) / args.fold))  # Ensure that training_num is at least 1
            # validation_num = math.ceil(total_num / args.fold)  # Ensure that validation_num is at least 1

            training_data = image_paths[class_name][:training_num]
            validation_data = image_paths[class_name][training_num:]
            test_data = test_image_paths[class_name]

            for path in training_data:
                copy_with_unique_name(path, class_train_dir)
            for path in validation_data:
                copy_with_unique_name(path, class_val_dir)
            for path in test_data:
                copy_with_unique_name(path, class_test_dir)

            # Rotate the data for the next fold
            # print(image_paths[class_name][:training_num])
            # print(image_paths[class_name][training_num:])
            image_paths[class_name] = image_paths[class_name][training_num:] + image_paths[class_name][:training_num]
            # input()

            # Verify the new dataset
        new_total_images, new_class_counts = count_images(f"{output_dir}_fold{fold_num}")
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