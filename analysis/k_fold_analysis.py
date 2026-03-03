import re
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def arg_parser():
    parser = argparse.ArgumentParser(description="K-Fold Cross Validation Analysis")
    parser.add_argument("--log_dir", type=str, default="output_dir_backup", help="Path to the log directory")
    parser.add_argument("--output_dir", type=str, default="analysis/analysis_results", help="Path to save the analysis results")
    return parser.parse_args()


def get_test_files(log_dir, dataset_names):
    # Get folders in the log directory
    folders = [f for f in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, f))]

    # Create a regex pattern to match the dataset names
    dataset_names_pattern = "|".join(dataset_names)

    test_file_list = []
    for folder in folders:
        name_parse = re.match(rf"(.+?)_({dataset_names_pattern})_seed(\d+)_finetune", folder)
        if name_parse:
            exp_name = name_parse.group(1)
            dataset_name = name_parse.group(2)
            seed = name_parse.group(3)
            test_file = os.path.join(log_dir, folder, "metrics_test.csv")
            test_file_list.append((exp_name, dataset_name, seed, test_file))
        else:
            print(f"Folder name does not match the expected pattern: {folder}")
    return test_file_list


def read_and_parse_csv(test_file):
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    df = pd.read_csv(test_file)
    col_info = {}
    for col in df.columns:
        value = df[col].iloc[0]
        col_info[col] = value
    return col_info


def build_dataframe(test_file_list):
    data_frame = pd.DataFrame(columns=["Experiment Name", "Dataset Name", "Seed"])
    for exp_name, dataset_name, seed, test_file in test_file_list:
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")

        col_info = read_and_parse_csv(test_file)
        columns_names = col_info.keys()
        # Add new columns to the data frame if they don't exist
        for col in columns_names:
            if col not in data_frame.columns:
                data_frame[col] = None

        # Append the values to the data frame
        new_row = {"Experiment Name": exp_name, "Dataset Name": dataset_name, "Seed": seed}
        for col in columns_names:
            new_row[col] = col_info[col]
        data_frame.loc[len(data_frame)] = new_row
    return data_frame


def test_scores(dataframe):
    # Compute the mean and standard deviation for each dataset
    metrics = [col for col in dataframe.columns if col not in ["Experiment Name", "Dataset Name", "Seed", "Epoch"]]

    # Group the data frame by dataset name and experiment name, and compute the mean and standard deviation for each metric
    dataframe_grouped = dataframe.groupby(["Dataset Name", "Experiment Name"])[metrics].agg(["mean", "std"])

    return dataframe_grouped


if __name__ == "__main__":
    args = arg_parser()

    # Initialize the dataset names
    dataset_names = [
        'APTOS2019',
        'Glaucoma_fundus',
        'IDRiD_data',
        'MESSIDOR2',
        'PAPILA',
        'Retina',
    ]

    # Initialize a dictionary to store the results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Get folders in the log directory
    folders = [f for f in os.listdir(args.log_dir) if os.path.isdir(os.path.join(args.log_dir, f))]

    # Get the test files
    test_file_list = get_test_files(args.log_dir, dataset_names)

    # Build the final dataframe
    dataframe = build_dataframe(test_file_list)

    # Compute the test scores
    test_result = test_scores(dataframe)

    output_csv_path = os.path.join(args.output_dir, "test_scores.csv")
    test_result.to_csv(output_csv_path)

    # Draw the test scores
    # draw_test_scores(test_result, args.output_dir)


