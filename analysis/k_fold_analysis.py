import re
import os
import argparse
import pandas as pd
# import matplotlib.pyplot as plt


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
        name_parse = re.match(rf"(.+?)_({dataset_names_pattern})_FOLD(\d+)_finetune", folder)
        if name_parse:
            exp_name = name_parse.group(1)
            dataset_name = name_parse.group(2)
            fold = name_parse.group(3)
            test_file = os.path.join(log_dir, folder, "metrics_test.csv")
            test_file_list.append((exp_name, dataset_name, fold, test_file))
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
    data_frame = pd.DataFrame(columns=["Experiment Name", "Dataset Name", "Fold"])
    for exp_name, dataset_name, fold, test_file in test_file_list:
        if not os.path.exists(test_file):
            # raise FileNotFoundError(f"Test file not found: {test_file}")
            print(f"Test file not found: {test_file}")
            continue

        col_info = read_and_parse_csv(test_file)
        columns_names = col_info.keys()
        # Add new columns to the data frame if they don't exist
        for col in columns_names:
            if col not in data_frame.columns:
                data_frame[col] = None

        # Append the values to the data frame
        new_row = {"Experiment Name": exp_name, "Dataset Name": dataset_name, "Fold": fold}
        for col in columns_names:
            new_row[col] = col_info[col]
        data_frame.loc[len(data_frame)] = new_row
    return data_frame


def test_scores(dataframe):
    # Compute the mean and standard deviation for each dataset
    metrics = [col for col in dataframe.columns if col not in ["Experiment Name", "Dataset Name", "Fold", "Epoch"]]

    # Group the data frame by dataset name and experiment name, and compute the mean and standard deviation for each metric
    # dataframe_grouped = dataframe.groupby(["Dataset Name", "Experiment Name"])[metrics].agg(["mean", "std"])
    dataframe_grouped = dataframe.groupby(["Dataset Name", "Experiment Name"])

    # Test each group have 5 rows (5 seeds) and drop the groups that don't have 5 rows
    for name, group in dataframe_grouped:
        # print(f"Group {name} has {len(group)} rows.")
        if len(group) != 5:
            print(f"Warning: Group {name} has {len(group)} rows, expected 5.")
    
    dataframe_grouped = dataframe_grouped[metrics].agg(["mean", "std"])

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
        # 'HK',
        # 'MIL',
        # 'SL'
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

    # Print the test scores
    print('accuracy mean and std:')
    for dataset_name in test_result.index.get_level_values("Dataset Name").unique():
        print(f"{dataset_name} mean std")
        dataset_result = test_result.loc[dataset_name]
        for exp_name in dataset_result.index:
            mean_std = dataset_result.loc[exp_name]
            metric = 'accuracy'
            print(exp_name, mean_std[metric, 'mean'], mean_std[metric, 'std'])
            # for metric in ['accuracy']:
            #     mean_value = mean_std.loc[metric, "mean"]
            #     std_value = mean_std.loc[metric, "std"]
            #     print(f"    {metric}: {mean_value} ± {std_value}")
    # Draw the test scores
    # draw_test_scores(test_result, args.output_dir)

    print("\n\n")

    print('roc_auc mean and std:')
    for dataset_name in test_result.index.get_level_values("Dataset Name").unique():
        print(f"{dataset_name} mean std")
        dataset_result = test_result.loc[dataset_name]
        for exp_name in dataset_result.index:
            mean_std = dataset_result.loc[exp_name]
            metric = 'roc_auc'
            print(exp_name, mean_std[metric, 'mean'], mean_std[metric, 'std'])
    
