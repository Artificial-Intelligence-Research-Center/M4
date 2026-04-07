import pandas as pd
import numpy as np
import os
import re
import argparse
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm
# For loop each experiment
# Read validation prediction and test prediction


def parse_args():
    parser = argparse.ArgumentParser(description="GastroNet MIL Analysis")
    parser.add_argument("--log_dir", type=str)
    return parser.parse_args()


def get_exp_list(log_dir, dataset_names):
    # Get folders in the log directory
    folders = [f for f in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, f))]

    # Create a regex pattern to match the dataset names
    dataset_names_pattern = "|".join(dataset_names)

    file_list = []
    for folder in folders:
        name_parse = re.match(rf"(.+?)_({dataset_names_pattern})_FOLD(\d+)_finetune", folder)
        if name_parse:
            exp_name = name_parse.group(1)
            dataset_name = name_parse.group(2)
            fold = name_parse.group(3)
            val_file = os.path.join(log_dir, folder, "predictions_val.csv")
            test_file = os.path.join(log_dir, folder, "predictions_test.csv")
            file_list.append([exp_name, dataset_name, fold, val_file, test_file])
        else:
            print(f"Folder name does not match the expected pattern: {folder}")
    return file_list



if __name__ == "__main__":
    args = parse_args()
    
    dataset_list = ["SL", "MIL"]
    exp_list = get_exp_list(args.log_dir, dataset_list)

    exp_dict = {}
    for exp_name, dataset_name, fold, val_file, test_file in tqdm(exp_list):
        if not os.path.exists(val_file):
            print(f"Validation file not found: {val_file}")
            continue

        val_df = pd.read_csv(val_file)
        val_df["patient_id"] = val_df["image_path"].apply(lambda x: os.path.basename(x).split("_")[0])

        val_df_grouped = val_df.groupby("patient_id").agg({
            "pred_class": "mean",
            "true_label": "first"
        }).reset_index()

        # Get the best threshold based on validation set
        pred = val_df_grouped["pred_class"].values
        true = val_df_grouped["true_label"].values
        fpr, tpr, thresholds = roc_curve(true, pred)
        # Youden's J statistic to find the optimal threshold
        best_threshold = thresholds[np.argmax(tpr - fpr)]


        if not os.path.exists(test_file):
            print(f"Test file not found: {test_file}")
            continue
        
        test_df = pd.read_csv(test_file)
        test_df["patient_id"] = test_df["image_path"].apply(lambda x: os.path.basename(x).split("_")[0])

        test_df_grouped = test_df.groupby("patient_id").agg({
            "pred_class": "mean",
            "true_label": "first"
        }).reset_index()

        # Apply the best threshold to get binary predictions
        test_df_grouped["pred_label"] = (test_df_grouped["pred_class"] >= best_threshold).astype(int)
        # Get accuracy and auroc
        accuracy = (test_df_grouped["pred_label"] == test_df_grouped["true_label"]).mean()
        auroc = roc_auc_score(test_df_grouped["true_label"], test_df_grouped["pred_class"])

        exp_dict[(exp_name, dataset_name, fold)] = {
            "accuracy": accuracy,
            "auroc": auroc,
            "best_threshold": best_threshold
        }

    # Get the average mean accuracy/auroc and std for each fold and dataset 
    test_df = pd.DataFrame(columns=["Experiment Name", "Dataset Name", "Fold", "Accuracy", "AUROC"])
    for (exp_name, dataset_name, fold), metrics in exp_dict.items():
        new_row = {
            "Experiment Name": exp_name,
            "Dataset Name": dataset_name,
            "Fold": fold,
            "Accuracy": metrics["accuracy"],
            "AUROC": metrics["auroc"]
        }
        test_df.loc[len(test_df)] = new_row
    
    summary_df = test_df.groupby(["Experiment Name", "Dataset Name"]).agg({
        "Accuracy": ["mean", "std"],
        "AUROC": ["mean", "std"]
    }).reset_index()

    print(summary_df)
    
    print('Accuracy Mean and Std:')
    # 因為前面 reset_index() 了，我們直接從欄位拿 unique 的 Dataset Name
    for dataset_name in summary_df["Dataset Name"].unique():
        print(f"{dataset_name} mean std")
        # 篩選該 dataset 的資料
        dataset_result = summary_df[summary_df["Dataset Name"] == dataset_name]
        
        for _, row in dataset_result.iterrows():
            exp_name = row["Experiment Name"].values[0] # 因為是 MultiIndex 欄位，需處理取值
            
            # 提取數值
            acc_mean = row[('Accuracy', 'mean')]
            acc_std = row[('Accuracy', 'std')]
            print(exp_name, acc_mean, acc_std)

    print('\n\n')
    print('roc_auc mean and std:')
    for dataset_name in summary_df["Dataset Name"].unique():
        print(f"{dataset_name} mean std")
        
        dataset_result = summary_df[summary_df["Dataset Name"] == dataset_name]
        
        for _, row in dataset_result.iterrows():
            exp_name = row["Experiment Name"].values[0]
            
            auc_mean = row[('AUROC', 'mean')]
            auc_std = row[('AUROC', 'std')]
            
            print(exp_name, auc_mean, auc_std)


