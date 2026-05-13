import pandas as pd
import numpy as np
import os
import re
import argparse
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from scipy import stats
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


def compute_metrics(y_true, y_pred, y_prob):
    """Compute all metrics given true labels, predicted labels, and predicted probabilities."""
    accuracy = (y_pred == y_true).mean()
    auroc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "accuracy": accuracy,
        "auroc": auroc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": prec,
        "f1": f1,
    }


def build_summary(exp_dict, split_name):
    """Build per-fold DataFrame and summary DataFrame for a given split."""
    metric_cols = ["Accuracy", "AUROC", "Sensitivity", "Specificity", "Precision", "F1"]
    df = pd.DataFrame(columns=["Experiment Name", "Dataset Name", "Fold"] + metric_cols)
    for (exp_name, dataset_name, fold), metrics in exp_dict.items():
        new_row = {
            "Experiment Name": exp_name,
            "Dataset Name": dataset_name,
            "Fold": fold,
            "Accuracy": metrics["accuracy"],
            "AUROC": metrics["auroc"],
            "Sensitivity": metrics["sensitivity"],
            "Specificity": metrics["specificity"],
            "Precision": metrics["precision"],
            "F1": metrics["f1"],
        }
        df.loc[len(df)] = new_row

    agg_dict = {col: ["mean", "std"] for col in metric_cols}
    summary = df.groupby(["Experiment Name", "Dataset Name"]).agg(agg_dict).reset_index()

    print(f"\n{'='*80}")
    print(f"  {split_name} Set Results")
    print(f"{'='*80}")
    print(summary)

    for metric_name in metric_cols:
        print(f'\n{split_name} {metric_name} Mean and Std:')
        for dataset_name in summary["Dataset Name"].unique():
            print(f"{dataset_name} mean std")
            dataset_result = summary[summary["Dataset Name"] == dataset_name]
            for _, row in dataset_result.iterrows():
                exp_name = row["Experiment Name"].values[0]
                m_mean = row[(metric_name, 'mean')]
                m_std = row[(metric_name, 'std')]
                print(exp_name, m_mean, m_std)

    return df, summary


def paired_ttest_sl_vs_mil(df_result, split_name):
    """For each Experiment Name, do a one-sided paired t-test (SL > MIL) across folds."""
    metric_cols = ["Accuracy", "AUROC", "Sensitivity", "Specificity", "Precision", "F1"]
    exp_names = df_result["Experiment Name"].unique()

    print(f"\n{'='*80}")
    print(f"  {split_name} Set — Paired t-test  (H1: SL > MIL)")
    print(f"{'='*80}")

    rows = []
    for exp_name in exp_names:
        sl_df = df_result[(df_result["Experiment Name"] == exp_name) & (df_result["Dataset Name"] == "SL")].sort_values("Fold")
        mil_df = df_result[(df_result["Experiment Name"] == exp_name) & (df_result["Dataset Name"] == "MIL")].sort_values("Fold")

        # Only proceed when both SL and MIL have the same folds
        if len(sl_df) == 0 or len(mil_df) == 0:
            continue
        merged = pd.merge(sl_df, mil_df, on=["Experiment Name", "Fold"], suffixes=("_SL", "_MIL"))

        row = {"Experiment Name": exp_name}
        for metric in metric_cols:
            sl_vals = merged[f"{metric}_SL"].astype(float).values
            mil_vals = merged[f"{metric}_MIL"].astype(float).values
            diff = sl_vals - mil_vals
            if np.std(diff) == 0:
                p_value = 1.0 if np.mean(diff) <= 0 else 0.0
            else:
                # one-sided paired t-test: H1: SL > MIL
                t_stat, p_value = stats.ttest_rel(sl_vals, mil_vals, alternative='greater')
            row[f"{metric}_p"] = p_value
        rows.append(row)

    if rows:
        ttest_df = pd.DataFrame(rows)
        print(ttest_df.to_string(index=False))
    else:
        print("No matching SL/MIL pairs found.")

    return rows


if __name__ == "__main__":
    args = parse_args()
    
    dataset_list = ["SL", "MIL"]
    exp_list = get_exp_list(args.log_dir, dataset_list)

    val_exp_dict = {}
    test_exp_dict = {}
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

        # Compute validation set metrics
        val_df_grouped["pred_label"] = (val_df_grouped["pred_class"] >= best_threshold).astype(int)
        val_metrics = compute_metrics(
            val_df_grouped["true_label"].values,
            val_df_grouped["pred_label"].values,
            val_df_grouped["pred_class"].values
        )
        val_metrics["best_threshold"] = best_threshold
        val_exp_dict[(exp_name, dataset_name, fold)] = val_metrics

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

        # Compute test set metrics
        test_metrics = compute_metrics(
            test_df_grouped["true_label"].values,
            test_df_grouped["pred_label"].values,
            test_df_grouped["pred_class"].values
        )
        test_metrics["best_threshold"] = best_threshold
        test_exp_dict[(exp_name, dataset_name, fold)] = test_metrics

    # Build and print summaries for both validation and test sets
    val_df_result, val_summary_df = build_summary(val_exp_dict, "Validation")
    test_df_result, test_summary_df = build_summary(test_exp_dict, "Test")

    # Paired t-test: SL vs MIL (H1: SL > MIL)
    paired_ttest_sl_vs_mil(val_df_result, "Validation")
    paired_ttest_sl_vs_mil(test_df_result, "Test")
