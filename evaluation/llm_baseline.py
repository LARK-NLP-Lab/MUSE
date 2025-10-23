
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='path to model results folder')
    parser.add_argument('--task', type=str)
    args = parser.parse_args()
    return args


args = parse_args()
LLM_NAME = args.model
TASK_NAME = args.task


print(f"Currently processing LLM {LLM_NAME} on Task {TASK_NAME}")
# define y_true
# test_df = pd.read_csv(f"../binary_selfconsistency/{LLM_NAME}/bootstrapped_{TASK_NAME}.csv")
test_df = pd.read_csv(f'../final_results/MIMIC_extract/full/deepseek-distill-qwen/output/{TASK_NAME}_results_round1.csv')
y_true = test_df["output"]


def compute_ece(y_true, y_probs, n_bins=10):
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        mask = (y_probs > bin_lower) & (y_probs <= bin_upper)
        if np.any(mask):
            bin_accuracy = np.mean(y_true[mask])
            bin_confidence = np.mean(y_probs[mask])
            ece += np.abs(bin_accuracy - bin_confidence) * np.sum(mask) / len(y_true)
    return ece

# ======================= 1. Sequence Likelihood =========================

r1_df = pd.read_csv(f"../final_results/MIMIC_extract/SLL_full/{LLM_NAME}/norm_probs/{TASK_NAME}_results_round1.csv")
# r1_df = r1_df[1000]
r1_df["gold"] = r1_df["gold"].map({"Yes": 1, "No": 0}).astype(int)
y_true = r1_df["gold"]
y_probs = r1_df["norm_yes"]
print("=== Sequence Likelihood (Single Run) ===")
print("AUROC:", roc_auc_score(y_true, y_probs))
print("ECE:", compute_ece(y_true, y_probs, n_bins=10))
print('Brier Score:', brier_score_loss(y_true, y_probs))

file_paths = sorted(glob(f"../final_results/MIMIC_extract/SLL_full/{LLM_NAME}/norm_probs/{TASK_NAME}_results_round*.csv"))
all_runs = []
for i, path in enumerate(file_paths):
    df = pd.read_csv(path)
    df = df[["id", "prob_yes", "prob_no", "gold"]].copy()
    df = df.rename(columns={
        "prob_yes": f"prob_yes_{i}",
        "prob_no": f"prob_no_{i}"
    })
    all_runs.append(df)

df_merged = all_runs[0]
for df in all_runs[1:]:
    df_merged = df_merged.merge(df, on=["id", "gold"])

prob_yes_cols = [col for col in df_merged.columns if col.startswith("prob_yes_")]
df_merged["prob_yes_mean"] = df_merged[prob_yes_cols].mean(axis=1)

df_final = df_merged[["id", "gold", "prob_yes_mean"]]
df_final.to_csv(f"../final_results/MIMIC_extract/SLL_full/{LLM_NAME}/{TASK_NAME}_aggregated_seq_probs.csv", index=False)
print("=== Sequence Likelihood (Averaged Over Runs) ===")
print("AUROC:", roc_auc_score(y_true, df_final["prob_yes_mean"]))
print("ECE:", compute_ece(y_true, df_final["prob_yes_mean"]))
print('Brier Score:', brier_score_loss(y_true, df_final['prob_yes_mean']))


# ======================= 2. Self-Consistency (LLM Generation) =========================

# NUM_INSTANCES = 1000
NUM_INSTANCES = 4790
# NUM_INSTANCES = 3790
NUM_ROUNDS = 10
BOOTSTRAP_ITERATIONS = 15
SAMPLE_SIZE = 9  # 90% of 10 rounds
file_template = f"../final_results/MIMIC_extract/full/{LLM_NAME}/parsed/{TASK_NAME}_results_round{{}}.csv"
# file_template = f"../bin_sc_full_fast/random_False/{LLM_NAME}/parsed/{TASK_NAME}_results_round{{}}.csv"

# y_true = y_true[1000:]

all_predictions = [[] for _ in range(NUM_INSTANCES)]
for i in range(1, NUM_ROUNDS + 1):
    df = pd.read_csv(file_template.format(i))
    preds = df["prediction"].astype(str).tolist()
    for j in range(NUM_INSTANCES):
        all_predictions[j].append(preds[j])

# One-time average version
avg_probs = []
for preds in all_predictions:
    valid_preds = [int(p) for p in preds if p in ['0', '1']]
    avg_probs.append(np.mean(valid_preds) if valid_preds else np.nan)

avg_probs = np.array(avg_probs)
valid_mask = ~np.isnan(avg_probs)
avg_y_true = np.array(y_true)[valid_mask]
print("=== Self-Consistency (One-time Average) ===")
print("AUROC:", roc_auc_score(avg_y_true, avg_probs[valid_mask]))
print("ECE:", compute_ece(avg_y_true, avg_probs[valid_mask]))
print('Brier Score:', brier_score_loss(avg_y_true, avg_probs[valid_mask]))

# Bootstrap version
def bootstrap_prob_yes(valid_preds, n_iter=15, sample_size=9):
    results = []
    for _ in range(n_iter):
        sample = np.random.choice(valid_preds, size=sample_size, replace=True)
        prob_yes = np.mean([int(p) for p in sample])
        results.append(prob_yes)
    return results

bootstrap_results = []
invalid_indices = [
    idx for idx, preds in enumerate(all_predictions)
    if all(p not in ['0', '1'] for p in preds)
]

for idx, preds in enumerate(all_predictions):
    if idx in invalid_indices:
        bootstrap_results.append([np.nan] * BOOTSTRAP_ITERATIONS)
    else:
        valid_preds = [p for p in preds if p in ['0', '1']]
        bootstrap_results.append(bootstrap_prob_yes(valid_preds))

bootstrap_df = pd.DataFrame(bootstrap_results)
bootstrap_df.insert(0, "Instance ID", list(range(NUM_INSTANCES)))
bootstrap_df.to_csv(f"../final_results/MIMIC_extract/full/{LLM_NAME}/bootstrap_prob_yes_per_instance_{TASK_NAME}.csv", index=False)
# bootstrap_df.to_csv(f"../bin_sc_full_fast/random_False/{LLM_NAME}/bootstrap_prob_yes_per_instance_{TASK_NAME}.csv", index=False)

bootstrap_means = bootstrap_df.iloc[:, 1:].mean(axis=1).values
filtered_y_true = np.array(y_true)[~np.isnan(bootstrap_means)]
filtered_probs = bootstrap_means[~np.isnan(bootstrap_means)]

print("=== Self-Consistency (Bootstrap Average) ===")
print("AUROC:", roc_auc_score(filtered_y_true, filtered_probs))
print("ECE:", compute_ece(filtered_y_true, filtered_probs))
print('Brier Score:', brier_score_loss(filtered_y_true, filtered_probs))