import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.spatial.distance import jensenshannon


# -----------------------------
# METRIC FUNCTIONS
# -----------------------------
def binary_entropy(p):
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def compute_ece2(y_true, y_probs, n_bins=10):
    """Expected Calibration Error."""
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


# -----------------------------
# SUBSET SELECTION ALGORITHMS
# -----------------------------
def select_calibrated_subset_jsdiv_multillm(probs_all_llms, beta=1.0, epis_tol=0.04,
                                            min_size=20, wgt_avg=True):
    """Greedy JS divergence minimization."""
    dists = [np.array([p, 1 - p]) for p, _, _ in probs_all_llms]
    confidences = [abs(p - 0.5) for p, _, _ in probs_all_llms]
    sorted_indices = np.argsort(-np.array(confidences))
    if len(sorted_indices) == 0:
        print("Warning: No confident predictions to sort.")
        return None, None, []
    selected = [sorted_indices[0]]
    prev_js = 0.0

    for idx in sorted_indices[1:]:
        candidate = selected + [idx]
        cand_dists = [dists[i] for i in candidate]
        mean_dist = np.mean(cand_dists, axis=0)
        js_divs = [jensenshannon(d, mean_dist, base=2) ** 2 for d in cand_dists]
        curr_js = np.mean(js_divs)
        subset_probs = [d[0] for d in cand_dists]
        curr_alea = np.mean([binary_entropy(p) for p in subset_probs])

        if len(selected) >= min_size and (curr_js - prev_js) > epis_tol:
            break

        selected.append(idx)
        prev_js = curr_js

    final_probs = [dists[i][0] for i in selected]
    if wgt_avg:
        weights = [1 - binary_entropy(p) for p in final_probs]
        mean_prob = np.average(final_probs, weights=weights)
    else:
        mean_prob = np.mean(final_probs)
    total_uncertainty = prev_js + beta * np.mean([binary_entropy(p) for p in final_probs])
    return mean_prob, total_uncertainty, selected


def select_calibrated_subset_jsdiv_multillm_v2(probs_all_llms, beta=1.0, tol=0.04,
                                               min_size=20, wgt_avg=False):
    """Conservative variant that minimizes total (epistemic + aleatoric) uncertainty."""
    dists = [np.array([p, 1 - p]) for p, _, _ in probs_all_llms]
    confidences = [abs(p - 0.5) for p, _, _ in probs_all_llms]
    sorted_indices = np.argsort(-np.array(confidences))
    selected = [sorted_indices[0]]
    prev_total = float('inf')

    for idx in sorted_indices[1:]:
        candidate = selected + [idx]
        cand_dists = [dists[i] for i in candidate]
        mean_dist = np.mean(cand_dists, axis=0)
        js_divs = [jensenshannon(d, mean_dist, base=2) ** 2 for d in cand_dists]
        curr_js = np.mean(js_divs)
        subset_probs = [d[0] for d in cand_dists]
        curr_alea = np.mean([binary_entropy(p) for p in subset_probs])
        curr_total = curr_js + beta * curr_alea

        if len(selected) >= min_size and curr_total > prev_total - tol:
            break

        selected.append(idx)
        prev_total = curr_total

    final_probs = [dists[i][0] for i in selected]
    if wgt_avg:
        weights = [1 - binary_entropy(p) for p in final_probs]
        mean_prob = np.average(final_probs, weights=weights)
    else:
        mean_prob = np.mean(final_probs)
    final_alea = np.mean([binary_entropy(p) for p in final_probs])
    final_total_uncertainty = curr_js + beta * final_alea
    return mean_prob, final_total_uncertainty, selected


# -----------------------------
# CORE PIPELINE
# -----------------------------
def load_data(dataset, task, mode):
    """Load model outputs and ground-truth labels."""
    if dataset.upper() == "MIMIC":
        task_name = f'_{task}'
    else:
        task_name = ""


    if mode == "gen":
        ds_path = f"../data/generated/{dataset}/output/deepseek_bs_prob_yes{task_name}.csv"
        qw_path = f"../data/generated/{dataset}/output/qwen_bs_prob_yes{task_name}.csv"
        gm_path = f"../data/generated/{dataset}/output/gemma_bs_prob_yes{task_name}.csv"
        mis_path = f"../data/generated/{dataset}/output/mistral_bs_prob_yes{task_name}.csv"
    elif mode == "probs":
        ds_path = f"../data/generated/{dataset}/likelihood/deepseek_agg_seq_prob{task_name}.csv"
        qw_path = f"../data/generated/{dataset}/likelihood/qwen_agg_seq_prob{task_name}.csv"
        gm_path = f"../data/generated/{dataset}/likelihood/gemma_agg_seq_prob{task_name}.csv"
        mis_path = f"../data/generated/{dataset}/likelihood/mistral_agg_seq_prob{task_name}.csv"
    else:
        raise ValueError("Mode must be 'gen' or 'probs'.")

    # the fuck is this TODO
    raw_input = pd.read_csv(f"../data/generated/{dataset}/output/mistral_bs_prob_yes{task_name}.csv")
    y_true = raw_input["output"]
    raw_input = raw_input.drop(columns=['output', 'prediction'])

    # test_df = pd.read_csv(f"../binary_selfconsistency/deepseek-distill-qwen/bootstrapped_{task_name}.csv")
    # y_true = test_df["output"]

    llm_files = {"ds": ds_path, "qwen": qw_path, "mis": mis_path, "gem": gm_path}
    llm_bootstrap_data = {}
    for llm, path in llm_files.items():
        df = pd.read_csv(path)
        if "gold" in df.columns:
            df = df.drop(columns=["gold"])
        llm_bootstrap_data[llm] = df.iloc[:, 1:].values

    return task_name, raw_input, y_true, llm_bootstrap_data


def run_algo(task_name, raw_input, y_true, llm_bootstrap_data, algo_v="v1", wgt_avg=False, output_dir="ehrshot"):
    """Run calibration algorithm and compute metrics."""
    os.makedirs(output_dir, exist_ok=True)

    N = min(1000, len(y_true))
    calibrated_probs, uncertainties = [], []

    for i in range(N):
        all_preds = []
        for llm_name, llm_data in llm_bootstrap_data.items():
            preds = llm_data[i]
            for p in preds:
                if not np.isnan(p):
                    all_preds.append((p, 1 - p, llm_name))
        if len(all_preds) < 3:
            calibrated_probs.append(np.nan)
            uncertainties.append(np.nan)
            continue

        if algo_v == "v1":
            p_hat, unc, _ = select_calibrated_subset_jsdiv_multillm(all_preds, wgt_avg=wgt_avg)
        else:
            p_hat, unc, _ = select_calibrated_subset_jsdiv_multillm_v2(all_preds, wgt_avg=wgt_avg)
        calibrated_probs.append(p_hat)
        uncertainties.append(unc)

    mask = ~np.isnan(calibrated_probs)
    y_true_filtered = np.array(y_true)[:len(mask)][mask]
    p_hat_filtered = np.array(calibrated_probs)[mask]

    combined = pd.concat([
        raw_input.reset_index(drop=True).iloc[:len(y_true_filtered)],
        pd.DataFrame({"p_hat": p_hat_filtered, "y_true": y_true_filtered})
    ], axis=1)

    save_path = os.path.join(output_dir, f"{task_name}_{algo_v}_wgt_avg_{wgt_avg}.csv")
    combined.to_csv(save_path, index=False)

    auroc = roc_auc_score(y_true_filtered, p_hat_filtered)
    ece = compute_ece2(y_true_filtered, p_hat_filtered)
    brier_sc = brier_score_loss(y_true_filtered, p_hat_filtered)
    algo_name = {"v1": "Greedy", "v2": "Conserv"}
    print(f"[{algo_name[algo_v]}] AUROC={auroc:.4f}, ECE={ece:.4f}, Brier={brier_sc:.4f}")
    return auroc, ece, brier_sc


# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Multi-LLM Calibration Runner")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (MIMIC, EHRShot, TQA)")
    parser.add_argument("--task", type=str, help="Task name (required if dataset=MIMIC)")
    parser.add_argument("--mode", type=str, default="gen", choices=["gen", "probs"], help="Mode: 'gen' or 'probs'")
    args = parser.parse_args()

    if args.dataset.upper() == "MIMIC" and not args.task:
        parser.error("--task is required when dataset is MIMIC")

    task_name, raw_input, y_true, llm_bootstrap_data = load_data(args.dataset, args.task, args.mode)
    print(f"Running Multi-LLM Calibration for {args.dataset} ({task_name}), mode={args.mode}")

    # Run all algorithm configurations
    for algo_v in ["v1", "v2"]:
        for wgt_avg in [False, True]:
            run_algo(task_name, raw_input, y_true, llm_bootstrap_data, algo_v, wgt_avg)


if __name__ == "__main__":
    main()
