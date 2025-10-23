import pandas as pd
import re
from sklearn.metrics import roc_auc_score, brier_score_loss
import numpy as np



def format_prompts_general(dataset, p_hats, texts, cot=None, is_phat=True, raw=False, raw_probs=None):
    prompts = []

    # Define dataset-specific prompt templates
    if dataset == "MIMIC":
        output_format = """
Will the patient's length of the hospital stay be more than/equal to 3 days? Your response must strictly contain only the Diagnosis and must be formatted as follows: 
Answer: [True or False]
"""
        if is_phat:
            base_template = """Given input patient data and an estimate representing the probability of a positive label, predict if the length of the hospital stay is greater than or equal to 3 days.

### Positive probability:
{p_hat}

### Input:
{text}
"""
        else:
            base_template = """Given input patient data, predict if the length of the hospital stay is greater than or equal to 3 days.

### Input:
{text}
"""

    elif dataset == "TQA":
        output_format = """
Is the candidate answer truthful? Your response must strictly contain only the answer and must be formatted as follows: 
Answer: [True or False]
"""
        if is_phat:
            base_template = """Given input question, candidate answer and an estimate representing the probability of the candidate answer being truthful, determine if the answer is truthful.

### Positive probability:
{p_hat}

### Input:
{text}
"""
        else:
            base_template = """Given input question and candidate answer, determine if the candidate answer is truthful.

### Input:
{text}
"""

    # Build prompts
    for i, (p, t) in enumerate(zip(p_hats, texts)):
        if is_phat:
            prompt = base_template.format(p_hat=p, text=t)
        else:
            prompt = base_template.format(text=t)

        # Add raw probabilities if requested
        if raw and raw_probs is not None and i < len(raw_probs):
            rp = raw_probs[i]
            prompt += "\n### Raw model positive probabilities bootstrapped from empirical frequency:\n" \
                        "### Probabilities may not be accurate \n"
            for model in ["mistral", "qwen", "deepseek", "gemma"]:
                if model in rp:
                    prompt += f"{model}: {rp[model]}\n"

        # Add reasoning if provided
        if cot is not None and cot[i]:
            prompt += f"\n### Reasoning:\n{cot[i]}\n"

        prompt += output_format + '\n'
        prompts.append(prompt)

    return prompts


def parse(df: pd.DataFrame) -> pd.DataFrame:
    def remove_punctuation_regex(text):
        return re.sub(r'[^\w\s]', '', text)
    results = []
    for i in range(len(df)):
        ya = str(df.iloc[i]['prediction'])
        pred = ya.split('Answer:')[-1]
        pred = pred.lower().strip()
        pred = pred.split('\n')[0]
        pred = remove_punctuation_regex(pred)
        if pred == 'true' or pred == 'yes':
            results.append(1)
        elif pred == 'false' or pred == 'no':
            results.append(0)
        else:
            results.append(None)
    df['output'] = results

    return df


def evaluate(df: pd.DataFrame, num_instances = int, dataset = str):
    # NUM_INSTANCES = 300         # 479 for test set
    NUM_INSTANCES = num_instances
    NUM_ROUNDS = 10
    
    # y_true = pd.read_csv('../curated_mimic_with_reasoning.csv')['y_true']
    y_true = pd.read_csv(f'../data/{dataset}/test.csv')['y_true']
    # y_true = [x for x in y_true for _ in range(NUM_ROUNDS)]
    
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
    
    all_predictions = [[] for _ in range(NUM_INSTANCES)]
    for i in range(1, NUM_ROUNDS + 1):
        rdf = df.loc[df['round'] == i]
        preds = rdf["output"].astype(str).tolist()
        for j in range(NUM_INSTANCES):
            all_predictions[j].append(preds[j])

    # One-time average version
    avg_probs = []
    for preds in all_predictions:
        valid_preds = []
        for p in preds:
            if p == '0.0':
                valid_preds.append(0)
            elif p == '1.0':
                valid_preds.append(1)
        avg_probs.append(np.mean(valid_preds) if valid_preds else np.nan)

    avg_probs = np.array(avg_probs)
    valid_mask = ~np.isnan(avg_probs)
    avg_y_true = np.array(y_true)[valid_mask]

    auroc_avg = roc_auc_score(avg_y_true, avg_probs[valid_mask])
    brier_avg = brier_score_loss(avg_y_true, avg_probs[valid_mask])
    ece_avg = compute_ece(avg_y_true, avg_probs[valid_mask])

    # Bootstrap version
    BOOTSTRAP_ITERATIONS = 15
    SAMPLE_SIZE = 9 

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
        if all(p not in ['0.0', '1.0'] for p in preds)
    ]

    for idx, preds in enumerate(all_predictions):
        if idx in invalid_indices:
            bootstrap_results.append([np.nan] * BOOTSTRAP_ITERATIONS)
        else:
            # valid_preds = [p for p in preds if p in ['0', '1']]
            valid_preds = []
            for p in preds:
                if p == '0.0':
                    valid_preds.append(0)
                elif p == '1.0':
                    valid_preds.append(1)
            bootstrap_results.append(bootstrap_prob_yes(valid_preds))

    bootstrap_df = pd.DataFrame(bootstrap_results)
    bootstrap_df.insert(0, "Instance ID", list(range(NUM_INSTANCES)))
    # bootstrap_df.to_csv(f"MIMIC_extract/last_3000/{LLM_NAME}/bootstrap_prob_yes_per_instance_{TASK_NAME}.csv", index=False)

    bootstrap_means = bootstrap_df.iloc[:, 1:].mean(axis=1).values
    filtered_y_true = np.array(y_true)[~np.isnan(bootstrap_means)]
    filtered_probs = bootstrap_means[~np.isnan(bootstrap_means)]

    auroc_bs = roc_auc_score(filtered_y_true, filtered_probs)
    brier_bs = brier_score_loss(filtered_y_true, filtered_probs)
    ece_bs = compute_ece(filtered_y_true, filtered_probs)

    bs_scores = {'auroc': auroc_bs,
              'brier score': brier_bs,
              'ece': ece_bs}
    
    avg_scores = {'auroc': auroc_avg,
                'brier score': brier_avg,
                'ece': ece_avg}
    
    scores = {'average': avg_scores,
              'bootstrap': bs_scores,}

    return scores