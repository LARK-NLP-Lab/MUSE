import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
import string
import argparse
import re
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='path to model results folder')
    args = parser.parse_args()
    return args


def format(prediction: str) -> int:
    translator = str.maketrans('', '', string.punctuation)
    print(prediction)
    prediction = prediction.lower()
    if '</think>' in prediction:                                        # deepseek
        prediction = prediction.split('</think>')[1]
    if 'step-by-step explanation:' in prediction:
        prediction = prediction.split('step-by-step explanation:')[0]   # deepseek 32B
    if 'assistant: ' in prediction:                                     # qwen
        prediction = prediction.split('assistant: ')[1]
    if 'user\n' in prediction:
        prediction = prediction.split('user\n')[1]
        prediction = prediction.split(' ')[-1]
    clean_text = prediction.replace('\n', '')
    clean_text = clean_text.translate(translator)
    if clean_text == 'no':
        return 0
    elif clean_text == 'yes':
        return 1
    else:
        return 2
    
def extract_yes_no_answer(text: str) -> str:
    """
    Removes literal 'yes/no' (case-insensitive) and extracts 'yes' or 'no' from the text.
    """
    if isinstance(text, float):
        return 2
    
    text = text.split('**Answer:**')[-1]
    cleaned_text = re.sub(r'\byes/no\b', '', text, flags=re.IGNORECASE).strip()

    # Look for answer after 'answer:'
    answer_match = re.search(r'answer:\s*(yes|no)', cleaned_text, re.IGNORECASE)
    if answer_match:
        return 1 if answer_match.group(1).lower() == 'yes' else 0

    # Look for standalone yes/no lines
    lines = cleaned_text.lower().splitlines()
    for line in lines:
        line = line.strip()
        if line == "yes":
            return 1
        if line == "no":
            return 0

    # Fallback to first occurrence
    fallback_match = re.search(r'\b(yes|no)\b', cleaned_text, re.IGNORECASE)
    if fallback_match:
        return 1 if fallback_match.group(1).lower() == 'yes' else 0

    return 2 # default if none found



def main():
    args = parse_args()
    model = args.model
    # model = 'gemma-7B-it'

    tasks = ['los3', 'los7', 'mort_hosp']
    parsed = f'../final_results/MIMIC_extract/full/{model}/parsed'
    # parsed = f'../bin_sc_full_fast/random_False/{model}/parsed'

    os.makedirs(parsed, exist_ok=True)

    for task in tasks:
        dfs = [pd.read_csv(f'../final_results/MIMIC_extract/full/{model}/output/{task}_results_round{x}.csv') for x in range(1, 11)]
        # dfs = [pd.read_csv(f'../bin_sc_full_fast/random_False/{model}/output/{task}_results_round{x}.csv') for x in range(1, 11)]
        gold_all, pred_all = [], []
        for i, df in enumerate(dfs):
            if model != 'mistral-7B-instruct':
                df = df.rename(columns={f"{task}": "prediction"})

            # df['prediction'] = df['prediction'].apply(format)
            # df['output'] = df['output'].apply(format)

            df['prediction'] = df['prediction'].apply(extract_yes_no_answer)
            df['output'] = df['output'].apply(extract_yes_no_answer)

            # change this - include all columns regardless of output value
            # clean_df = df[~df['prediction'].isna()]
            clean_df = df

            gold = clean_df['output'].tolist()
            pred = clean_df['prediction'].tolist()

            gold_all.extend(gold)
            pred_all.extend(pred)

            df.to_csv(f'../final_results/MIMIC_extract/full/{model}/parsed/{task}_results_round{i+1}.csv', index=False)
            # df.to_csv(f'../bin_sc_full_fast/random_False/{model}/parsed/{task}_results_round{i+1}.csv', index=False)
        print(f'Results on {model} {task}')
            
        acc = accuracy_score(gold, pred)
        try:
            auroc = roc_auc_score(gold, pred)
        except ValueError:
            auroc = 0

        print(f'{task} accuracy: {acc:.2f}')
        print(f'{task} auroc: {auroc:.2f}\n')


if __name__ == '__main__':
    main()