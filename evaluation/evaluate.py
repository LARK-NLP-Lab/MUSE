import argparse
from eval_utils import normalize_probs, parse, evaluate
import pandas as pd
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    return args

def main():
    # take dataset and model args to find correct results
    # parse
    # normalize logits
    # evaluate
    args = parse_args()
    dataset = args.dataset
    model = args.model

    sll_path = f'../data/generated/{dataset}/{model}_likelihood.csv'
    gen_path = f'../data/generated/{dataset}/{model}_likelihood.csv'
    
    score_path = f'../results/{dataset}/{model}'
    os.makedirs(score_path, exist_ok=True) 

    sll_df = pd.read_csv(sll_path)
    gen_df = pd.read_csv(gen_path)

    if dataset == 'MIMIC':
        num_instances = 4790    # full dataset
    elif dataset == 'TQA':
        num_instances = 145
    elif dataset == 'EHRShot':
        pass

    normalized_sll = normalize_probs(sll_df)
    parsed_gen = parse(gen_df)

    scores = evaluate(parsed_gen, normalized_sll, num_instances)


    with open(f'{score_path}_scores.json', 'w') as f:
        json.dump(scores, f)

if __name__ == '__main__':
    main()