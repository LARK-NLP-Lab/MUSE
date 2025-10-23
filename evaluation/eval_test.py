from eval_utils import parse, evaluate
import argparse
import pandas as pd
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--cot', type=str, default=None)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--p_hat', action='store_true')
    parser.add_argument('--raw', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_name = args.model
    cot_setting = args.cot  # bayes or og or no_p_hat
    dataset = args.dataset  # MIMIC or TQA
    p_hat =  args.p_hat
    raw = args.raw

    print(f'Evaluating {model_name} on {dataset} with cot {cot_setting} and p_hat {p_hat}')

    if dataset == 'MIMIC':
        num_instances = 479
    elif dataset == 'TQA':
        num_instances = 145

    setting_path = f'{dataset}/{model_name}_cot_{cot_setting}_p_hat_{p_hat}_raw_{raw}_textgen'
    df = pd.read_csv(f'eval_results/{setting_path}.csv')
    parsed_df = parse(df)
    scores = evaluate(parsed_df, num_instances, dataset)

    with open(f'eval_scores/{setting_path}_scores.json', 'w') as f:
        json.dump(scores, f)


if __name__ == '__main__':
    main()