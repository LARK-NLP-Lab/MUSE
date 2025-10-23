import torch
import pandas as pd
import os
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='path to model results folder')
    args = parser.parse_args()
    return args


def normalize_probs(df: pd.DataFrame) -> pd.DataFrame:
    norm_yes, norm_no = [], []

    for i, row in df.iterrows():
        yes = row['prob_yes']
        no = row['prob_no']

        values = torch.Tensor([yes, no])
        normalized = values / values.sum()

        norm_yes.append(float(normalized[0]))
        norm_no.append(float(normalized[1]))

    df['norm_yes'] = norm_yes
    df['norm_no'] = norm_no
    return df


def main():
    args = parse_args()
    model = args.model
    # model = "gemma-7B-it"
    
    path = f'../final_results/MIMIC_extract/SLL_full/{model}/logits'
    dest = f'../final_results/MIMIC_extract/SLL_full/{model}/norm_probs'

    os.makedirs(dest, exist_ok=True)

    for filename in tqdm(os.listdir(path)):
        df = pd.read_json(f'{path}/{filename}')
        prob_df = normalize_probs(df)
        prob_df.to_csv(f'{dest}/{filename}', index=False)


if __name__ == '__main__':
    main()