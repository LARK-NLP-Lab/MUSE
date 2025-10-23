import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from datasets import Dataset
from tqdm import tqdm
from eval_utils import format_prompts_general
import pandas as pd
import os
import re
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--cot', type=str, default=None)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--p_hat', action='store_true')
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--raw', action='store_true')
    args = parser.parse_args()
    return args


def load_data(file_path, phat, raw, tokenizer, dataset='MIMIC'):
    df = pd.read_csv(file_path)


    raw_input = df['input'].to_list()
    p_hats = df['p_hat'].to_list()
    labels = df['y_true'].to_list()

    if 'reasoning' in df.columns:
        reasoning = df['reasoning'].to_list()
    else:
        reasoning = None

    if raw:
        raw_probs = []
        with open('../raw_probs/bs_probs.json') as f:
            data = json.load(f)

        for i in range(len(data)):
            raw_probs.append({
                # "mistral": data[i]['mistral'],
                "qwen": data[i]['qwen'],
                "deepseek": data[i]['deepseek']
                # "gemma": data[i]['gemma']
            })
    else:
        raw_probs = None

    text = format_prompts_general(
        dataset=dataset,
        p_hats=p_hats,
        texts=raw_input,
        cot=reasoning,
        is_phat=phat,
        raw=raw,
        raw_probs=raw_probs
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        'labels': labels,
        'p_hat': p_hats,
    })
    return dataset

def sc_generation(model, tokenizer, dataset, output_file_path, device, max_new_tokens=50):
    for i in range(1, 11):
        print(f'Start running round {i}')
        for example in tqdm(dataset):
            input_text = tokenizer.decode(example['input_ids'], skip_special_tokens=True)
            temp = {
                'input': [input_text],
                "label": [example['labels']],
                'p_hat': [example['p_hat']],
                'round': [i]
            }
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(device)

            output_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        temperature = 0.7
                    )
            decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # remove input from output whatever
            text = decoded_output[len(input_text):]
            temp['prediction'] = [text]
            
            df = pd.DataFrame(temp)
            file_exists = os.path.exists(output_file_path)
            df.to_csv(output_file_path, mode='a', header=not file_exists, index=False)


def main():
    args = parse_args()
    model_name = args.model
    cot_setting = args.cot  # bayes or og or no_p_hat
    dataset = args.dataset  # MIMIC or TQA
    p_hat =  args.p_hat
    point_no = args.ckpt
    raw = args.raw

    model_paths = {'mistral': "mistralai/Mistral-7B-v0.1",
                   'qwen': 'qwen/qwen2.5-7B',
                   "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                   'gemma': 'google/gemma-7b-it',
                   'deepseek32': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'
    }

    base_model_path = model_paths[model_name]

    test_dataset_path = f"../data/{dataset}/test.csv"
    # change checkpoint number / improve code here
    # 240 for MIMIC, 60 for TQA
    peft_model_path = f"/path/to/checkpoint"
    max_new_tokens = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==== LOAD TOKENIZER AND BASE MODEL ====
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16) #device_map='auto')
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    model.to(device)
    model.eval()

    test_dataset = load_data(test_dataset_path, p_hat, raw=False, tokenizer=tokenizer, dataset=dataset)
    output_file_path = f'eval_results/{dataset}/{model_name}_cot_{cot_setting}_p_hat_{p_hat}_raw_{raw}_textgen.csv'
    print(f'SC Generation: Running {model_name} on {dataset} with p_hat {p_hat} and cot {cot_setting} and raw {raw}')
    sc_generation(model, tokenizer, test_dataset, output_file_path, device)


if __name__ == '__main__':
    main()