from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
import os
import torch
import pandas as pd
from datasets import Dataset
import torch
from peft import get_peft_model, LoraConfig
from utils import MSETrainer, compute_metrics, format_prompts_general
import argparse
import json



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--cot', type=str, default=None)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--p_hat', action='store_true')
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--outputdir', type=str)
    args = parser.parse_args()
    return args


def model_setup(model_name):
    AVAILABLE_MODELS = {
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "qwen": 'qwen/qwen2.5-7B',
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    'gemma': 'google/gemma-7b-it',
    'deepseek32': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'
    }

    model_name = AVAILABLE_MODELS[model_name]

    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_compute_dtype=torch.bfloat16
                                            )
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                            quantization_config=quantization_config,
                                                            device_map='auto',
                                                            id2label={0: "No", 1: "Yes"},
                                                            label2id={"No": 0, "Yes": 1})

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = model.to(device)
    return model, tokenizer

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
        # with open('raw_probs/tqa_bs_probs.json') as f:
        with open('raw_probs/bs_probs.json') as f:
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

    lengths = [len(tokenizer(t, truncation=True, max_length=2048)["input_ids"]) for t in text]
    print(min(lengths), max(lengths), sum(lengths)/len(lengths))

    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        'labels': labels,
        'p_hat': p_hats,
    })
    return dataset


def main():
    args = parse_args()
    model_name = args.model
    cot_setting = args.cot  # bayes or og or no_p_hat
    dataset = args.dataset  # MIMIC or TQA
    p_hat =  args.p_hat
    raw = args.raw
    outputdir = args.outputdir

    print(f'Running {model_name} on {dataset}, p_hat {p_hat}, cot {cot_setting}, raw {raw}')

    model, tokenizer = model_setup(model_name)

    if not cot_setting:
        train_dataset = load_data(f'data/{dataset}/train.csv', p_hat, raw, tokenizer, dataset)
    else:
        train_dataset = load_data(f'data/{dataset}/cot_{cot_setting}.csv', p_hat, raw, tokenizer, dataset)
    # dev_dataset = load_data(f'data/{dataset}/dev.csv', p_hat, tokenizer)
    test_dataset = load_data(f'data/{dataset}/test.csv', p_hat, raw=False, tokenizer=tokenizer, dataset=dataset)

    lora_config = LoraConfig(r=8,
                             lora_alpha=16, 
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
                            #  target_modules=["q_proj", "v_proj", 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
                            )
    model = get_peft_model(model, lora_config)

    save_dir = f'{dataset}/{model_name}/{model_name}_cot_{cot_setting}_p_hat_{p_hat}_raw_{raw}'
    os.makedirs(save_dir, exist_ok=True) 
    print(save_dir)

    training_args = TrainingArguments(
        output_dir=f"{outputdir}/{save_dir}",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=5e-6,
        num_train_epochs=3,
        fp16=False,
        bf16=False,
        logging_steps=1,              # Log every step
        logging_first_step=True,
        logging_dir=f"{outputdir}/{save_dir}",
        logging_strategy='steps',
        eval_strategy='steps',
        eval_steps=30,
        save_steps=30,
        save_total_limit=2,
        label_names=["labels"]
    )

    trainer = MSETrainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    model.save_pretrained(f"{outputdir}/{save_dir}")

    result = trainer.evaluate()
    with open(f'test_results_logits/{save_dir}.json', 'w') as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()