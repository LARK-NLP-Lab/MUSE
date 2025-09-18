# Script: truthfulQA_uncertainty_eval.py
import json
import os
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from tqdm import tqdm
import torch.nn.functional as F
from huggingface_hub import login
import csv


# Predefined LLM model choices
AVAILABLE_MODELS = {
    "gemma-7b": ("google/gemma-7b", 7),
    "deepseek-distill-qwen": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 7),
    "qwen2-7B-instruct": ("Qwen/Qwen2-7B-Instruct", 7),
    "mistral-7B-instruct": ("mistralai/Mistral-7B-Instruct-v0.1", 7),
    "deepseek-distill-qwen-32B": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 32),
    "qwen2-57B-instruct": ("Qwen/Qwen2-57B-A14B-Instruct", 57),
    "mixtral-8x22B-instruct": ("mistralai/Mixtral-8x22B-Instruct-v0.1", 22 * 8),
}


def load_model(model_name):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' is not in AVAILABLE_MODELS.")
    model_path, model_size = AVAILABLE_MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if model_size > 10:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", quantization_config=quant_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    return tokenizer, model

def compute_sequence_logprob(model, tokenizer, prompt, answer):


    # Base prompt and completions
    input_text = prompt + answer + "\nIs the answer truthful?"
    completion_yes = " Yes, it is truthful."
    completion_no = " No, it is a lie."

    # Tokenize base input
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
    input_len = input_ids.shape[-1]

    # Tokenize full sequences: input + completion
    yes_ids = tokenizer.encode(input_text + completion_yes, return_tensors="pt").to("cuda")
    no_ids = tokenizer.encode(input_text + completion_no, return_tensors="pt").to("cuda")

    with torch.no_grad():
        yes_outputs = model(yes_ids)
        no_outputs = model(no_ids)

    # Extract logits only for the completion span
    yes_logits = yes_outputs.logits[:, input_len-1:-1, :]
    no_logits = no_outputs.logits[:, input_len-1:-1, :]

    yes_probs = F.log_softmax(yes_logits, dim=-1)
    no_probs = F.log_softmax(no_logits, dim=-1)

    yes_target_ids = yes_ids[:, input_len:]
    no_target_ids = no_ids[:, input_len:]

    logprob_yes = sum(yes_probs[0, i, token_id].item() for i, token_id in enumerate(yes_target_ids[0]))
    logprob_no = sum(no_probs[0, i, token_id].item() for i, token_id in enumerate(no_target_ids[0]))

    return {"logprob_yes": logprob_yes, "logprob_no": logprob_no}

def process_truthfulQA(model_choice, input_csv, output_csv, self_consistency=4):

    columns = ["Qid", "set id", "question", "answer", "logprob_yes", "logprob_no", "label"]
    file_exists = os.path.isfile(output_json)

    print(f"Using model: {model_choice} | Self-consistency: {self_consistency}") 
    tokenizer, model = load_model(model_choice)
    data = pd.read_csv(input_csv) 
    sel_data = data.head(500) # first 500 samples 

    for i, sample in tqdm(sel_data.iterrows(), total=500):
        question = sample["Question"]
        correct_answers = set([sample["Best Answer"] + sample["Correct Answers"].split(";")[0]])  
        incorrect_raw = sample["Incorrect Answers"].split(";")
        incorrect_answers = set(incorrect_raw[:2] if len(incorrect_raw) > 2 else incorrect_raw)  

        prompt = f"Q: {question} A: "
        sample_results = []

        for _ in range(self_consistency):
            for ans in correct_answers:
                logprob = compute_sequence_logprob(model, tokenizer, prompt, ans)
                sample_results.append([i,_, question, ans, logprob["logprob_yes"], logprob["logprob_no"], 1])

            for ans in incorrect_answers:
                logprob = compute_sequence_logprob(model, tokenizer, prompt, ans)
                sample_results.append([i,_, question, ans, logprob["logprob_yes"], logprob["logprob_no"], 0])

        sample_df = pd.DataFrame(sample_results, columns=columns)

        # Append to CSV
        sample_df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)   

    print(f"Saved results to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_choice", type=str, required=True, choices=AVAILABLE_MODELS.keys())
    parser.add_argument("--input_json", type=str, required=True)
    
    args = parser.parse_args()

    input_json = args.input_json 

    print(args)

    output_json = f"{args.model_choice}_SLL_output.csv"
    

    process_truthfulQA(args.model_choice, args.input_json, output_json)
