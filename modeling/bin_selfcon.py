import os
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from huggingface_hub import login
from bin_answer_likelihood import *
import json
import random


# Predefined LLM model choices
AVAILABLE_MODELS = {
    "deepseek-distill-qwen": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 7),
    "qwen2-7B-instruct": ("Qwen/Qwen2-7B-Instruct", 7),
    "mistral-7B-instruct": ("mistralai/Mistral-7B-Instruct-v0.1", 7),
    "deepseek-distill-qwen-32B": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 32),
    "qwen2-57B-instruct": ("Qwen/Qwen2-57B-A14B-Instruct", 57),
    "mixtral-8x22B-instruct": ("mistralai/Mixtral-8x22B-Instruct-v0.1", 22 * 8),  # Mixtral MoE with 8 experts
}

# Function to load model and tokenizer
def load_model(model_name):
    print(f"Loading model: {model_name} ...")
    if model_name not in AVAILABLE_MODELS:
        print(f"Error: Model '{model_name}' not found in AVAILABLE_MODELS.")
        print(f"Available models: {list(AVAILABLE_MODELS.keys())}")
        raise ValueError(f"Model '{model_name}' is not in AVAILABLE_MODELS. Check the model name spelling.")

    model_path, model_size = AVAILABLE_MODELS[model_name]
    
    print(f"Loading model: {model_name} ({model_size}B parameters) ...")
    # cache_dir = "/data/gaoyan"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    #model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",cache_dir=cache_dir)
    if model_size > 10:
        print(f"Applying 8-bit quantization for model: {model_name} ...")
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", quantization_config=quant_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    return tokenizer, model

# Function to generate diagnoses
def generate_diagnoses(model, tokenizer, input_text, model_choice):
    llm_max_new_tokens = 100 
    prompt = "You are a medical doctor. Your job is to predict the patient situation. Answer only with yes or no."

    if model_choice == "deepseek":
        # input_prompt = input_text + "\n" + prompt
        input_prompt = prompt + "\n" + input_text
        input_ids = tokenizer.encode(input_prompt, return_tensors="pt").to("cuda")
        llm_max_new_tokens = 1000 
    else:
        # Use system prompt for chat-based LLMs
        messages = [
            {"role": "system", "content": "You are a medical doctor. Your job is to predict the patient situation."},
            # {"role": "user", "content": input_text + "\n" + prompt}
            {"role": "user", "content": prompt + "\n" + input_text}
        ]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

    # Generate output
    # output_ids = model.generate(input_ids, max_new_tokens=llm_max_new_tokens,pad_token_id=tokenizer.pad_token_id, do_sample=True, temperature=0.7)
    output_ids = model.generate(input_ids, max_new_tokens=llm_max_new_tokens,pad_token_id=tokenizer.pad_token_id, do_sample=True, temperature=0.5)
    new_tokens = output_ids[:, input_ids.shape[-1]:]  # Extract only new tokens
    generated_text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)

    return generated_text.strip()


def insert_randomness(text: str):
    insert_chars = [',', '.', '#', ' ', '  ', '!', '?', '_']
    rand_char = random.randint(0, len(insert_chars)-1)
    
    text_list = text.split(' ')
    random_idx = random.randint(0, len(text_list)-1)
    text_list.insert(random_idx, insert_chars[rand_char])

    new_text = ' '.join(text_list)
    return new_text

# Main function to process dataset
def main(model_choice, input_csv, task, self_con, random_ins, output_dir):
    if not task:
        raise ValueError('Please provide task')
    
    # task_names = {'los3': 'los3_fix', 'los7': 'los7_fix', 'mort_hosp': 'mort_hosp', 'mort_icu': 'mort_icu'}
    task_names = {'los3': 'los3_fix', 'los7': 'los7_fix', 'mort_hosp': 'mort_hosp'}
    
    print(f"Using LLM: {model_choice}")

    # Load model and tokenizer
    tokenizer, model = load_model(model_choice)

    # Load dataset (Limit to 50 rows for efficiency)
    print(f"Loading task: {task} (First 50 rows only)")
    df = pd.read_json(f'{input_csv}/test_data_mimic_{task_names[task]}.json').head(1000)    # LIMIT TO FIRST 50 ROWS
    # df = pd.read_json(f'{input_csv}/test_data_mimic_{task_names[task]}.json')               # Full dataset


    # add loop in here
    if self_con == True:
        n = 10
    else:
        n = 1
    no_add_idx = random.randint(0, n-1)

    os.makedirs(f'{output_dir}/random_{random_ins}/{model_choice}', exist_ok=True)
    os.makedirs(f'{output_dir}/logits/random_{random_ins}/{model_choice}', exist_ok=True)

    for i in range(n):
        rdf = df
        results = []

        if random_ins == True:
            if i != no_add_idx:
                rdf['input'] = rdf['input'].apply(insert_randomness)

        for idx, sample in rdf.iterrows():
            # get likelihood
            input_text = sample['input']
            response = compute_answer_likelihood(model, tokenizer, input_text)
            # Store results
            results.append({
                "id": sample.get("id", idx),
                "input": input_text,
                "prob_yes": response["prob_yes"],
                "prob_no": response["prob_no"],
                "gold": sample['output']
            })

            # if idx % 10 == 0:
            #     print(f"Processed {idx+1}/{len(rdf)} samples...")
        # save logits
        output_path = f'{output_dir}/logits/random_{random_ins}/{model_choice}/{task}_results_round{i+1}.csv'
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

        # generate diagnoses
        print(f"Round {i+1}: Generating prediction...")
        rdf[f'prediction'] = rdf['input'].apply(lambda text: generate_diagnoses(model, tokenizer, text, model_choice))
        
        # save input to csv with round number
        rdf.to_csv(f'{output_dir}/random_{random_ins}/{model_choice}/{task}_results_round{i+1}.csv', index=False)
        print(f'Round {i+1} done.')

# Argument parser setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary Prediction")
    parser.add_argument(
        "--model_choice", type=str, required=True, choices=AVAILABLE_MODELS.keys(),
        help="Choose an LLM from: deepseek, qwen2-7B-instruct, mistral-7B-instruct"
    )
    parser.add_argument("--input_csv", type=str, default="/data/maya/Uncertainty/mimic_extract", help="Path to the mimic extract folder")
    parser.add_argument('--task', type=str, help='Binary prediction task, choose from los3, los7 mort_hosp, mort_icu')
    parser.add_argument('--self_con', action='store_true')
    parser.add_argument('--random_ins', action='store_true')
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()
    main(args.model_choice, args.input_csv, args.task, args.self_con, args.random_ins, args.output_dir)
