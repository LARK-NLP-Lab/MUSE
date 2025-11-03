import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
from tqdm import tqdm
from huggingface_hub import login
from utils import compute_answer_likelihood


AVAILABLE_MODELS = {'mistral': 'mistralai/Mistral-7B-Instruct-v0.1',
                    'qwen': 'Qwen/Qwen2.5-VL-7B-Instruct',
                    'deepseek': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
                    "gemma-7b": "google/gemma-7b"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--num_loops', type=int, default=10, help='number of self loops')
    parser.add_argument('--inputdir', type=str, default='../data/input/EHRShot', help='input directory')
    args = parser.parse_args()
    return args


def model_setup(model_selection: str):
    model_name = AVAILABLE_MODELS[model_selection]

    max_new_tokens = 25 if 'deepseek' not in model_name else 100

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_8bit = AutoModelForCausalLM.from_pretrained(model_name,
                                                    quantization_config=quantization_config,
                                                    device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print('Model and Tokenizer loading complete')
    return model_8bit, tokenizer, max_new_tokens


def df2chron_str(df: pd.DataFrame):
    timestamps = df['TIME'].to_list()
    text = df['TEXT'].to_list()

    chron_str = ''

    for x in zip(timestamps, text): 
        chron_str += "\t".join(map(str, x))
        chron_str += '\n'
    return chron_str


def main():
    args = parse_args()
    num_loops = args.num_loops
    input_folder = args.inputdir
    model_selection = args.model

    output_folder = f'../data/generated/EHRShot'
    os.makedirs(output_folder, exist_ok=True) 

    model, tokenizer, max_new_tokens = model_setup(model_selection)
    
    instructions = {
    'hypertension': """
Role: You are an ICU clinician diagnosing whether a patient will be diagnosed with hypertension within a year of discharge.

Task: Review the patient's medical data and decide whether they will be diagnosed with hypertension within the next year.

Medical Data:
""",

    'hyperlipidemia': """
Role: You are an ICU clinician diagnosing whether a patient will be diagnosed with hyperlipidemia within a year of discharge.

Task: Review the patient's medical data and decide whether they will be diagnosed with hyperlipidemia within the next year.

Medical Data:
""",

    'acute_mi': """
Role: You are an ICU clinician diagnosing whether a patient will be diagnosed with an acute myocardial infarction (heart attack) within a year of discharge.

Task: Review the patient's medical data and decide whether they will be diagnosed with an acute myocardial infarction within the next year.

Medical Data:
""",

    'celiac': """
Role: You are an ICU clinician diagnosing whether a patient will be diagnosed with celiac disease within a year of discharge.

Task: Review the patient's medical data and decide whether they will be diagnosed with celiac disease within the next year.

Medical Data:
""",

    'lupus': """
Role: You are an ICU clinician diagnosing whether a patient will be diagnosed with lupus within a year of discharge.

Task: Review the patient's medical data and decide whether they will be diagnosed with lupus within the next year.

Medical Data:
""",

    'pancan': """
Role: You are an ICU clinician diagnosing whether a patient will be diagnosed with pancreatic cancer within a year of discharge.

Task: Review the patient's medical data and decide whether they will be diagnosed with pancreatic cancer within the next year.

Medical Data:
"""
}

    conditions = ['acute_mi', 'celiac', 'hyperlipidemia', 'hypertension', 'lupus', 'pancan']
    
    condition_map = {
            'acute_mi': 'acute myocardial infarction (heart attack)',
            'celiac': 'celiac disease',
            'lupus': 'lupus',
            'pancan': 'pancreatic cancer',
            'hypertension': 'hypertension',
            'hyperlipidemia': 'hyperlipidemia'
    }
    
    files = os.listdir(input_folder)

    for l in range(1, num_loops + 1):
        print(f"Beginning round {l}")
        for filename in tqdm(files):
            chronology = pd.read_csv(f'{input_folder}/{filename}')
            chronology_str = df2chron_str(chronology)
            for condition in conditions:
                
                output_format = f"""
Based on the medical data, will the patient be Positive or Negative for the condition {condition_map[condition]} ? 
Your response must be either 'Postive' or 'Negative' based on your diagnosis. Your response must not contain anything else.

Diagnosis:
"""
                
                prompt = instructions[condition] + chronology_str + output_format

                #max_new_tokens is 100 for deepseek model only
                if max_new_tokens == 100:
                    prompt = prompt + "<think>\n\n</think>\n\n"

                inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

                temp = {
                    "patient_file": [filename],
                    "condition": [condition],
                    "run": [l]
                }
            
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)

                text = tokenizer.batch_decode(outputs)[0]

                text = text[len(prompt):]

                temp['prediction'] = [text]

                # print(f"Done with round {l} for condition {condition} and patient_file {filename}")

                df = pd.DataFrame(temp)
                file_exists = os.path.exists(f'{output_folder}/{model_selection}_output.csv')
                df.to_csv(f'{output_folder}/{model_selection}_output.csv', mode='a', header=not file_exists, index=False)

                #Calculate sequence likelihood
                if l == 1:
                    print(f"Computing likelihood")
                    prob_rec = compute_answer_likelihood(model=model, tokenizer=tokenizer, input_text=prompt)

                    temp['prob_pos'] = [prob_rec['prob_pos']]
                    temp['prob_neg'] = [prob_rec['prob_neg']]

                    df_likelihood = pd.DataFrame(temp)

                    file_exists = os.path.exists(f'{output_folder}/{model_selection}_likelihood.csv')
                    df_likelihood.to_csv(f'{output_folder}/{model_selection}_likelihood.csv', mode='a', header=not file_exists, index=False)


if __name__ == '__main__':
    main()
