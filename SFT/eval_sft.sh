#!/usr/bin/env bash

models=(mistral qwen)
dataset='MIMIC'

for model in "${models[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1 python generate_test.py --model "$model" --dataset "$dataset" --p_hat --raw --ckpt 270
    CUDA_VISIBLE_DEVICES=0,1 python eval_test.py --model "$model" --dataset "$dataset" --p_hat --raw
    CUDA_VISIBLE_DEVICES=0,1 python generate_test.py --model "$model" --dataset "$dataset" --raw --ckpt 270
    CUDA_VISIBLE_DEVICES=0,1 python eval_test.py --model "$model" --dataset "$dataset" --raw

    CUDA_VISIBLE_DEVICES=0,1 python generate_test.py --model "$model" --dataset "$dataset" --p_hat --cot 'og' --ckpt 270 --raw
    CUDA_VISIBLE_DEVICES=0,1 python eval_test.py --model "$model" --dataset "$dataset" --p_hat --cot 'og' --raw
    CUDA_VISIBLE_DEVICES=0,1 python generate_test.py --model "$model" --dataset "$dataset" --p_hat --cot 'bayes' --ckpt 270 --raw
    CUDA_VISIBLE_DEVICES=0,1 python eval_test.py --model "$model" --dataset "$dataset" --p_hat --cot 'bayes' --raw
    CUDA_VISIBLE_DEVICES=0,1 python generate_test.py --model "$model" --dataset "$dataset" --cot 'no_p_hat' --ckpt 270 --raw
    CUDA_VISIBLE_DEVICES=0,1 python eval_test.py --model "$model" --dataset "$dataset" --cot 'no_p_hat' --raw
done
