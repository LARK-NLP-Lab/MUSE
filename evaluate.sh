#!/usr/bin/env bash

DATASET=$1
MODEL=$2

if [[ -z "$DATASET" || -z "$MODEL" ]]; then
  echo "Usage: $0 <dataset> <model>"
  echo "  dataset: TQA | MIMIC | EHRShot"
  echo "  model: model name - mistral qwen gemma deepseek"
  exit 1
fi

CUDA_VISIBLE_DEVICES=0,1 python evaluate.py --dataset "$DATASET" --model "$MODEL"
