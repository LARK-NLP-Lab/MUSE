#!/usr/bin/env bash

DATASET=$1
MODEL=$2

if [[ -z "$DATASET" || -z "$MODEL" ]]; then
  echo "Usage: $0 <dataset> <model>"
  echo "  dataset: TQA | MIMIC | EHRShot"
  echo "  model: model name - mistral qwen gemma deepseek"
  exit 1
fi

case "$DATASET" in
  TQA)
    CUDA_VISIBLE_DEVICES=0,1 python generate_tqa.py --model-choice "$MODEL"
    CUDA_VISIBLE_DEVICES=0,1 python likelihood_tqa.py --model-choice "$MODEL"
    ;;

  MIMIC)
    CUDA_VISIBLE_DEVICES=0,1 python generate_MIMIC_extract.py --model_choice "$MODEL" --self_con --task los3
    CUDA_VISIBLE_DEVICES=0,1 python generate_MIMIC_extract.py --model_choice "$MODEL" --self_con --task los7
    CUDA_VISIBLE_DEVICES=0,1 python generate_MIMIC_extract.py --model_choice "$MODEL" --self_con --task mort_hosp
    ;;

  EHRShot)
    CUDA_VISIBLE_DEVICES=0,1 python generate_EHRShot.py --model "$MODEL"
    ;;

  *)
    echo "Error: Unknown dataset '$DATASET'."
    echo "Valid options: TQA | MIMIC | EHRShot"
    exit 1
    ;;
esac