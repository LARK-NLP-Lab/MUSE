# MUSE - Multi-LLM Uncertainty via Subset Ensembles

## Overview
This repository contains code for the paper "Simple Yet Effective: An Information-Theoretic Approach to Multi-LLM Uncertainty Quantification".

Link: https://arxiv.org/abs/2507.07236

## Data
We used three binary datasets, two medical- and one general-domain:

1. MIMIC-Extract:
2. EHRShot: https://stanford.redivis.com/datasets/53gc-8rhx41kgt
3. TruthfulQA: https://github.com/sylinrl/TruthfulQA

On MIMIC-Extract, we focus on the tasks of length-of-stay prediction (3 and 7 days) and in-hospital mortality prediction. On EHRShot, we evaluate six binary diagnosis prediction tasks (predicting whether the patient will develop the diagnosis within one year post-discharge). TruthfulQA is used to evaluate our approach on general domain QA.

## How to Run
1. Run generate.sh with the desired dataset (MIMIC, EHRShot, TQA) and model (mistral, qwen, deepseek, gemma) to generate the single-LLM outputs used as input for the MUSE algorithm
2. Run evaluate.sh with the desired dataset and model to evaluate single-LLM performance - this script also creates files with the bootstrapped probabilities / aggregated likelihood scores which are used by MUSE
3. To execute the MUSE algorithm, run MUSE.py with the arguments dataset, mode (gen or probs) and task (only applicable for MIMIC: los3, los7, mort_hosp)
