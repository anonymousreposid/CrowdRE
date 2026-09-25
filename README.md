# Beyond Performance: Evaluating LLM Explanations and Errors in CrowdRE Requirement Classification

This repository contains the replication package for the study on Large Language Models (LLMs) for **Crowd-based Requirements Engineering (CrowdRE)** sector classification and explanation.

The study evaluates open-source LLMs across multiple classification granularities and prompting strategies, with a joint analysis of:

- Sector classification performance
- Explanation quality
- Reasoning failures reflected in misclassified requirements

The repository is anonymized for double-blind review.

## Repository Structure

```text
.
├── Codes/
│   ├── dataset_gen.py
│   ├── experiment.py
│   ├── score.py
│   └── misclassified.py
│
├── dataset_gold/
│   ├── Experiment_dataset/
│   ├── Binary_dataset/
│   ├── Tertiary_dataset/
│   ├── Quaternary_dataset/
│   └── Quinary_dataset/
│
└── README.md

```

## Code Overview

The **`Codes/`** directory contains the scripts used for dataset generation, LLM inference, evaluation, and error analysis.

1. **`dataset_gen.py`** — Generates balanced Binary, Tertiary, Quaternary, and Quinary datasets using hierarchical sampling with a fixed random seed.  
2. **`experiment.py`** — Runs the LLM inference experiments across the selected datasets and prompting strategies and saves the generated classifications and explanations.  
3. **`score.py`** — Computes classification and explanation quality metrics, including Macro-F1, CTC, BERTScore, ROUGE-L, ROUGE-1, ROUGE-2 and METEOR, for the generated results.  
4. **`misclassified.py`** — Extracts misclassified requirements from prediction files and organizes them by true-predicted label pairs for explanation-driven error analysis.
