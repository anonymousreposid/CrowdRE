# Beyond Performance: Evaluating LLM Explanations and Errors in CrowdRE Requirement Classification

This repository contains the replication package for the study on Large Language Models (LLMs) for Crowd-based Requirements Engineering (CrowdRE) sector classification.

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
