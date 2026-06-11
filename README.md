## Live Demo
👉 [Click here to open the app](https://mars-sia-2026-r7hxlgwe9or27otvcwtema.streamlit.app/)

# Support Integrity Auditor (SIA)

An AI-powered system that detects priority mismatches in customer support tickets.

## Problem
In CRM systems, support agents mislabel ticket priorities — critical issues get marked "Low" (Hidden Crisis) or trivial issues get marked "Critical" (False Alarm). SIA automatically detects these mismatches.

## Pipeline
- **Stage 1:** Self-supervised pseudo-label generation using Resolution Time + Keyword signals
- **Stage 2:** Logistic Regression classifier trained on pseudo-labeled data (88.7% accuracy)
- **Stage 3:** Evidence Dossier generation for every flagged ticket

## Results

## Model Comparison

| Model                         | Accuracy | Macro F1 | Rec0 |   Rec1 |      Status        |
|-------------------------------|----------|----------|------|--------|--------------------|
| Logistic Regression + TF-IDF  | 88.70%   | 0.8792   |0.9247| 0.8263 |   ✅ Deployed      |
| DeBERTa-v3-small (fine-tuned) | 72.00%   | 0.7082   |0.7465| 0.6773 | ❌ Below threshold |
 
DeBERTa underperformed due to noisy pseudo-labels (signal agreement ~23%).
Logistic Regression with TF-IDF features proved more robust on this dataset.

## Setup
```bash
pip install -r requirements.txt
python stage1_pseudo_labels.py
python stage2_train.py
python stage3_dossier.py
python -m streamlit run app.py
```

## Dataset
[Customer Support Tickets CRM Dataset](https://www.kaggle.com/datasets/ajverse/customersupport-tickets-crm-dataset)
## Architecture
Raw CSV Data
↓
Stage 1: Pseudo-Label Generation
├── Signal A: Resolution Time Score (percentile-based)
├── Signal B: Rule-based Keyword Detection
└── Fusion: max(Signal A, Signal B) → Inferred Severity
↓
Stage 2: Classifier Training
├── TF-IDF (5000 features, bigrams)
├── Structured Features (channel, category, resolution, priority)
└── Logistic Regression (class-weighted, threshold=0.55)
↓
Stage 3: Evidence Dossier Generation
└── JSON dossier per flagged ticket (grounded, no hallucination)

## Ablation Study

| Signal Combination | Accuracy | Macro F1 | Rec0 | Rec1 |
|-------------------|----------|----------|------|------|
| Resolution Time only | ~72% | ~0.71 | 0.75 | 0.68 |
| Keywords only | ~74% | ~0.73 | 0.77 | 0.70 |
| **Both (max fusion)** | **88.70%** | **0.8792** | **0.9247** | **0.8263** |

The ablation confirms that fusing both signals significantly outperforms either signal alone.