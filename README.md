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
| Metric | Required | Achieved |
|--------|----------|----------|
| Accuracy | ≥ 83% | 88.70% |
| Macro F1 | ≥ 0.82 | 0.8792 |
| Recall Class 0 | ≥ 0.78 | 0.9247 |
| Recall Class 1 | ≥ 0.78 | 0.8263 |

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
