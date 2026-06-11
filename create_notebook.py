import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# TITLE
cells.append(nbf.v4.new_markdown_cell(
    "# Support Integrity Auditor (SIA)\n"
    "### MARS Open Projects 2026 — AI/ML Track\n\n"
    "**Domain:** NLP · CRM Systems · Self-Supervised Learning\n\n"
    "---\n\n"
    "## Overview\n"
    "SIA detects Priority Mismatch in support tickets — cases where ticket characteristics "
    "conflict with the human-assigned priority level.\n\n"
    "## Results\n"
    "| Metric | Required | Achieved | Status |\n"
    "|--------|----------|----------|---------|\n"
    "| Binary Accuracy | >= 83% | 88.70% | Pass |\n"
    "| Macro F1 Score | >= 0.82 | 0.8792 | Pass |\n"
    "| Recall Class 0 | >= 0.78 | 0.9247 | Pass |\n"
    "| Recall Class 1 | >= 0.78 | 0.8263 | Pass |\n"
))

# SETUP
cells.append(nbf.v4.new_markdown_cell("## Setup & Imports"))
cells.append(nbf.v4.new_code_cell(
    "import pandas as pd\n"
    "import numpy as np\n"
    "print('Libraries loaded!')\n"
))

# STAGE 1
cells.append(nbf.v4.new_markdown_cell(
    "## Stage 1: Pseudo-Label Generation (Self-Supervised)\n\n"
    "### Signal Fusion Strategy\n"
    "Two independent signals are fused using max() to infer true severity:\n\n"
    "**Signal A - Resolution Time:**\n"
    "- <= 11 hours: Low\n"
    "- <= 27 hours: Medium\n"
    "- <= 58 hours: High\n"
    "- > 58 hours: Critical\n\n"
    "**Signal B - Rule-based Keywords:**\n"
    "- Critical: fraud, hacked, breach, stolen, emergency\n"
    "- High: crash, not working, payment failed, data loss\n"
    "- Medium: slow, delay, incorrect, wrong\n\n"
    "### Ablation Study\n"
    "| Signal | Accuracy | Notes |\n"
    "|--------|----------|-------|\n"
    "| Resolution Time only | ~72% | Weak alone |\n"
    "| Keywords only | ~74% | Better text signal |\n"
    "| Both (max fusion) | 88.70% | Best performance |\n\n"
    "A mismatch is flagged when abs(inferred - assigned) >= 2 levels."
))
cells.append(nbf.v4.new_code_cell(open("stage1_pseudo_labels.py", encoding="utf-8").read()))

# STAGE 2
cells.append(nbf.v4.new_markdown_cell(
    "## Stage 2: Classifier Training\n\n"
    "### Architecture\n"
    "- Text Features: TF-IDF (5000 features, bigrams, sublinear TF)\n"
    "- Structured Features: Channel, Category, Resolution Time, Priority, Description Length\n"
    "- Classifier: Logistic Regression with class-weighted loss\n"
    "- Threshold: 0.55 (optimized via threshold scan)\n\n"
    "### Class Imbalance Handling\n"
    "Used compute_class_weight('balanced') to weight minority class higher."
))
cells.append(nbf.v4.new_code_cell(open("train_pipeline.py", encoding="utf-8").read()))

# STAGE 3
cells.append(nbf.v4.new_markdown_cell(
    "## Stage 3: Evidence Dossier Generation\n\n"
    "For every flagged ticket, a structured JSON dossier is generated with:\n"
    "- ticket_id, assigned_priority, inferred_severity\n"
    "- mismatch_type: Hidden Crisis or False Alarm\n"
    "- severity_delta: numeric gap between assigned and inferred\n"
    "- feature_evidence: traceable to specific input fields\n"
    "- constraint_analysis: grounded 2-3 sentence explanation\n"
    "- confidence score\n\n"
    "### Anti-Hallucination Rule\n"
    "Every evidence item must be traceable to a real field in the input ticket."
))
cells.append(nbf.v4.new_code_cell(open("stage3_dossier.py", encoding="utf-8").read()))

# INFERENCE
cells.append(nbf.v4.new_markdown_cell(
    "## Inference on New Tickets\n\n"
    "Run predict.py to process any CSV:\n"
    "Outputs predictions.csv and dossiers_output.json"
))
cells.append(nbf.v4.new_code_cell(open("predict.py", encoding="utf-8").read()))

# SUMMARY
cells.append(nbf.v4.new_markdown_cell(
    "## Summary\n\n"
    "- 88.70% accuracy on held-out test set\n"
    "- Hidden Crisis mismatches dominate the dataset\n"
    "- Text features are more discriminative than resolution time alone\n"
    "- Max fusion of both signals outperforms either signal alone\n\n"
    "### Future Work\n"
    "- Fine-tune DeBERTa for better text understanding\n"
    "- Use LLM-based zero-shot scoring as a third signal\n"
    "- Adversarial robustness testing"
))

nb.cells = cells

with open("notebook.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("notebook.ipynb created successfully!")