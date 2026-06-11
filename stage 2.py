import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
import pickle

# ── 1. LOAD DATA ──────────────────────────────────────────────
df = pd.read_csv("archive/tickets_with_labels.csv")
print(f"Total tickets: {len(df)}")
print(f"Mismatch rate: {df['mismatch'].mean():.2%}")

# ── 2. PREPARE TEXT ───────────────────────────────────────────
df["text"] = (df["Ticket_Subject"].fillna("") + " [SEP] " +
              df["Ticket_Description"].fillna("") + " [SEP] " +
              df["Issue_Category"].fillna("") + " [SEP] " +
              df["Priority_Level"].fillna(""))

# ── 3. SPLIT ──────────────────────────────────────────────────
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["mismatch"]
)
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# ── 4. TOKENIZER ──────────────────────────────────────────────
MODEL_NAME = "microsoft/deberta-v3-small"
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── 5. DATASET CLASS ──────────────────────────────────────────
class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = TicketDataset(
    train_df["text"].tolist(),
    train_df["mismatch"].tolist(),
    tokenizer
)
test_dataset = TicketDataset(
    test_df["text"].tolist(),
    test_df["mismatch"].tolist(),
    tokenizer
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

# ── 6. LOAD MODEL ─────────────────────────────────────────────
print("Loading DeBERTa-v3-small model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, torch_dtype=torch.float32
)
model.to(device)

# ── 7. CLASS WEIGHTS ──────────────────────────────────────────
n_consistent = (train_df["mismatch"] == 0).sum()
n_mismatch   = (train_df["mismatch"] == 1).sum()
weight_0 = len(train_df) / (2 * n_consistent)
weight_1 = len(train_df) / (2 * n_mismatch)
class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(device)
print(f"Class weights: {weight_0:.3f}, {weight_1:.3f}")

# ── 8. TRAINING ───────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn   = torch.nn.CrossEntropyLoss(weight=class_weights)

EPOCHS = 3
print(f"\nTraining for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if i % 100 == 0:
            print(f"  Epoch {epoch+1}, Step {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} avg loss: {total_loss/len(train_loader):.4f}")

# ── 9. EVALUATION ─────────────────────────────────────────────
print("\nEvaluating...")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds   = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average="macro")
recalls  = recall_score(all_labels, all_preds, average=None)

print(f"\n{'='*40}")
print(f"  Binary Accuracy : {accuracy:.2%}")
print(f"  Macro F1        : {macro_f1:.4f}")
print(f"  Recall Class 0  : {recalls[0]:.4f}")
print(f"  Recall Class 1  : {recalls[1]:.4f}")
print(f"{'='*40}")
print(classification_report(all_labels, all_preds,
      target_names=["Consistent", "Mismatch"]))

# ── 10. SAVE MODEL ────────────────────────────────────────────
model.save_pretrained("archive/deberta_model")
tokenizer.save_pretrained("archive/deberta_model")
print("✅ DeBERTa model saved to archive/deberta_model/")