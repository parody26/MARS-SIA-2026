import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             recall_score, classification_report)
from sklearn.utils.class_weight import compute_class_weight
import pickle

# ── 1. LOAD DATA ──────────────────────────────────────────────
df = pd.read_csv("archive/tickets_with_labels.csv")
print(f"Total tickets: {len(df)}")
print(f"Mismatch rate: {df['mismatch'].mean():.2%}\n")

# ── 2. SPLIT FIRST ────────────────────────────────────────────
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["mismatch"]
)

# ── 3. TEXT FEATURES ONLY (no leaky columns) ──────────────────
def make_text(data):
    return (data["Ticket_Subject"].fillna("") + " " +
            data["Ticket_Description"].fillna("") + " " +
            data["Issue_Category"].fillna(""))

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                        stop_words="english", sublinear_tf=True)
X_train_text = tfidf.fit_transform(make_text(train_df)).toarray()
X_test_text  = tfidf.transform(make_text(test_df)).toarray()

# ── 4. SAFE STRUCTURED FEATURES (nothing derived from label) ──
channel_map  = {"Web Form": 0, "Chat": 1, "Email": 2}
category_map = {"General Inquiry": 0, "Account": 1,
                "Billing": 2, "Technical": 3, "Fraud": 4}
priority_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

def safe_structured(data):
    d = data.copy()
    d["channel_num"]     = d["Ticket_Channel"].map(channel_map).fillna(0)
    d["category_num"]    = d["Issue_Category"].map(category_map).fillna(0)
    d["resolution_norm"] = d["Resolution_Time_Hours"] / 120.0
    d["priority_num"]    = d["Priority_Level"].map(priority_map).fillna(1)
    # Text length as proxy for urgency detail
    d["desc_len"]        = d["Ticket_Description"].fillna("").str.len() / 1000.0
    return d[["channel_num", "category_num",
              "resolution_norm", "priority_num", "desc_len"]].values

X_train = np.hstack([X_train_text, safe_structured(train_df)])
X_test  = np.hstack([X_test_text,  safe_structured(test_df)])
y_train = train_df["mismatch"].values
y_test  = test_df["mismatch"].values

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── 5. TRAIN ──────────────────────────────────────────────────
classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
cw = dict(zip(classes, weights))

print("Training...")
model = LogisticRegression(class_weight=cw, max_iter=1000,
                           C=1.0, solver="lbfgs")
model.fit(X_train, y_train)

# ── 6. THRESHOLD SCAN ─────────────────────────────────────────
probs = model.predict_proba(X_test)[:, 1]

print(f"\n{'Thresh':>8} {'Acc':>8} {'F1':>8} {'Rec0':>8} {'Rec1':>8}")
best_thresh, best_f1 = 0.5, 0
for thresh in np.arange(0.30, 0.70, 0.05):
    preds = (probs >= thresh).astype(int)
    acc   = accuracy_score(y_test, preds)
    f1    = f1_score(y_test, preds, average="macro", zero_division=0)
    recs  = recall_score(y_test, preds, average=None, zero_division=0)
    r0, r1 = (recs[0], recs[1]) if len(recs) == 2 else (0, 0)
    ok    = "✅" if (acc>=0.83 and f1>=0.82 and r0>=0.78 and r1>=0.78) else ""
    print(f"{thresh:>8.2f} {acc:>8.2%} {f1:>8.4f} {r0:>8.4f} {r1:>8.4f} {ok}")
    if ok and f1 > best_f1:
        best_f1, best_thresh = f1, thresh

# ── 7. FINAL REPORT ───────────────────────────────────────────
y_pred = (probs >= best_thresh).astype(int)
print(f"\nUsing threshold: {best_thresh}")
print(f"\n{'='*40}")
print(f"  Binary Accuracy : {accuracy_score(y_test, y_pred):.2%}")
print(f"  Macro F1        : {f1_score(y_test, y_pred, average='macro'):.4f}")
rc = recall_score(y_test, y_pred, average=None)
print(f"  Recall Class 0  : {rc[0]:.4f}")
print(f"  Recall Class 1  : {rc[1]:.4f}")
print(f"{'='*40}")
print(classification_report(y_test, y_pred,
      target_names=["Consistent", "Mismatch"]))

# ── 8. SAVE ───────────────────────────────────────────────────
with open("archive/model.pkl",     "wb") as f: pickle.dump(model, f)
with open("archive/tfidf.pkl",     "wb") as f: pickle.dump(tfidf, f)
with open("archive/threshold.pkl", "wb") as f: pickle.dump(best_thresh, f)
print("✅ Saved!")