import pandas as pd
import numpy as np
import pickle
import json

# ── 1. LOAD EVERYTHING ────────────────────────────────────────
df = pd.read_csv("archive/tickets_with_labels.csv")

with open("archive/model.pkl",     "rb") as f: model     = pickle.load(f)
with open("archive/tfidf.pkl",     "rb") as f: tfidf     = pickle.load(f)
with open("archive/threshold.pkl", "rb") as f: threshold = pickle.load(f)

print(f"Loaded {len(df)} tickets")
print(f"Using threshold: {threshold:.2f}")

# ── 2. RUN INFERENCE ON ALL TICKETS ───────────────────────────
channel_map  = {"Web Form": 0, "Chat": 1, "Email": 2}
category_map = {"General Inquiry": 0, "Account": 1,
                "Billing": 2, "Technical": 3, "Fraud": 4}
priority_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
priority_rev = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}

df["text"] = (df["Ticket_Subject"].fillna("") + " " +
              df["Ticket_Description"].fillna("") + " " +
              df["Issue_Category"].fillna(""))

X_text = tfidf.transform(df["text"]).toarray()

df["channel_num"]     = df["Ticket_Channel"].map(channel_map).fillna(0)
df["category_num"]    = df["Issue_Category"].map(category_map).fillna(0)
df["resolution_norm"] = df["Resolution_Time_Hours"] / 120.0
df["priority_num"]    = df["Priority_Level"].map(priority_map).fillna(1)
df["desc_len"]        = df["Ticket_Description"].fillna("").str.len() / 1000.0

X_struct = df[["channel_num", "category_num",
               "resolution_norm", "priority_num", "desc_len"]].values
X = np.hstack([X_text, X_struct])

probs      = model.predict_proba(X)[:, 1]
df["pred_mismatch"]  = (probs >= threshold).astype(int)
df["pred_confidence"] = probs

print(f"\nPredicted mismatches: {df['pred_mismatch'].sum()}")

# ── 3. KEYWORD DETECTOR (for evidence) ────────────────────────
CRITICAL_WORDS = ["fraud","hacked","breach","stolen","unauthorized",
                  "scam","locked out","emergency","system down","compromised"]
HIGH_WORDS     = ["crash","not working","cannot access","payment failed",
                  "broken","failure","data loss","not syncing"]
MEDIUM_WORDS   = ["slow","delay","incorrect","wrong","trouble","issue","problem"]

def find_keywords(text):
    text = text.lower()
    found = []
    for kw in CRITICAL_WORDS:
        if kw in text:
            found.append({"keyword": kw, "severity_signal": "Critical"})
    for kw in HIGH_WORDS:
        if kw in text:
            found.append({"keyword": kw, "severity_signal": "High"})
    for kw in MEDIUM_WORDS:
        if kw in text:
            found.append({"keyword": kw, "severity_signal": "Medium"})
    return found[:3]  # top 3 keywords only

# ── 4. RESOLUTION TIME INTERPRETER ───────────────────────────
def interpret_resolution(hours):
    if hours <= 11:
        return "Low — resolved quickly, suggesting minor issue"
    elif hours <= 27:
        return "Medium — moderate resolution time"
    elif hours <= 58:
        return "High — extended resolution time suggests complexity"
    else:
        return "Critical — very long resolution time indicates serious issue"

# ── 5. GENERATE DOSSIER FOR ONE TICKET ───────────────────────
def generate_dossier(row):
    ticket_id        = row["Ticket_ID"]
    assigned         = row["Priority_Level"]
    inferred         = row["inferred_severity"]
    assigned_num     = priority_map.get(assigned, 1)
    inferred_num     = priority_map.get(inferred, 1)
    delta            = inferred_num - assigned_num
    confidence       = round(float(row["pred_confidence"]), 3)

   # Mismatch type
    if delta > 0:
        mismatch_type = "Hidden Crisis"   # under-prioritized
    elif delta < 0:
        mismatch_type = "False Alarm"     # over-prioritized
    else:
        mismatch_type = "Consistent"      # shouldn't appear in dossiers

    # Feature evidence — all traceable to actual ticket fields
    evidence = []

    # Evidence 1: Keywords from description
    text = str(row["Ticket_Description"]) + " " + str(row["Ticket_Subject"])
    keywords = find_keywords(text)
    if keywords:
        for kw in keywords:
            evidence.append({
                "signal":        "keyword",
                "source_field":  "Ticket_Description / Ticket_Subject",
                "value":         kw["keyword"],
                "weight":        kw["severity_signal"]
            })
    else:
        evidence.append({
            "signal":       "keyword",
            "source_field": "Ticket_Description / Ticket_Subject",
            "value":        "No urgent keywords detected",
            "weight":       "Low"
        })

    # Evidence 2: Resolution time
    evidence.append({
        "signal":          "resolution_time",
        "source_field":    "Resolution_Time_Hours",
        "value":           f"{row['Resolution_Time_Hours']} hours",
        "interpretation":  interpret_resolution(row["Resolution_Time_Hours"])
    })

    # Evidence 3: Issue category
    evidence.append({
        "signal":       "issue_category",
        "source_field": "Issue_Category",
        "value":        row["Issue_Category"],
        "weight":       "High risk" if row["Issue_Category"] == "Fraud" else "Standard"
    })

    # Constraint analysis (grounded, no hallucination)
    analysis = (
        f"The ticket was human-assigned '{assigned}' priority via {row['Ticket_Channel']}. "
        f"However, the resolution took {row['Resolution_Time_Hours']} hours "
        f"({interpret_resolution(row['Resolution_Time_Hours']).split('—')[0].strip()} severity signal), "
        f"and the ticket falls under the '{row['Issue_Category']}' category. "
        f"Combined signals indicate '{inferred}' severity, creating a {abs(delta)}-level priority gap."
    )

    dossier = {
        "ticket_id":          ticket_id,
        "assigned_priority":  assigned,
        "inferred_severity":  inferred,
        "mismatch_type":      mismatch_type,
        "severity_delta":     delta,
        "feature_evidence":   evidence,
        "constraint_analysis": analysis,
        "confidence":         confidence
    }
    return dossier

# ── 6. GENERATE DOSSIERS FOR ALL FLAGGED TICKETS ─────────────
flagged = df[df["pred_mismatch"] == 1].copy()
print(f"\nGenerating dossiers for {len(flagged)} flagged tickets...")

dossiers = []
for _, row in flagged.iterrows():
    dossiers.append(generate_dossier(row))

# ── 7. SAVE ALL DOSSIERS ──────────────────────────────────────
with open("archive/dossiers.json", "w") as f:
    json.dump(dossiers, f, indent=2)

print(f"✅ Saved {len(dossiers)} dossiers to archive/dossiers.json")

# ── 8. SHOW 2 EXAMPLES ───────────────────────────────────────
print("\n" + "="*50)
print("EXAMPLE 1 — Hidden Crisis:")
print("="*50)
hidden = [d for d in dossiers if d["mismatch_type"] == "Hidden Crisis"]
if hidden:
    print(json.dumps(hidden[0], indent=2))

print("\n" + "="*50)
print("EXAMPLE 2 — False Alarm:")
print("="*50)
false_alarm = [d for d in dossiers if d["mismatch_type"] == "False Alarm"]
if false_alarm:
    print(json.dumps(false_alarm[0], indent=2))