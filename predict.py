import pandas as pd
import numpy as np
import pickle
import json
import sys
import os

# ── LOAD MODELS ───────────────────────────────────────────────
with open("archive/model.pkl",     "rb") as f: model     = pickle.load(f)
with open("archive/tfidf.pkl",     "rb") as f: tfidf     = pickle.load(f)
with open("archive/threshold.pkl", "rb") as f: threshold = pickle.load(f)

# ── CONSTANTS ─────────────────────────────────────────────────
priority_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
priority_rev = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
channel_map  = {"Web Form": 0, "Chat": 1, "Email": 2}
category_map = {"General Inquiry": 0, "Account": 1,
                "Billing": 2, "Technical": 3, "Fraud": 4}

CRITICAL_WORDS = ["fraud","hacked","breach","stolen","unauthorized",
                  "scam","locked out","emergency","system down","compromised"]
HIGH_WORDS     = ["crash","not working","cannot access","payment failed",
                  "broken","failure","data loss","not syncing"]
MEDIUM_WORDS   = ["slow","delay","incorrect","wrong","trouble","issue","problem"]

def find_keywords(text):
    text = text.lower()
    found = []
    for kw in CRITICAL_WORDS:
        if kw in text: found.append({"keyword": kw, "severity_signal": "Critical"})
    for kw in HIGH_WORDS:
        if kw in text: found.append({"keyword": kw, "severity_signal": "High"})
    for kw in MEDIUM_WORDS:
        if kw in text: found.append({"keyword": kw, "severity_signal": "Medium"})
    return found[:3]

def interpret_resolution(hours):
    if hours <= 11:   return "Low — resolved quickly"
    elif hours <= 27: return "Medium — moderate resolution time"
    elif hours <= 58: return "High — extended resolution time"
    else:             return "Critical — very long resolution time"

def predict_row(row):
    subject     = str(row.get("Ticket_Subject", ""))
    description = str(row.get("Ticket_Description", ""))
    category    = str(row.get("Issue_Category", "Technical"))
    channel     = str(row.get("Ticket_Channel", "Web Form"))
    resolution  = float(row.get("Resolution_Time_Hours", 24))
    priority    = str(row.get("Priority_Level", "Medium"))

    # Build features
    text   = subject + " " + description + " " + category
    X_text = tfidf.transform([text]).toarray()
    X_struct = np.array([[
        channel_map.get(channel, 0),
        category_map.get(category, 0),
        resolution / 120.0,
        priority_map.get(priority, 1),
        len(description) / 1000.0
    ]])
    X    = np.hstack([X_text, X_struct])
    prob = model.predict_proba(X)[0][1]
    pred = int(prob >= threshold)

    # Infer severity
    kw_score  = 3 if any(kw in (subject+description).lower() for kw in CRITICAL_WORDS) else \
                2 if any(kw in (subject+description).lower() for kw in HIGH_WORDS) else \
                1 if any(kw in (subject+description).lower() for kw in MEDIUM_WORDS) else 0
    res_score = 0 if resolution<=11 else 1 if resolution<=27 else 2 if resolution<=58 else 3
    inferred_num = max(kw_score, res_score)
    inferred     = priority_rev[inferred_num]
    assigned_num = priority_map.get(priority, 1)
    delta        = inferred_num - assigned_num

    if delta > 0:   mismatch_type = "Hidden Crisis"
    elif delta < 0: mismatch_type = "False Alarm"
    else:           mismatch_type = "Consistent"

    # Build evidence
    keywords = find_keywords(subject + " " + description)
    evidence = []
    if keywords:
        for kw in keywords:
            evidence.append({"signal": "keyword",
                             "source_field": "Ticket_Description",
                             "value": kw["keyword"],
                             "weight": kw["severity_signal"]})
    else:
        evidence.append({"signal": "keyword",
                         "source_field": "Ticket_Description",
                         "value": "No urgent keywords detected",
                         "weight": "Low"})
    evidence.append({"signal": "resolution_time",
                     "source_field": "Resolution_Time_Hours",
                     "value": f"{resolution} hours",
                     "interpretation": interpret_resolution(resolution)})
    evidence.append({"signal": "issue_category",
                     "source_field": "Issue_Category",
                     "value": category,
                     "weight": "High risk" if category == "Fraud" else "Standard"})

    analysis = (f"Ticket assigned '{priority}' via {channel}. "
                f"Resolution took {resolution}h ({interpret_resolution(resolution).split('—')[0].strip()} signal). "
                f"Category: '{category}'. Combined signals suggest '{inferred}' severity.")

    return {
        "ticket_id":           row.get("Ticket_ID", "N/A"),
        "assigned_priority":   priority,
        "inferred_severity":   inferred,
        "mismatch":            pred,
        "mismatch_type":       mismatch_type if pred == 1 else "Consistent",
        "severity_delta":      delta,
        "confidence":          round(prob, 3),
        "constraint_analysis": analysis,
        "feature_evidence":    evidence
    }

# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # Get input CSV path from command line, default to test file
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "archive/customer_support_tickets.csv"

    if not os.path.exists(input_csv):
        print(f"❌ File not found: {input_csv}")
        sys.exit(1)

    print(f"📂 Loading: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"📊 Processing {len(df)} tickets...")

    results   = []
    dossiers  = []

    for _, row in df.iterrows():
        r = predict_row(row)
        results.append({
            "Ticket_ID":         r["ticket_id"],
            "Assigned_Priority": r["assigned_priority"],
            "Inferred_Severity": r["inferred_severity"],
            "Mismatch":          "Yes" if r["mismatch"] else "No",
            "Mismatch_Type":     r["mismatch_type"],
            "Severity_Delta":    r["severity_delta"],
            "Confidence":        r["confidence"]
        })
        if r["mismatch"] == 1:
            dossiers.append(r)

    # Save predictions CSV
    out_csv = "predictions.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"✅ Predictions saved to: {out_csv}")

    # Save dossiers JSON
    out_json = "dossiers_output.json"
    with open(out_json, "w") as f:
        json.dump(dossiers, f, indent=2)
    print(f"✅ Dossiers saved to: {out_json}")

    # Summary
    total    = len(results)
    mismatches = sum(1 for r in results if r["Mismatch"] == "Yes")
    print(f"\n📈 Summary:")
    print(f"   Total tickets    : {total}")
    print(f"   Mismatches found : {mismatches} ({mismatches/total*100:.1f}%)")
    print(f"   Hidden Crisis    : {sum(1 for r in results if r['Mismatch_Type'] == 'Hidden Crisis')}")
    print(f"   False Alarm      : {sum(1 for r in results if r['Mismatch_Type'] == 'False Alarm')}")
    