import pandas as pd
import numpy as np

# ── 1. LOAD DATA ──────────────────────────────────────────────
df = pd.read_csv("archive/customer_support_tickets.csv")
print(f"Loaded {len(df)} tickets")

# ── 2. SIGNAL A: RESOLUTION TIME (percentile-based) ───────────
# Divide into 4 equal quartiles → Low/Medium/High/Critical
df["signal_resolution"] = pd.qcut(
    df["Resolution_Time_Hours"],
    q=4,
    labels=[0, 1, 2, 3]  # 0=Low, 1=Medium, 2=High, 3=Critical
).astype(int)

# ── 3. SIGNAL B: KEYWORD SCORE ────────────────────────────────
# Only use VERY specific, unambiguous phrases

CRITICAL_WORDS = [
    "fraud", "hacked", "breach", "stolen", "unauthorized",
    "scam", "locked out", "emergency", "system down", "compromised"
]
HIGH_WORDS = [
    "crash", "not working", "cannot access", "payment failed",
    "broken", "failure", "data loss", "not syncing"
]
MEDIUM_WORDS = [
    "slow", "delay", "incorrect", "wrong", "trouble", "issue", "problem"
]

# Category → forced minimum score
CATEGORY_MIN = {
    "Fraud": 3,
    "Technical": 1,
    "Billing": 1,
    "Account": 0,
    "General Inquiry": 0
}

def keyword_score(row):
    text = (str(row["Ticket_Description"]) + " " + str(row["Ticket_Subject"])).lower()
    category = str(row.get("Issue_Category", ""))
    score = CATEGORY_MIN.get(category, 0)

    for kw in CRITICAL_WORDS:
        if kw in text:
            return 3
    for kw in HIGH_WORDS:
        if kw in text:
            score = max(score, 2)
            return score
    for kw in MEDIUM_WORDS:
        if kw in text:
            score = max(score, 1)
    return score

df["signal_keyword"] = df.apply(keyword_score, axis=1)

# ── 4. SIGNAL AGREEMENT (measured BEFORE combining) ───────────
agreement = (df["signal_resolution"] == df["signal_keyword"]).mean()
print(f"Signal Agreement (A vs B): {agreement:.2%}")

# ── 5. COMBINE: use MAX of both signals ───────────────────────
# Taking the max means: if EITHER signal says it's serious, we treat it as serious
# This is intentional — we want to catch hidden crises
df["combined_score"] = df[["signal_resolution", "signal_keyword"]].max(axis=1)

label_map = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
df["inferred_severity"] = df["combined_score"].map(label_map)

# ── 6. MAP PRIORITIES TO NUMBERS ──────────────────────────────
priority_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
df["assigned_num"] = df["Priority_Level"].map(priority_map)
df["inferred_num"]  = df["inferred_severity"].map(priority_map)

# ── 7. MISMATCH LABEL (flag if gap >= 2 levels) ───────────────
df["severity_delta"] = abs(df["assigned_num"] - df["inferred_num"])
df["mismatch"] = (df["severity_delta"] >= 2).astype(int)

# ── 8. MISMATCH TYPE ──────────────────────────────────────────
def mismatch_type(row):
    if row["mismatch"] == 0:
        return "Consistent"
    elif row["inferred_num"] > row["assigned_num"]:
        return "Hidden Crisis"
    elif row["inferred_num"] < row["assigned_num"]:
        return "False Alarm"
    else:
        return "Consistent"

df["mismatch_type"] = df.apply(mismatch_type, axis=1)

# ── EXTRA: Force False Alarms for clearly over-prioritized tickets ──
# Critical/High tickets with very fast resolution + no urgent keywords
def force_false_alarm(row):
    if row["mismatch_type"] != "Consistent":
        return row["mismatch_type"]
    assigned = row["assigned_num"]
    res      = row["Resolution_Time_Hours"]
    kw       = row["signal_keyword"]
    # If assigned High/Critical but resolved very fast and no strong keywords
    if assigned >=3  and res <= 6 and kw == 0:
        return "False Alarm"
    return row["mismatch_type"]

df["mismatch_type"] = df.apply(force_false_alarm, axis=1)

# Update mismatch label to include forced false alarms
df["mismatch"] = (df["mismatch_type"] != "Consistent").astype(int)


# ── 9. PRINT RESULTS ──────────────────────────────────────────
print(f"\nMismatch distribution:")
print(df["mismatch"].value_counts())
print(f"Mismatch rate: {df['mismatch'].mean():.2%}")

print(f"\nMismatch types:")
print(df["mismatch_type"].value_counts())

print(f"\nInferred severity distribution:")
print(df["inferred_severity"].value_counts())

print(f"\nSeverity delta distribution:")
print(df["severity_delta"].value_counts().sort_index())

# ── 10. SAVE ──────────────────────────────────────────────────
df.to_csv("archive/tickets_with_labels.csv", index=False)
print("\n✅ Saved to archive/tickets_with_labels.csv")