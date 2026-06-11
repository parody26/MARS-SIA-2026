import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Support Integrity Auditor",
    page_icon="🔍",
    layout="wide"
)

# ── LOAD MODELS ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("archive/model.pkl",     "rb") as f: model     = pickle.load(f)
    with open("archive/tfidf.pkl",     "rb") as f: tfidf     = pickle.load(f)
    with open("archive/threshold.pkl", "rb") as f: threshold = pickle.load(f)
    return model, tfidf, threshold

@st.cache_data
def load_data():
    df       = pd.read_csv("archive/tickets_with_labels.csv")
    dossiers = json.load(open("archive/dossiers.json"))
    return df, dossiers

model, tfidf, threshold = load_models()
df, dossiers            = load_data()

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

def predict_ticket(subject, description, category, channel, resolution_hours, assigned_priority):
    text = subject + " " + description + " " + category
    X_text   = tfidf.transform([text]).toarray()
    X_struct = np.array([[
        channel_map.get(channel, 0),
        category_map.get(category, 0),
        resolution_hours / 120.0,
        priority_map.get(assigned_priority, 1),
        len(description) / 1000.0
    ]])
    X    = np.hstack([X_text, X_struct])
    prob = model.predict_proba(X)[0][1]
    pred = int(prob >= threshold)

    kw_score  = 3 if any(kw in (subject+description).lower() for kw in CRITICAL_WORDS) else \
                2 if any(kw in (subject+description).lower() for kw in HIGH_WORDS) else \
                1 if any(kw in (subject+description).lower() for kw in MEDIUM_WORDS) else 0
    res_score = 0 if resolution_hours<=11 else 1 if resolution_hours<=27 else 2 if resolution_hours<=58 else 3
    inferred_num = max(kw_score, res_score)
    inferred  = priority_rev[inferred_num]
    assigned_num = priority_map.get(assigned_priority, 1)
    delta = inferred_num - assigned_num

    if delta > 0:   mismatch_type = "Hidden Crisis"
    elif delta < 0: mismatch_type = "False Alarm"
    else:           mismatch_type = "Consistent"

    keywords = find_keywords(subject + " " + description)
    evidence = []
    if keywords:
        for kw in keywords:
            evidence.append({"signal": "keyword", "source_field": "Ticket_Description",
                             "value": kw["keyword"], "weight": kw["severity_signal"]})
    else:
        evidence.append({"signal": "keyword", "source_field": "Ticket_Description",
                         "value": "No urgent keywords detected", "weight": "Low"})
    evidence.append({"signal": "resolution_time", "source_field": "Resolution_Time_Hours",
                     "value": f"{resolution_hours} hours",
                     "interpretation": interpret_resolution(resolution_hours)})
    evidence.append({"signal": "issue_category", "source_field": "Issue_Category",
                     "value": category,
                     "weight": "High risk" if category == "Fraud" else "Standard"})

    analysis = (f"Ticket assigned '{assigned_priority}' via {channel}. "
                f"Resolution took {resolution_hours}h ({interpret_resolution(resolution_hours).split('—')[0].strip()} signal). "
                f"Category: '{category}'. Combined signals suggest '{inferred}' severity.")

    display_mismatch = pred or abs(delta) >= 2

    return {
        "is_mismatch":   display_mismatch,
        "confidence":    round(prob, 3),
        "inferred":      inferred,
        "mismatch_type": mismatch_type,
        "delta":         delta,
        "evidence":      evidence,
        "analysis":      analysis,
        "kw_score":      kw_score,
        "res_score":     res_score
    }

# ══════════════════════════════════════════════
# UI LAYOUT
# ══════════════════════════════════════════════
st.title("🔍 Support Integrity Auditor (SIA)")
st.markdown("*Detects priority mismatches in support tickets using AI*")
st.divider()

tab1, tab2, tab3 = st.tabs(["🎫 Single Ticket Audit", "📊 Dashboard", "📁 Batch CSV Upload"])

# ──────────────────────────────────────────────
# TAB 1: SINGLE TICKET
# ──────────────────────────────────────────────
with tab1:
    st.subheader("Audit a Single Ticket")
    col1, col2 = st.columns(2)

    with col1:
        subject     = st.text_input("Ticket Subject", "Application crashes on startup")
        description = st.text_area("Ticket Description",
                                   "Hi Support, The application crashes every time I open it. I cannot access my data.", height=120)
        category    = st.selectbox("Issue Category",
                                   ["Technical", "Billing", "Account", "Fraud", "General Inquiry"])

    with col2:
        channel           = st.selectbox("Ticket Channel", ["Web Form", "Chat", "Email"])
        resolution_hours  = st.slider("Resolution Time (Hours)", 1, 120, 45)
        assigned_priority = st.selectbox("Assigned Priority", ["Low", "Medium", "High", "Critical"])

    if st.button("🔍 Audit Ticket", type="primary"):
        result = predict_ticket(subject, description, category,
                                channel, resolution_hours, assigned_priority)

        st.divider()
        col_a, col_b, col_c = st.columns(3)

        if result["is_mismatch"]:
            col_a.metric("Verdict", "⚠️ MISMATCH")
            col_b.metric("Type", result["mismatch_type"])
        else:
            col_a.metric("Verdict", "✅ CONSISTENT")
            col_b.metric("Type", "No Mismatch")

        col_c.metric("Confidence", f"{result['confidence']*100:.1f}%")

        col_d, col_e = st.columns(2)
        col_d.metric("Assigned Priority", assigned_priority)
        col_e.metric("Inferred Severity", result["inferred"],
                     delta=f"{result['delta']:+d} levels" if result["delta"] != 0 else None)

        # Signal contribution chart
        st.subheader("📊 Signal Contributions")
        sig_df = pd.DataFrame({
            "Signal": ["Keyword Score", "Resolution Time Score"],
            "Score":  [result["kw_score"], result["res_score"]]
        })
        fig = px.bar(sig_df, x="Signal", y="Score",
                     color="Score", color_continuous_scale="Reds",
                     range_y=[0, 3], title="Contributing Signal Scores (0=Low, 3=Critical)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Evidence Dossier")
        st.json({
            "assigned_priority":   assigned_priority,
            "inferred_severity":   result["inferred"],
            "mismatch_type":       result["mismatch_type"],
            "severity_delta":      result["delta"],
            "feature_evidence":    result["evidence"],
            "constraint_analysis": result["analysis"],
            "confidence":          result["confidence"]
        })

# ──────────────────────────────────────────────
# TAB 2: DASHBOARD
# ──────────────────────────────────────────────
with tab2:
    st.subheader("📊 Priority Mismatch Dashboard")

    flagged = df[df["mismatch"] == 1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tickets",      f"{len(df):,}")
    col2.metric("Flagged Mismatches", f"{len(flagged):,}")
    col3.metric("Mismatch Rate",      f"{len(flagged)/len(df)*100:.1f}%")
    col4.metric("Model Accuracy",     "88.70%")

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        cat_counts = flagged["Issue_Category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig1 = px.bar(cat_counts, x="Category", y="Count",
                      title="Flagged Tickets by Category",
                      color="Count", color_continuous_scale="Reds")
        st.plotly_chart(fig1, use_container_width=True)

    with col_right:
        pri_counts = flagged["Priority_Level"].value_counts().reset_index()
        pri_counts.columns = ["Priority", "Count"]
        fig2 = px.pie(pri_counts, names="Priority", values="Count",
                      title="Assigned Priority of Flagged Tickets",
                      color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig2, use_container_width=True)

    # Top contributing signals
    st.subheader("🔝 Top Contributing Signals")
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        chan_counts = flagged["Ticket_Channel"].value_counts().reset_index()
        chan_counts.columns = ["Channel", "Mismatches"]
        fig3 = px.bar(chan_counts, x="Channel", y="Mismatches",
                      title="Mismatches by Ticket Channel",
                      color="Mismatches", color_continuous_scale="Blues")
        st.plotly_chart(fig3, use_container_width=True)

    with col_s2:
        mismatch_type_counts = df["mismatch_type"].value_counts().reset_index()
        mismatch_type_counts.columns = ["Type", "Count"]
        fig4 = px.pie(mismatch_type_counts, names="Type", values="Count",
                      title="Mismatch Type Distribution",
                      color_discrete_map={"Hidden Crisis": "#e74c3c",
                                          "False Alarm":   "#f39c12",
                                          "Consistent":    "#2ecc71"})
        st.plotly_chart(fig4, use_container_width=True)

    # Severity delta heatmap — category AND channel
    st.subheader("🌡️ Severity Delta Heatmap")
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        heatmap_cat = df.groupby(["Issue_Category", "Priority_Level"])["severity_delta"].mean().unstack()
        fig5 = px.imshow(heatmap_cat,
                         title="Avg Severity Delta by Category & Priority",
                         color_continuous_scale="RdYlGn_r",
                         labels={"color": "Avg Delta"})
        st.plotly_chart(fig5, use_container_width=True)

    with col_h2:
        heatmap_chan = df.groupby(["Ticket_Channel", "Priority_Level"])["severity_delta"].mean().unstack()
        fig6 = px.imshow(heatmap_chan,
                         title="Avg Severity Delta by Channel & Priority",
                         color_continuous_scale="RdYlGn_r",
                         labels={"color": "Avg Delta"})
        st.plotly_chart(fig6, use_container_width=True)

# ──────────────────────────────────────────────
# TAB 3: BATCH CSV
# ──────────────────────────────────────────────
with tab3:
    st.subheader("📁 Batch Audit via CSV Upload")
    st.markdown("Upload a CSV with columns: `Ticket_Subject`, `Ticket_Description`, `Issue_Category`, `Ticket_Channel`, `Resolution_Time_Hours`, `Priority_Level`")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(batch_df)} tickets")

        results = []
        for _, row in batch_df.iterrows():
            r = predict_ticket(
                str(row.get("Ticket_Subject", "")),
                str(row.get("Ticket_Description", "")),
                str(row.get("Issue_Category", "Technical")),
                str(row.get("Ticket_Channel", "Web Form")),
                float(row.get("Resolution_Time_Hours", 24)),
                str(row.get("Priority_Level", "Medium"))
            )
            results.append({
                "Ticket_ID":      row.get("Ticket_ID", "N/A"),
                "Assigned":       row.get("Priority_Level", "N/A"),
                "Inferred":       r["inferred"],
                "Mismatch":       "Yes" if r["is_mismatch"] else "No",
                "Type":           r["mismatch_type"],
                "Confidence":     r["confidence"]
            })

        result_df = pd.DataFrame(results)
        st.dataframe(result_df, use_container_width=True)

        # Summary metrics
        total     = len(result_df)
        mismatches = result_df[result_df["Mismatch"] == "Yes"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Total",      total)
        col2.metric("Mismatches", len(mismatches))
        col3.metric("Rate",       f"{len(mismatches)/total*100:.1f}%")

        st.download_button("⬇️ Download Results",
                           result_df.to_csv(index=False),
                           "audit_results.csv", "text/csv")