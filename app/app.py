import streamlit as st
import joblib
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Grievance Classifier",
    page_icon="🏛️",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
    <style>
    .stButton>button {
        background-color: #2980b9;
        color: white;
        font-size: 18px;
        height: 3em;
        width: 100%;
        border-radius: 10px;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cat_model   = joblib.load(os.path.join(BASE_DIR, 'model', 'category_model.pkl'))
pri_model   = joblib.load(os.path.join(BASE_DIR, 'model', 'priority_model.pkl'))
tfidf       = joblib.load(os.path.join(BASE_DIR, 'model', 'tfidf_vectorizer.pkl'))
le_category = joblib.load(os.path.join(BASE_DIR, 'model', 'label_category.pkl'))
le_priority = joblib.load(os.path.join(BASE_DIR, 'model', 'label_priority.pkl'))

# ── Helper Functions ──────────────────────────────────────
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

category_icons = {
    "Roads":       "🛣️",
    "Water":       "💧",
    "Electricity": "⚡",
    "Garbage":     "🗑️",
    "Drainage":    "🌊"
}

priority_colors = {
    "High":   "#e74c3c",
    "Medium": "#f39c12",
    "Low":    "#2ecc71"
}

priority_icons = {
    "High":   "🔴",
    "Medium": "🟡",
    "Low":    "🟢"
}

priority_msg = {
    "High":   "Action required within 24 hours",
    "Medium": "Action required within 1 week",
    "Low":    "Action required within 1 month"
}

department_map = {
    "Roads":       "Public Works Department (PWD)",
    "Water":       "Water Supply & Sanitation Board",
    "Electricity": "Tamil Nadu Electricity Board (TNEB)",
    "Garbage":     "Solid Waste Management Department",
    "Drainage":    "Storm Water Drain Department"
}

# ── Header ────────────────────────────────────────────────
st.title("🏛️ Smart City Grievance Priority Classifier")
st.markdown("#### AI-powered system to classify and prioritize public complaints")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.header("ℹ️ About")
st.sidebar.info(
    "This system uses NLP & Machine Learning to "
    "automatically classify citizen complaints by "
    "department and assign priority levels."
)
st.sidebar.markdown("**Model:** Random Forest + TF-IDF")
st.sidebar.markdown("**Accuracy:** 100%")
st.sidebar.markdown("**Categories:** Roads, Water, Electricity, Garbage, Drainage")
st.sidebar.markdown("---")
st.sidebar.markdown("**Priority Levels:**")
st.sidebar.markdown("🔴 High — Urgent, within 24 hrs")
st.sidebar.markdown("🟡 Medium — Within 1 week")
st.sidebar.markdown("🟢 Low — Within 1 month")

# ── Input ─────────────────────────────────────────────────
st.subheader("📝 Enter Citizen Complaint")

col1, col2 = st.columns([3, 1])
with col1:
    complaint = st.text_area(
        "Type the complaint below:",
        placeholder="e.g. Large pothole on main road causing accidents near school...",
        height=150
    )
with col2:
    st.markdown("### 💡 Example Complaints")
    st.markdown("- Pothole on main road")
    st.markdown("- No water supply for 3 days")
    st.markdown("- Street light not working")
    st.markdown("- Garbage not collected")
    st.markdown("- Drain blocked causing flood")

st.markdown("---")

# ── Predict ───────────────────────────────────────────────
if st.button("🔍 Classify Complaint"):

    if len(complaint.strip()) == 0:
        st.error("⚠️ Please enter a complaint first!")

    else:
        # Preprocess & predict
        cleaned   = clean_text(complaint)
        vectorized = tfidf.transform([cleaned])

        cat_pred = cat_model.predict(vectorized)[0]
        pri_pred = pri_model.predict(vectorized)[0]
        cat_proba = cat_model.predict_proba(vectorized)[0]
        pri_proba = pri_model.predict_proba(vectorized)[0]

        category   = le_category.inverse_transform([cat_pred])[0]
        priority   = le_priority.inverse_transform([pri_pred])[0]
        cat_conf   = round(max(cat_proba) * 100, 2)
        pri_conf   = round(max(pri_proba) * 100, 2)

        cat_icon = category_icons.get(category, "📋")
        pri_icon = priority_icons.get(priority, "🟢")
        pri_color = priority_colors.get(priority, "#2ecc71")
        department = department_map.get(category, "Municipal Corporation")

        # ── Result Cards ──────────────────────────────────
        st.markdown("## 📋 Classification Result")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown(f"""
                <div style="background:#1a1a2e; padding:20px; border-radius:10px;
                border:1px solid #2980b9; text-align:center;">
                    <p style="color:#aaaaaa; font-size:14px; margin:0;">📂 Category</p>
                    <p style="color:#2980b9; font-size:28px; font-weight:bold; margin:5px 0;">{cat_icon} {category}</p>
                    <p style="color:#aaaaaa; font-size:12px; margin:0;">Confidence: {cat_conf}%</p>
                </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
                <div style="background:#1a1a2e; padding:20px; border-radius:10px;
                border:1px solid {pri_color}; text-align:center;">
                    <p style="color:#aaaaaa; font-size:14px; margin:0;">⚠️ Priority</p>
                    <p style="color:{pri_color}; font-size:28px; font-weight:bold; margin:5px 0;">{pri_icon} {priority}</p>
                    <p style="color:#aaaaaa; font-size:12px; margin:0;">Confidence: {pri_conf}%</p>
                </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
                <div style="background:#1a1a2e; padding:20px; border-radius:10px;
                border:1px solid #8e44ad; text-align:center;">
                    <p style="color:#aaaaaa; font-size:14px; margin:0;">🏢 Department</p>
                    <p style="color:#8e44ad; font-size:16px; font-weight:bold; margin:5px 0;">{department}</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"**Action Required:** {priority_msg[priority]}")
        st.markdown("---")

        # ── Category Probabilities ────────────────────────
        st.markdown("### 📊 Category Confidence Scores")
        cat_classes = le_category.classes_
        for i, cls in enumerate(cat_classes):
            prob = round(cat_proba[i] * 100, 2)
            icon = category_icons.get(cls, "📋")
            st.progress(int(prob), text=f"{icon} {cls} → {prob}%")

        st.markdown("---")

        # ── Complaint Summary ─────────────────────────────
        st.markdown("### 📝 Complaint Summary")
        st.markdown(f"""
            <div style="background:#1a1a2e; padding:20px; border-radius:10px;
            border-left: 4px solid {pri_color};">
                <p style="color:#ffffff; margin:0;"><b>Original Complaint:</b> {complaint}</p>
                <p style="color:#aaaaaa; margin:5px 0;"><b>Category:</b> {cat_icon} {category}</p>
                <p style="color:{pri_color}; margin:5px 0;"><b>Priority:</b> {pri_icon} {priority}</p>
                <p style="color:#8e44ad; margin:5px 0;"><b>Assigned To:</b> {department}</p>
                <p style="color:#aaaaaa; margin:5px 0;"><b>Action By:</b> {priority_msg[priority]}</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption("🏛️ Smart City Grievance System — Powered by AI & NLP")