import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from google import genai

# --- 1. PAGE SETUP & THEME ---
st.set_page_config(
    page_title="AI Model Doctor | Diagnostic ICU",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Medical Dashboard CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #0ea5e9, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    
    .med-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    .status-badge-critical {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }

    .status-badge-healthy {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid #22c55e;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR CONFIG & CONTROLS ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric/512/caduceus.png", width=70)
    st.title("Doctor's Station")
    api_key = st.text_input("Gemini API Key", type="password", help="Paste your Gemini API key to activate AI consultations.")
    
    st.divider()
    st.markdown("### 📂 Dataset Input")
    uploaded_file = st.file_uploader("Upload Custom CSV Dataset", type=["csv"])
    
    st.markdown("### 🎛️ Treatment Hyperparameters")
    smote_ratio = st.slider("SMOTE Balance Ratio", min_value=0.3, max_value=1.0, value=1.0, step=0.1, help="Target ratio of minority to majority class after resampling.")
    decision_threshold = st.slider("Decision Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05, help="Probability threshold required to trigger a Churn classification.")

# --- 3. HEADER SECTION ---
st.markdown('<div class="main-title">🩺 AI Model Doctor ICU</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Autonomous diagnostic scanner, surgical pipeline, and GenAI pathology report.</div>', unsafe_allow_html=True)

# --- 4. DATA ENGINE ---
@st.cache_data
def load_default_data():
    df = pd.read_csv('data/churn_data.csv')
    return df

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Custom dataset loaded successfully!")
else:
    df = load_default_data()

# Automatically assume the last column is the target variable (Binary classification)
target_col = df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

# Ensure numeric features only for simplicity in baseline
X = X.select_dtypes(include=['number'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Baseline Execution
base_model = RandomForestClassifier(random_state=42)
base_model.fit(X_train, y_train)
base_probs = base_model.predict_proba(X_test)[:, 1]
base_preds = (base_probs >= decision_threshold).astype(int)

base_acc = accuracy_score(y_test, base_preds)
base_rec = recall_score(y_test, base_preds, zero_division=0)
base_f1 = f1_score(y_test, base_preds, zero_division=0)

# --- 5. TABS INTERFACE ---
tab1, tab2 = st.tabs(["🔬 Patient Vitals & Diagnosis", "💉 Surgical Operating Room"])

with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        st.markdown("### 🫀 Baseline Health Check")
        st.markdown('<span class="status-badge-critical">⚠️ CRITICAL: Minority Class Ignored</span>', unsafe_allow_html=True)
        st.write("")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{base_acc:.1%}")
        m2.metric("Recall", f"{base_rec:.1%}", "Deficit", delta_color="inverse")
        m3.metric("F1 Score", f"{base_f1:.1%}")
        
        st.info(f"💡 **Clinical Observation:** Evaluating target column `'{target_col}'`. Adjust thresholds in the sidebar to observe sensitivity changes.")

    with col_right:
        st.markdown("### 📊 Target Distribution")
        class_counts = y.value_counts()
        chart_data = pd.DataFrame({
            "Category": [str(k) for k in class_counts.index],
            "Count": class_counts.values
        })
        st.bar_chart(chart_data.set_index("Category"), color="#0ea5e9")

with tab2:
    st.markdown("### 💉 SMOTE Data Resampling Unit")
    st.write("Synthesize minority class representations using your customized sidebar parameters.")
    
    if st.button("⚡ Perform Data Surgery & Generate AI Clinical Notes", use_container_width=True, type="primary"):
        with st.spinner("Executing data synthesis and retraining model..."):
            smote = SMOTE(sampling_strategy=smote_ratio, random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            
            treated_model = RandomForestClassifier(random_state=42)
            treated_model.fit(X_train_smote, y_train_smote)
            
            treated_probs = treated_model.predict_proba(X_test)[:, 1]
            treated_preds = (treated_probs >= decision_threshold).astype(int)
            
            treated_acc = accuracy_score(y_test, treated_preds)
            treated_rec = recall_score(y_test, treated_preds, zero_division=0)
            treated_f1 = f1_score(y_test, treated_preds, zero_division=0)
        
        st.markdown('<span class="status-badge-healthy">✅ POST-SURGERY: Vital Signs Restored</span>', unsafe_allow_html=True)
        st.write("")
        
        # Comparative Metrics
        r1, r2, r3 = st.columns(3)
        r1.metric("Treated Accuracy", f"{treated_acc:.1%}", f"{(treated_acc - base_acc)*100:.1f}%")
        r2.metric("Treated Recall", f"{treated_rec:.1%}", f"+{(treated_rec - base_rec)*100:.1f}%", delta_color="normal")
        r3.metric("Treated F1 Score", f"{treated_f1:.1%}", f"+{(treated_f1 - base_f1)*100:.1f}%", delta_color="normal")
        
        st.divider()
        
        # GenAI Consultation Card
        if api_key:
            st.markdown("### 🩺 GenAI Clinical Consultation Notes")
            with st.spinner("AI Doctor is reviewing surgical outcomes..."):
                client = genai.Client(api_key=api_key)
                prompt = f"""
                You are an elite AI Diagnostic Specialist reviewing a machine learning model.
                Patient underwent SMOTE data surgery with a sampling ratio of {smote_ratio} and a decision threshold of {decision_threshold}.
                
                BEFORE METRICS:
                - Recall: {base_rec:.2%}
                - Accuracy: {base_acc:.2%}
                - F1: {base_f1:.2%}
                
                AFTER SMOTE SURGERY:
                - Recall: {treated_rec:.2%}
                - Accuracy: {treated_acc:.2%}
                - F1: {treated_f1:.2%}
                
                Provide a structured, executive-level medical diagnostic report with:
                1. 📋 Diagnostic Summary
                2. 🔬 Pathology Analysis (Impact of custom sliders)
                3. 💼 Real-World Operational Prognosis (Revenue & churn impact)
                Keep the tone clinical, sharp, and highly professional.
                """
                
                try:
                    response = client.models.generate_content(
                        model='gemini-3.6-flash',
                        contents=prompt
                    )
                    st.markdown(f'<div class="med-card">{response.text}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Consultation generation failed: {e}")
        else:
            st.warning("👉 Enter your Gemini API key in the sidebar station to review AI clinical notes.")