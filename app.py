import streamlit as st
import joblib
import numpy as np

# -------------------- Page Config --------------------
st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="wide")

# -------------------- Styles (High Contrast, Vibrant) --------------------
st.markdown("""
<style>
/* Global background and font */
html, body, [data-testid="stAppViewContainer"]{
  background: #f3f4f6;
  color: #111827;
  font-family: 'Inter', sans-serif;
}

/* Centered header */
.header-title {
  text-align: center;
  font-size: 3rem;
  font-weight: 900;
  color: #1e40af;
  margin-bottom: 0.25rem;
}
.sub {
  text-align: center;
  color: #4b5563;
  margin-bottom: 2rem;
  font-size: 1.1rem;
}

/* Card design */
.card {
  background: white;
  border-radius: 16px;
  padding: 1.5rem 1.25rem;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
  border: 1px solid #e5e7eb;
}

/* Textarea */
.stTextArea textarea {
  background: white !important;
  color: #111827 !important;
  border-radius: 12px !important;
  border: 1px solid #cbd5e1 !important;
  font-size: 1rem !important;
  padding: 12px !important;
}

/* Predict button */
.stButton>button {
  background: linear-gradient(135deg, #2563eb, #1e40af);
  color: #fff;
  font-size: 1.1rem;
  font-weight: 600;
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 10px;
  transition: 0.2s ease;
  box-shadow: 0 6px 18px rgba(37, 99, 235, 0.3);
}
.stButton>button:hover {
  background: linear-gradient(135deg, #1e3a8a, #1e40af);
  transform: translateY(-2px);
}

/* Result badges */
.badge {
  display:inline-block;
  padding: .4rem .85rem;
  border-radius: 9999px;
  font-weight: 600;
  color: white;
  margin-bottom: 1rem;
  font-size: .95rem;
}
.badge.spam { background: #dc2626; }
.badge.ham { background: #16a34a; }

/* Footer */
footer, .viewerBadge_container__1QSob, .stDeployButton { display:none !important; }
.footer {
  margin-top: 2rem;
  text-align: center;
  color: #6b7280;
  font-size: .9rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Load Model --------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("spam_detector_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

# -------------------- Header --------------------
st.markdown('<div class="header-title">📧 Spam Email Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Paste your email content and classify it as <strong>Spam</strong> or <strong>Not Spam</strong>.</div>', unsafe_allow_html=True)

# -------------------- Input --------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    email_text = st.text_area("Email content:", height=200, placeholder="Paste the email body here...")
    predict = st.button("Predict 🔍")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Prediction --------------------
if predict:
    if not email_text.strip():
        st.error("Please enter an email message to classify.")
    else:
        email_vector = vectorizer.transform([email_text])
        pred = model.predict(email_vector)[0]
        proba = model.predict_proba(email_vector)[0] if hasattr(model, "predict_proba") else [1, 0] if pred == 0 else [0, 1]

        spam_prob = proba[1] * 100
        ham_prob = proba[0] * 100

        # Result Card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if pred == 1:
            st.markdown('<span class="badge spam">🚨 Spam Detected</span>', unsafe_allow_html=True)
            st.write(f"### This email is **Spam** (Probability: **{spam_prob:.2f}%**)")
            st.progress(int(spam_prob))
        else:
            st.markdown('<span class="badge ham">✅ Not Spam</span>', unsafe_allow_html=True)
            st.write(f"### This email is **Safe** (Probability: **{ham_prob:.2f}%**)")
            st.progress(int(ham_prob))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Ham Probability", f"{ham_prob:.2f}%")
        with col2:
            st.metric("Spam Probability", f"{spam_prob:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Footer --------------------
st.markdown('<div class="footer">© 2025 • Spam Email Detector</div>', unsafe_allow_html=True)
