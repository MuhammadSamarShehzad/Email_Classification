import streamlit as st
import joblib
import numpy as np

# --------------------  Setup  --------------------
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="wide"
)

# --------------------  Styles  --------------------
st.markdown("""
<style>
:root{
  --bg: #0f172a;
  --fg: #e2e8f0;
  --card: rgba(255,255,255,0.06);
  --accent: #3b82f6;
  --accent-dark: #1d4ed8;
  --danger: #ef4444;
  --safe: #22c55e;
  --muted: #94a3b8;
  --radius: 16px;
  --font: 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, 'Roboto', 'Helvetica Neue', sans-serif;
}
html, body, [data-testid="stAppViewContainer"]{
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
  color: var(--fg);
  font-family: var(--font);
}
h1, h2, h3, h4, h5, h6 { color: var(--fg); }
.block-container{
  padding-top: 2rem;
  padding-bottom: 2rem;
}
.card{
  background: var(--card);
  backdrop-filter: blur(12px);
  border-radius: var(--radius);
  padding: 1.75rem 1.5rem;
  border: 1px solid rgba(255,255,255,0.06);
  box-shadow: 0 8px 24px rgba(0,0,0,0.15);
}
.header-title{
  font-size: 3rem;
  font-weight: 800;
  text-align: center;
  letter-spacing: -0.02em;
  margin-bottom: .25rem;
}
.sub{
  text-align:center;
  color: var(--muted);
  margin-bottom: 2rem;
}
textarea{
  border-radius: var(--radius) !important;
}
.stTextArea textarea{
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.08);
  color: var(--fg);
}
.stButton>button{
  background: var(--accent);
  color: white;
  border: 0;
  padding: .75rem 1.25rem;
  font-weight: 600;
  border-radius: .75rem;
  transition: all .15s ease;
  box-shadow: 0 8px 16px rgba(59,130,246,0.2);
}
.stButton>button:hover{
  background: var(--accent-dark);
  box-shadow: 0 12px 24px rgba(59,130,246,0.25);
}
.badge{
  display:inline-block;
  padding:.35rem .75rem;
  border-radius:9999px;
  font-weight:600;
  color:white;
  margin-bottom:.75rem;
}
.badge.spam{ background: var(--danger); }
.badge.ham{ background: var(--safe); }
footer, .viewerBadge_container__1QSob, .stDeployButton { display: none !important; }
.footer{
  margin-top:3rem; text-align:center; color:var(--muted); font-size:.9rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------  Load artifacts  --------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('spam_detector_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_artifacts()

# --------------------  UI  --------------------
st.markdown('<div class="header-title">Spam Email Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Paste email content below to classify it as <strong>Spam</strong> or <strong>Ham</strong>.</div>', unsafe_allow_html=True)

with st.container():
    with st.spinner(""):
        email_text = st.text_area("Email content", height=220, placeholder="Paste the email body here...")

col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict = st.button("Predict 🔍", use_container_width=True)

# --------------------  Inference  --------------------
if predict:
    if not email_text.strip():
        st.error("Please enter an email message to classify.")
    else:
        email_vector = vectorizer.transform([email_text])
        pred = model.predict(email_vector)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(email_vector)[0]
        else:
            # Fallback if the model doesn't expose probabilities
            proba = np.array([1.0, 0.0]) if pred == 0 else np.array([0.0, 1.0])

        is_spam = int(pred) == 1
        spam_prob = float(proba[1]) * 100
        ham_prob = float(proba[0]) * 100

        # Result Card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if is_spam:
            st.markdown('<span class="badge spam">🚨 Spam detected</span>', unsafe_allow_html=True)
            st.markdown("### This email is likely **Spam**.")
            st.progress(min(int(spam_prob), 100))
            st.metric("Spam probability", f"{spam_prob:.2f}%")
            st.caption("Be cautious with links, attachments, and requests for sensitive information.")
            st.snow()
        else:
            st.markdown('<span class="badge ham">✅ Ham (Not Spam)</span>', unsafe_allow_html=True)
            st.markdown("### This email appears **Safe**.")
            st.progress(min(int(ham_prob), 100))
            st.metric("Ham probability", f"{ham_prob:.2f}%")
            st.caption("Always stay alert for phishing patterns even in legitimate-looking emails.")
            st.balloons()

        # Show both probabilities in columns
        p1, p2 = st.columns(2)
        with p1:
            st.metric("Ham (Not Spam)", f"{ham_prob:.2f}%")
        with p2:
            st.metric("Spam", f"{spam_prob:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)

# --------------------  Info  --------------------
with st.expander("ℹ️ How it works"):
    st.write("""
    - Your text is vectorized using the same technique used during training (e.g., TF‑IDF).
    - The trained model (e.g., Multinomial Naive Bayes) outputs a class and probabilities.
    - Results are shown with a clean, minimal, glassmorphic UI.
    """)

st.markdown('<div class="footer">© 2025 • Spam Email Detector</div>', unsafe_allow_html=True)
