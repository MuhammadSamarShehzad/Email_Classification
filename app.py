import streamlit as st
import joblib
import numpy as np

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="wide")

# -------------------------------------------------
# THEME (simple light/dark toggle from sidebar)
# -------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

theme = st.sidebar.radio("Theme", ["Dark", "Light"], index=0 if st.session_state.theme == "Dark" else 1)
st.session_state.theme = theme

DARK = theme == "Dark"

# -------------------------------------------------
# Global Styles
# -------------------------------------------------
dark_css = """
:root{
  --bg1:#0f172a;
  --bg2:#1e293b;
  --fg:#e2e8f0;
  --muted:#94a3b8;
  --card:rgba(255,255,255,0.06);
  --border:rgba(255,255,255,0.08);
  --accent:#6366f1;
  --accent-2:#3b82f6;
  --danger:#ef4444;
  --safe:#22c55e;
}
"""

light_css = """
:root{
  --bg1:#f8fafc;
  --bg2:#e2e8f0;
  --fg:#0f172a;
  --muted:#64748b;
  --card:rgba(255,255,255,0.7);
  --border:rgba(0,0,0,0.06);
  --accent:#6366f1;
  --accent-2:#3b82f6;
  --danger:#ef4444;
  --safe:#16a34a;
}
"""

st.markdown(f"""
<style>
{dark_css if DARK else light_css}

html, body, [data-testid="stAppViewContainer"]{{
  background: linear-gradient(135deg, var(--bg1) 0%, var(--bg2) 100%) !important;
  color: var(--fg);
  font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}}

.block-container{{ padding-top:2rem; padding-bottom:2rem; }}

h1,h2,h3,h4,h5,h6{{ color: var(--fg); }}

.header-title{{
  font-size: clamp(2.3rem, 6vw, 4rem);
  font-weight: 900;
  text-align:center;
  letter-spacing:-0.03em;
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: hue 8s linear infinite;
}}
@keyframes hue {{
  0% {{ filter:hue-rotate(0deg); }}
  100% {{ filter:hue-rotate(360deg); }}
}}

.sub{{
  text-align:center;
  color: var(--muted);
  margin-bottom:2.5rem;
  font-size:1.05rem;
}}

.card{{
  background: var(--card);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 1.75rem 1.5rem;
  box-shadow: 0 12px 32px rgba(0,0,0,{0.22 if DARK else 0.08});
}}

.stTextArea textarea{{
  background: rgba(255,255,255,{0.03 if DARK else 0.6});
  border: 1px solid var(--border);
  color: var(--fg);
  border-radius: 16px !important;
}}

.stButton>button{{
  width:100%;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  color:#fff;
  border:0;
  padding:.9rem 1.25rem;
  font-weight:600;
  border-radius:14px;
  transition: all .15s ease;
  box-shadow: 0 8px 18px rgba(99,102,241,0.25);
}}
.stButton>button:hover{{
  transform: translateY(-1px);
  box-shadow: 0 12px 26px rgba(99,102,241,0.32);
}}

.badge{{
  display:inline-block;
  padding:.4rem .85rem;
  border-radius:9999px;
  font-weight:600;
  color:white;
  margin-bottom:1rem;
  font-size:.9rem;
}}
.badge.spam{{ background: var(--danger); }}
.badge.ham{{ background: var(--safe); }}

.metric-grid > div > div{{
  background: transparent !important;
  border: 1px dashed var(--border);
  border-radius: 14px;
  padding: .75rem;
}}

footer, .viewerBadge_container__1QSob, .stDeployButton {{ display:none !important; }}
.footer{{ margin-top:3rem; text-align:center; color:var(--muted); font-size:.85rem; }}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load model + vectorizer
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("spam_detector_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown('<div class="header-title">Spam Email Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Elegant, minimal, and fast. Paste text → get a decision with probabilities.</div>', unsafe_allow_html=True)

# -------------------------------------------------
# Input
# -------------------------------------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    email_text = st.text_area("Email content", height=260, placeholder="Paste the email body here...")
    predict = st.button("Predict 🔍")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# Inference & Output
# -------------------------------------------------
if predict:
    if not email_text.strip():
        st.error("Please enter an email message to classify.")
    else:
        email_vec = vectorizer.transform([email_text])
        pred = model.predict(email_vec)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(email_vec)[0]
        else:
            proba = np.array([1.0, 0.0]) if pred == 0 else np.array([0.0, 1.0])

        is_spam = int(pred) == 1
        spam_prob = float(proba[1]) * 100
        ham_prob = float(proba[0]) * 100

        st.markdown('<div class="card">', unsafe_allow_html=True)

        if is_spam:
            st.markdown('<span class="badge spam">🚨 Spam detected</span>', unsafe_allow_html=True)
            st.markdown("### This email is **Spam**.")
            st.progress(min(int(spam_prob), 100))
            st.metric("Spam probability", f"{spam_prob:.2f}%")
            st.caption("Be cautious: links, attachments, and urgent financial requests are common in spam.")
            st.snow()
        else:
            st.markdown('<span class="badge ham">✅ Ham (Not Spam)</span>', unsafe_allow_html=True)
            st.markdown("### This email looks **Safe**.")
            st.progress(min(int(ham_prob), 100))
            st.metric("Ham probability", f"{ham_prob:.2f}%")
            st.caption("Still, stay vigilant for unusual URLs, sender spoofing, or unexpected attachments.")
            st.balloons()

        # Dual metrics
        st.markdown("#### Breakdown")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Ham (Not Spam)", f"{ham_prob:.2f}%")
        with c2:
            st.metric("Spam", f"{spam_prob:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# Extras
# -------------------------------------------------
with st.expander("ℹ️ How it works"):
    st.write(
        "- The email is vectorized (e.g., TF‑IDF) exactly like training.\n"
        "- The classifier outputs a label and probabilities.\n"
        "- You view a sleek, glassmorphic UI showing the result and both probabilities."
    )

st.markdown('<div class="footer">© 2025 • Spam Email Detector</div>', unsafe_allow_html=True)
