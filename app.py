import streamlit as st
import joblib
import numpy as np

# Load the pre-trained model and vectorizer
model = joblib.load('spam_detector_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Set the page title and layout, also set the favicon (using an emoji for simplicity)
st.set_page_config(page_title="Spam Email Detection", layout="wide", page_icon="📧")

# Add modern, elegant styling
st.markdown("""
    <style>
    /* Modern header */
    .header {
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 48px;
        color: #34495E;
        font-weight: bold;
        margin-top: 20px;
        letter-spacing: 1px;
    }
    .sub-header {
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 24px;
        color: #7F8C8D;
        margin-top: 10px;
        font-weight: 500;
    }

    /* Description text */
    .description {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 18px;
        color: #2C3E50;
        text-align: center;
        margin-top: 20px;
    }

    /* Input area */
    .input-box {
        text-align: center;
        margin-top: 30px;
    }

    /* Button style */
    .predict-button {
        background-color: #3498DB;
        color: white;
        font-size: 18px;
        padding: 12px 24px;
        border-radius: 10px;
        border: none;
        cursor: pointer;
        font-weight: bold;
        box-shadow: 0px 10px 20px rgba(52, 152, 219, 0.2);
    }

    .predict-button:hover {
        background-color: #2980B9;
        box-shadow: 0px 15px 25px rgba(52, 152, 219, 0.3);
    }

    /* Footer style */
    .footer {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 16px;
        text-align: center;
        color: #BDC3C7;
        margin-top: 50px;
        background-color: #2C3E50;
        padding: 20px 0;
        border-radius: 5px;
        box-shadow: 0px -5px 15px rgba(0, 0, 0, 0.1);
    }
    .footer a {
        color: #E67E22;
        text-decoration: none;
        font-weight: bold;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header">Spam Email Detection</div>', unsafe_allow_html=True)

# Subheader with improved color and font weight
st.markdown('<div class="sub-header">Classify emails as <strong>Spam</strong> or <strong>Ham (Not Spam)</strong></div>', unsafe_allow_html=True)

# Description section with improved text styling
st.markdown('<div class="description">Simply paste the email text below, and our model will predict whether it\'s Spam or Ham.</div>', unsafe_allow_html=True)

# Add a stylish input box for email text
email_text = st.text_area("Enter the email content:", height=200)

# Add a submit button with modern style
if st.button('Predict', key="predict_button", help="Click to classify the email"):
    if email_text:
        # Transform the input using the vectorizer
        email_vector = vectorizer.transform([email_text])

        # Predict using the loaded model
        prediction = model.predict(email_vector)
        prediction_proba = model.predict_proba(email_vector)  # Get probability values

        if prediction[0] == 1:
            # Spam detected
            probability = prediction_proba[0][1] * 100  # Probability for Spam (1)
            st.markdown(
                f'''
                <div style="text-align:center; background-color:#FF6F61; padding: 20px; border-radius: 10px; border: 2px solid red; box-shadow: 0px 0px 10px rgba(255, 0, 0, 0.5);">
                    <h3 style="color: white; font-family: 'Segoe UI', sans-serif;">🚨 Spam Email Detected</h3>
                    <p style="font-size: 18px; color: white; font-weight: bold;"> 
                        <span style="font-size: 22px; color: white;">Probability:</span> {probability:.2f}%
                    </p>
                    <p style="font-size: 16px; color: white;">This email is likely to be spam. Please be cautious! 🚫</p>
                </div>
                ''',
                unsafe_allow_html=True
            )
        else:
            # Ham detected
            probability = prediction_proba[0][0] * 100  # Probability for Ham (0)
            st.markdown(
                f'''
                <div style="text-align:center; background-color:#2ECC71; padding: 20px; border-radius: 10px; border: 2px solid green; box-shadow: 0px 0px 10px rgba(0, 255, 0, 0.5);">
                    <h3 style="color: white; font-family: 'Segoe UI', sans-serif;">✔️ Ham Email (Not Spam)</h3>
                    <p style="font-size: 18px; color: white; font-weight: bold;"> 
                        <span style="font-size: 22px; color: white;">Probability:</span> {probability:.2f}%
                    </p>
                    <p style="font-size: 16px; color: white;">This email appears safe. You can trust it! ✅</p>
                </div>
                ''',
                unsafe_allow_html=True
            )
    else:
        st.error("Please enter an email message to classify.")

# Add footer with an elegant design and modern colors
st.markdown("""
    <div class="footer">
        <p><strong>Spam Email Detection Project</strong><br>
        Developed as part of the virtual internship at <a href="https://ezitech.org" target="_blank">Ezitech Institute</a>.<br>
        Model: <strong>Multinomial Naive Bayes</strong></p>
    </div>
""", unsafe_allow_html=True)
