# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 13:05:49 2026

@author: DELL
"""
import streamlit as st
import pickle
import numpy as np


# Page Configuration

st.set_page_config(
    page_title="Email Classifier",
    page_icon="📧",
    layout="centered"
)


# Load model & vectorizer


model_path = r"C:/Users/DELL PC/OneDrive/Documents/ML projects Interncred/Spam Email Classifier/spam_model.pkl"
vectorizer_path = r"C:/Users/DELL PC/OneDrive/Documents/ML projects Interncred/Spam Email Classifier/vectorizer.pkl"

loaded_model = pickle.load(open(model_path, "rb"))
loaded_vectorizer = pickle.load(open(vectorizer_path, "rb"))


# App Header


st.markdown(
    "<h1 style='text-align: center;'>📧Email Classifier</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>"
    "This app uses <b>Machine Learning</b> to classify emails as "
    "<b>Spam</b> or <b>Ham</b> with confidence scores."
    "</p>",
    unsafe_allow_html=True
)

st.divider()


# Input Section

st.subheader("✉️ Enter Email Content")

user_input = st.text_area(
    "Paste the email message below:",
    height=180,
    placeholder="Example: Congratulations! You have won a free gift..."
)


# Prediction Section

if st.button("🔍 Analyze Email", use_container_width=True):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter an email message to analyze.")
    else:
        input_mail = [user_input]
        input_mail_feature = loaded_vectorizer.transform(input_mail)

        prediction = loaded_model.predict(input_mail_feature)
        probability = loaded_model.predict_proba(input_mail_feature)
        confidence = np.max(probability) * 100

        st.divider()
        st.subheader("📊 Prediction Result")

        if prediction[0] == 0:
            st.success("✅ **Ham Mail** (Not Spam)")
        else:
            st.error("🚨 **Spam Mail Detected**")

        st.markdown("### 🔐 Model Confidence")
        st.progress(int(confidence))
        st.write(f"**Confidence Score:** {confidence:.2f}%")

        st.caption(
            "Confidence represents how sure the model is about its prediction."
        )


# Footer

st.divider()
st.caption("Built with ❤️ using Python, Scikit-learn, and Streamlit")
