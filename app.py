import streamlit as st
from sidebar import sidebar_upload

from about import show_about
from dashboard import dashboard_page
from modeling import modeling_page
from analysis_model import analysis_model_page
from prediction import prediction_page
from contact import contact_page

st.set_page_config(
    page_title="Machine Learning Classification Dashboard",
    layout="wide"
)

# SIDEBAR (HANYA DATASET)
sidebar_upload()

st.markdown(
    "<h1 style='text-align:center'>Machine Learning Classification Dashboard</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;color:gray'>Proyek Akhir UAS – Analisis Klasifikasi Berbasis Dataset Upload</p>",
    unsafe_allow_html=True
)

# ======================
# MENU TAB
# ======================
tabs = st.tabs([
    "📘 About Dataset",
    "📊 Dashboards",
    "🤖 Machine Learning",
    "🧠 Analisis Model Terbaik",
    "🔮 Prediction App",
    "📞 Contact Me"
])

with tabs[0]:
    show_about()

with tabs[1]:
    dashboard_page()

with tabs[2]:
    modeling_page()

with tabs[3]:
    analysis_model_page()

with tabs[4]:
    prediction_page()

with tabs[5]:
    contact_page()
