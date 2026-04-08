import streamlit as st
import pickle
import numpy as np

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Drug Prediction App",
    page_icon="💊",
    layout="centered"
)

# -------------------- Custom CSS --------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .title {
        text-align: center;
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- Load Model --------------------
model = pickle.load(open("dtmodel.pkl", "rb"))
le_sex, le_BP, le_chol, le_drug = pickle.load(open("encoders.pkl", "rb"))

# -------------------- Header --------------------
st.markdown("<h1 class='title'>💊 Drug Prediction System</h1>", unsafe_allow_html=True)
st.markdown("### 🧑‍⚕️ Enter Patient Details")

# -------------------- Layout (Columns) --------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("🎂 Age", 15, 75)
    sex = st.selectbox("👤 Sex", ["F", "M"])

with col2:
    bp = st.selectbox("🩸 Blood Pressure", ["LOW", "NORMAL", "HIGH"])
    chol = st.selectbox("🧪 Cholesterol", ["NORMAL", "HIGH"])

na_k = st.slider("⚗️ Sodium to Potassium Ratio", 5.0, 40.0)

st.markdown("---")

# -------------------- Encode Inputs --------------------
sex_enc = le_sex.transform([sex])[0]
bp_enc = le_BP.transform([bp])[0]
chol_enc = le_chol.transform([chol])[0]

# -------------------- Prediction --------------------
if st.button("🔍 Predict Drug"):
    input_data = np.array([[age, sex_enc, bp_enc, chol_enc, na_k]])
    prediction = model.predict(input_data)

    drug_name = le_drug.inverse_transform(prediction)[0]

    st.success(f"💊 Recommended Drug: **{drug_name}**")

    st.balloons()