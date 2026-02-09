import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="💼",
    layout="centered"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.main {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 20px;
}

h1, h2, h3, label {
    color: white !important;
}

.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    height: 3em;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    transform: scale(1.02);
}

.result-box {
    background: rgba(0, 255, 136, 0.15);
    padding: 20px;
    border-radius: 15px;
    font-size: 22px;
    text-align: center;
    color: #00ff99;
}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.markdown("<h1 style='text-align:center;'>💼 AI Salary Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#d1d1d1;'>Predict employee salary using Machine Learning</p>", unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    return pd.read_csv("salary_prediction_data.csv")

df = load_data()

# -------------------- PREVIEW --------------------
with st.expander("📊 View Dataset"):
    st.dataframe(df.head())

# -------------------- PREPROCESSING --------------------
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("Salary", axis=1)
y = df["Salary"]

# -------------------- TRAIN MODEL --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

accuracy = r2_score(y_test, model.predict(X_test))

# -------------------- MODEL INFO --------------------
st.markdown(f"""
<div style="background:rgba(255,255,255,0.12);
padding:15px;border-radius:15px;color:white;text-align:center;">
📈 Model Accuracy (R² Score): <b>{accuracy:.2f}</b>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------- USER INPUT --------------------
st.markdown("<h3>🧑‍💼 Enter Employee Details</h3>", unsafe_allow_html=True)

input_data = {}

for col in X.columns:
    if col in label_encoders:
        selected = st.selectbox(col, label_encoders[col].classes_)
        input_data[col] = label_encoders[col].transform([selected])[0]
    else:
        input_data[col] = st.slider(col, 0.0, 50.0, 1.0)

# -------------------- PREDICTION --------------------
if st.button("🔮 Predict Salary"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    st.markdown(f"""
    <div class="result-box">
        💰 Predicted Salary <br><br>
        ₹ {prediction:,.2f}
    </div>
    """, unsafe_allow_html=True)
