import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import time

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Student Marks Prediction",
    page_icon="🎓",
    layout="wide"
)

# -------------------------------
# Header
# -------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🎓 Student Marks Prediction System</h1>
    <p style="text-align:center; font-size:18px;">
    Predict <b>Math Score</b> using Machine Learning & Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("StudentsPerformance.csv")

df = load_data()

# -------------------------------
# Preprocessing
# -------------------------------
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("math score", axis=1)
y = df_encoded["math score"]

# -------------------------------
# Train Model
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# Input Section
# -------------------------------
st.subheader("📥 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["male", "female"])
    race = st.selectbox(
        "Race / Ethnicity",
        ["group A", "group B", "group C", "group D", "group E"]
    )
    parent_edu = st.selectbox(
        "Parental Level of Education",
        [
            "some high school",
            "high school",
            "some college",
            "associate's degree",
            "bachelor's degree",
            "master's degree"
        ]
    )

with col2:
    lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
    test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])
    reading_score = st.slider("📖 Reading Score", 0, 100, 70)
    writing_score = st.slider("✍️ Writing Score", 0, 100, 70)

# -------------------------------
# Create Input DataFrame (FIXED)
# -------------------------------
# Create empty row with SAME columns as training data
input_df = pd.DataFrame(0, index=[0], columns=X.columns)

# Numerical features
input_df["reading score"] = reading_score
input_df["writing score"] = writing_score

# Categorical features (only if column exists)
gender_col = f"gender_{gender}"
if gender_col in input_df.columns:
    input_df[gender_col] = 1

race_col = f"race/ethnicity_{race}"
if race_col in input_df.columns:
    input_df[race_col] = 1

parent_col = f"parental level of education_{parent_edu}"
if parent_col in input_df.columns:
    input_df[parent_col] = 1

lunch_col = f"lunch_{lunch}"
if lunch_col in input_df.columns:
    input_df[lunch_col] = 1

test_col = f"test preparation course_{test_prep}"
if test_col in input_df.columns:
    input_df[test_col] = 1

# -------------------------------
# Prediction Section
# -------------------------------
st.markdown("---")

if st.button("🎯 Predict Math Score", use_container_width=True):

    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    prediction = model.predict(input_df)[0]

    st.markdown(
        f"""
        <div style="
            background-color:#f0f2f6;
            padding:30px;
            border-radius:15px;
            text-align:center;
        ">
            <h2 style="color:#2E7D32;">📊 Predicted Math Score</h2>
            <h1 style="color:blue;">{prediction:.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    r2 = r2_score(y_test, model.predict(X_test))
    st.info(f"📈 Model Performance (R² Score): **{r2:.2f}**")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("🚀 Day 2 · Student Marks Prediction · Built with Streamlit & Machine Learning")
