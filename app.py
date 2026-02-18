import os
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Health Symptom Checker",
    page_icon="🩺",
    layout="wide"
)

# ------------------ LOAD ENV ------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in .env file")
    st.stop()

# ------------------ LOAD DATASET ------------------
try:
    df = pd.read_csv("dataset.csv")
except FileNotFoundError:
    st.error("❌ dataset.csv file not found!")
    st.stop()

symptom_cols = [c for c in df.columns if c.lower().startswith("symptom")]

if "condition" not in df.columns or not symptom_cols:
    st.error("❌ Dataset must contain condition and Symptom_* columns")
    st.stop()

df["condition"] = df["condition"].str.strip()
for col in symptom_cols:
    df[col] = df[col].astype(str).str.lower().str.strip()

# ------------------ SYMPTOM MATCH ------------------
def retrieve_conditions(user_symptoms):
    user_symptoms = [s.lower().strip().replace(" ", "_") for s in user_symptoms]
    matched_conditions = set()

    for _, row in df.iterrows():
        row_symptoms = set(
            s for s in row[symptom_cols].values
            if s and s != "nan"
        )
        if row_symptoms.intersection(user_symptoms):
            matched_conditions.add(row["condition"])

    return list(matched_conditions)

# ------------------ GEMINI ------------------
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0.1,
    google_api_key=GOOGLE_API_KEY
)

system_prompt = """
You are a health assistant.
STRICT RULES:
- DO NOT diagnose diseases
- DO NOT suggest medicines
- Output ONLY valid JSON
JSON FORMAT:
{
  "possible_conditions": ["Condition1"],
  "next_steps": "General health guidance",
  "disclaimer": "This is not medical advice"
}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# ------------------ DARK / LIGHT TOGGLE ------------------
theme = st.sidebar.toggle("🌙 Dark Mode")

if theme:
    bg_color = "#0e1117"
    text_color = "white"
else:
    bg_color = "#f4f9ff"
    text_color = "#0a3d62"

# ------------------ CUSTOM CSS ------------------
st.markdown(f"""
<style>
.stApp {{
    background-color: {bg_color};
    color: {text_color};
}}

.big-title {{
    font-size: 48px;
    font-weight: 800;
    text-align: center;
}}

.subtitle {{
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}}

.result-card {{
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.15);
    margin-top: 15px;
}}

.heartbeat {{
    font-size: 60px;
    text-align: center;
    animation: beat 1s infinite;
}}

@keyframes beat {{
    0% {{ transform: scale(1); }}
    25% {{ transform: scale(1.2); }}
    40% {{ transform: scale(1); }}
    60% {{ transform: scale(1.2); }}
    100% {{ transform: scale(1); }}
}}

.stButton>button {{
    background-color: #007BFF;
    color: white;
    border-radius: 10px;
    padding: 10px 25px;
}}

</style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR PROFILE ------------------
st.sidebar.header("👤 Patient Profile")

name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", min_value=1, max_value=120)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
history = st.sidebar.text_area("Medical History")

# ------------------ MAIN TITLE ------------------
st.markdown('<div class="big-title">🩺 Smart Health Symptom Checker</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered preliminary health guidance</div>', unsafe_allow_html=True)

# ------------------ SYMPTOM CHIPS ------------------
st.subheader("Select Your Symptoms")

all_symptoms = sorted(
    set(
        s.replace("_", " ")
        for col in symptom_cols
        for s in df[col].dropna().unique()
        if s != "nan"
    )
)

selected_symptoms = st.multiselect(
    "Choose symptoms",
    options=all_symptoms
)

# ------------------ CHECK BUTTON ------------------
check = st.button("🔍 Analyze Symptoms")

if check:
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
        st.stop()

    retrieved_conditions = retrieve_conditions(selected_symptoms)

    user_input = f"""
Patient: {name}, Age: {age}, Gender: {gender}
Medical History: {history}
Symptoms: {', '.join(selected_symptoms)}
Possible dataset conditions: {', '.join(retrieved_conditions)}
"""

    st.markdown('<div class="heartbeat">❤️</div>', unsafe_allow_html=True)

    with st.spinner("Analyzing with AI..."):
        try:
            response = llm.invoke(prompt.format_prompt(input=user_input))
            raw_text = response.content.strip()
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1
            result = json.loads(raw_text[start:end])
        except:
            result = {
                "possible_conditions": retrieved_conditions or ["Unknown"],
                "next_steps": "Consult a healthcare professional.",
                "disclaimer": "This is not medical advice"
            }

    # ------------------ RESULTS ------------------
    st.subheader("🧠 Possible Conditions")
    st.markdown(f'<div class="result-card">{", ".join(result["possible_conditions"])}</div>', unsafe_allow_html=True)

    st.subheader("📌 Recommended Next Steps")
    st.markdown(f'<div class="result-card">{result["next_steps"]}</div>', unsafe_allow_html=True)

    st.subheader("⚠️ Disclaimer")
    st.markdown(f'<div class="result-card">{result["disclaimer"]}</div>', unsafe_allow_html=True)
