# yash vaghasiya
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# ----------------- Sample Training Data (10 parameters) -----------------
data = {
    "cgpa": [7.3, 8.1, 8.5, 9.0, 6.9, 7.8, 8.9, 7.1],
    "python": [1, 1, 0, 1, 0, 1, 1, 0],
    "ml": [1, 0, 1, 1, 0, 1, 1, 0],
    "communication": [1, 0, 1, 1, 0, 1, 0, 0],
    "creativity": [1, 0, 1, 1, 0, 0, 1, 0],
    "teamwork": [1, 1, 0, 1, 0, 1, 0, 1],
    "leadership": [0, 1, 0, 1, 0, 1, 1, 0],
    "data_analysis": [1, 0, 1, 1, 0, 1, 1, 0],
    "research": [1, 0, 1, 0, 1, 0, 1, 0],
    "tools": [1, 1, 0, 1, 0, 1, 1, 0],
    "career": [
        "AI Engineer", "Software Developer", "Data Scientist", "AI Engineer",
        "Tester", "ML Engineer", "Data Scientist", "Web Developer"
    ]
}

df = pd.DataFrame(data)

X = df.drop("career", axis=1)
y = df["career"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ----------------- Career Skills -----------------
career_skills = {
    "AI Engineer": ["Deep Learning", "TensorFlow", "NLP Projects"],
    "Software Developer": ["System Design", "DSA", "Backend Development"],
    "Data Scientist": ["Data Visualization", "Pandas", "Statistics"],
    "ML Engineer": ["MLOps", "Model Deployment", "Cloud AI"],
    "Tester": ["Automation Testing", "Selenium", "Bug Tracking"],
    "Web Developer": ["ReactJS", "Django", "Fullstack Projects"]
}

# ----------------- ML Prediction -----------------
def recommend_career(params):
    prediction = model.predict([params])[0]
    probs = model.predict_proba([params])[0]
    prob_df = pd.DataFrame({"Career": model.classes_, "Probability": probs})
    return prediction, career_skills.get(prediction, []), prob_df

# ----------------- AI Chatbot -----------------
def ai_chatbot(user_input):
    tokens = word_tokenize(user_input.lower())
    filtered = [w for w in tokens if w not in stopwords.words('english')]
    if "ai" in filtered or "intelligence" in filtered:
        return "AI Engineer"
    elif "data" in filtered or "analysis" in filtered or "visualization" in filtered:
        return "Data Scientist"
    elif "web" in filtered or "frontend" in filtered or "backend" in filtered:
        return "Web Developer"
    elif "test" in filtered or "qa" in filtered:
        return "Tester"
    elif "ml" in filtered or "machine" in filtered:
        return "ML Engineer"
    elif "software" in filtered or "coding" in filtered:
        return "Software Developer"
    else:
        return "Not sure, please describe more."

# ----------------- Streamlit GUI -----------------
st.set_page_config(page_title="AI + ML Career Recommender", layout="wide")
st.title("🎓 AI + ML Career & Skill Recommender")
st.write("Predict your career path with ML and get AI-driven skill advice.")

tab1, tab2 = st.tabs(["📊 ML Career Prediction", "🤖 AI Chatbot Advisor"])

with tab1:
    st.subheader("📊 Enter Your Academic & Skills Profile")

    cgpa = st.slider("Your CGPA", 5.0, 10.0, 8.0)
    python = st.selectbox("Do you know Python?", ["Yes", "No"])
    ml_skill = st.selectbox("Do you know Machine Learning?", ["Yes", "No"])
    comm = st.selectbox("Are you good at Communication?", ["Yes", "No"])
    creativity = st.selectbox("Are you Creative?", ["Yes", "No"])
    teamwork = st.selectbox("Are you good at Teamwork?", ["Yes", "No"])
    leadership = st.selectbox("Do you have Leadership skills?", ["Yes", "No"])
    data_analysis = st.selectbox("Do you know Data Analysis?", ["Yes", "No"])
    research = st.selectbox("Are you good at Research & Learning?", ["Yes", "No"])
    tools = st.selectbox("Do you know AI/ML Tools (TensorFlow, PyTorch etc.)?", ["Yes", "No"])

    if st.button("🔮 Predict Career"):
        params = [
            cgpa,
            1 if python == "Yes" else 0,
            1 if ml_skill == "Yes" else 0,
            1 if comm == "Yes" else 0,
            1 if creativity == "Yes" else 0,
            1 if teamwork == "Yes" else 0,
            1 if leadership == "Yes" else 0,
            1 if data_analysis == "Yes" else 0,
            1 if research == "Yes" else 0,
            1 if tools == "Yes" else 0,
        ]
        career, skills, prob_df = recommend_career(params)
        st.success(f"🎯 Predicted Career Path: **{career}**")
        st.write("📘 Suggested Skills to Learn:", ", ".join(skills))
        st.write("📊 Career Prediction Probabilities:")
        st.bar_chart(prob_df.set_index("Career"))

with tab2:
    st.subheader("🤖 Chat with AI Career Advisor")
    user_input = st.text_area("Tell me about your interests, skills, or dreams:")
    if st.button("💬 Get Advice"):
        career_ai = ai_chatbot(user_input)
        if career_ai == "Not sure, please describe more.":
            st.warning(career_ai)
        else:
            st.success(f"🤖 AI suggests you may fit as a **{career_ai}**")
            st.write("📘 Suggested Skills:", ", ".join(career_skills.get(career_ai, [])))
