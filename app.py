import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import pandas as pd

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="AI Resume Matcher", layout="wide")

# ---------------------------
# CUSTOM UI
# ---------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.card {
    padding: 20px;
    border-radius: 12px;
    background: #1c1f26;
    margin-bottom: 15px;
    border: 1px solid #2a2d34;
}
.metric {
    font-size: 20px;
    color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# HEADER
# ---------------------------
st.markdown("""
<div style="padding:20px; border-radius:12px; background: linear-gradient(90deg, #4CAF50, #2E7D32);">
    <h1 style="color:white;">💼 AI Resume Screening System</h1>
    <p style="color:white;">Smart Hiring & Job Matching Platform</p>
</div>
""", unsafe_allow_html=True)

mode = st.radio("", ["👤 Job Seeker", "🏢 Recruiter"], horizontal=True)

# ---------------------------
# FUNCTIONS
# ---------------------------
def extract_text(file):
    text = ""
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    else:
        text = file.read().decode("utf-8")
    return text.lower()

skills_db = [
    "python","java","c++","html","css","javascript",
    "react","sql","machine learning","data science",
    "excel","power bi","django","flask"
]

def extract_skills(text):
    return [skill for skill in skills_db if skill in text]

# Load model
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---------------------------
# JOB SEEKER MODE
# ---------------------------
if "Job Seeker" in mode:
    st.subheader("👤 Job Seeker Dashboard")

    jobs = [
        "python developer machine learning data science",
        "web developer html css javascript react",
        "java developer spring boot backend",
        "data analyst excel sql power bi"
    ]

    file = st.file_uploader("📄 Upload Resume", type=["pdf","txt"])

    if st.button("🔍 Analyze Resume"):
        if file:
            text = extract_text(file)

            skills = extract_skills(text)

            resume_vec = vectorizer.transform([text])
            job_vec = vectorizer.transform(jobs)
            scores = cosine_similarity(resume_vec, job_vec)[0]

            best_job = jobs[scores.argmax()]
            score = max(scores) * 100

            # RESULT UI
            st.markdown("### 🎯 Analysis Result")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
                <div class="card">
                    <h4>💼 Best Job Role</h4>
                    <p class="metric">{best_job}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="card">
                    <h4>📊 Match Score</h4>
                    <p class="metric">{score:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

            st.progress(int(score))

            # SKILLS
            st.markdown("### 🧠 Detected Skills")
            if skills:
                st.write(", ".join(skills))
            else:
                st.warning("No skills detected")

        else:
            st.warning("Upload resume first")

# ---------------------------
# RECRUITER MODE
# ---------------------------
elif "Recruiter" in mode:
    st.subheader("🏢 Recruiter Dashboard")

    st.markdown("""
    <div class="card">
        <h3>📌 Upload resumes and find best candidates</h3>
    </div>
    """, unsafe_allow_html=True)

    job_desc = st.text_area("📝 Enter Job Description")
    files = st.file_uploader("📄 Upload Resumes", type=["pdf","txt"], accept_multiple_files=True)

    if st.button("📊 Analyze Candidates"):
        if job_desc and files:

            names = []
            scores_list = []

            job_vec = vectorizer.transform([job_desc])

            for file in files:
                text = extract_text(file)
                resume_vec = vectorizer.transform([text])

                score = cosine_similarity(job_vec, resume_vec)[0][0] * 100

                names.append(file.name)
                scores_list.append(score)

            df = pd.DataFrame({
                "Candidate": names,
                "Score": scores_list
            }).sort_values(by="Score", ascending=False)

            # TOP CANDIDATE
            top_candidate = df.iloc[0]

            st.markdown(f"""
            <div class="card">
                <h3>🏆 Top Candidate</h3>
                <p><b>{top_candidate['Candidate']}</b></p>
                <p class="metric">{top_candidate['Score']:.2f}% Match</p>
            </div>
            """, unsafe_allow_html=True)

            # TABLE
            st.markdown("### 📋 Candidate Ranking")
            st.dataframe(df)

        else:
            st.warning("Fill all fields")