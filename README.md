# 💼 AI Resume Screening System

## 📌 Project Overview
The **AI Resume Screening System** is a smart web-based application developed using Streamlit. It helps both **job seekers** and **recruiters** by analyzing resumes and matching them with relevant job roles using machine learning techniques.

---

## 🚀 Features

### 👤 Job Seeker Mode
- Upload resume (PDF/TXT)
- Get best matching job role
- View match score (%)
- Automatic skill detection

### 🏢 Recruiter Mode
- Enter job description
- Upload multiple candidate resumes
- Get match score for each candidate
- Rank candidates automatically
- Highlight top candidate
- Visualize results using charts

---

## 🧠 Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib
- PyPDF2

---

## ⚙️ How It Works

1. Resume text is extracted (PDF/TXT)
2. Text is converted into numerical vectors using TF-IDF
3. Cosine similarity is calculated
4. Best match is displayed

---

## ▶️ How to Run

pip install -r requirements.txt  
streamlit run app.py

---


