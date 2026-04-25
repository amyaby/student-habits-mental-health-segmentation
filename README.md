Here's your updated README.md:

# 😉 StudyWELL — Student Habits & Mental Health Segmentation

A machine learning web application that analyzes student lifestyle habits and mental health patterns to automatically classify them into behavioral segments, then provides personalized AI-generated improvement plans.

## Live Demo
[Open StudyWELL](https://student-habits-mental-health-segmentation-m5z3qe5szfjq3u2v79hl.streamlit.app/)

> 💡 Try the app to discover your student profile — answer 4 short sections about your daily habits and get a personalized AI improvement plan in seconds.  
> ⚠️ **If you're forking this project:** don't forget to add your API key in Streamlit Cloud.  
> Go to **Manage App** (bottom right of your app) → **Settings** → **Secrets** → paste:  
> ```toml
> GROQ_API_KEY = "gsk_..."
> ```
> Without this the AI chatbot won't work.

---

## Project Overview

Universities treat all students the same way — one generic advice fits nobody. StudyWELL identifies which type of student you are based on your daily habits and mental health indicators, then gives you targeted, specific advice instead of generic recommendations.

## How it works

1. **Data** — Two Kaggle datasets merged: student lifestyle habits (80,000+ students) + mental health indicators
2. **ML Model** — K-Means clustering discovers 4 natural student segments from 25 behavioral features
3. **Validation** — PCA scatter plot, radar chart, and box plot confirm segments are meaningful
4. **App** — Student answers 4 sections of simple questions → model classifies them → AI generates personalized plan

## The 4 Student Segments

| Segment | Profile |
|---------|---------|
| 🔴 High-Anxiety Low-Motivation Students | High stress and anxiety, low motivation despite studying |
| 🟢 High-Screen-Time Relaxed Students | High social media usage, low academic engagement |
| 🟡 High-Screen-Time Burned-Out Students | Excessive screen time combined with academic burnout |
| 🩵 Moderate-Study Low-Stress Students | Balanced habits, moderate performance |

## Project Structure

```
student-habits-mental-health-segmentation/
├── app/
│   └── app.py              ← Streamlit web application
├── models/
│   ├── kmeans.pkl          ← trained K-Means model
│   ├── scaler.pkl          ← fitted StandardScaler
│   ├── features.pkl        ← feature names list
│   └── cluster_names.pkl   ← cluster labels mapping
├── data/
│   └── processed/
│       └── students_clustered.csv
├── radar_clusters.html     ← interactive radar chart
├── pca_clusters.html       ← PCA scatter plot
├── boxplot_scores.html     ← exam score distribution
├── script.ipynb            ← full ML notebook
└── requirements.txt        ← all dependencies for deployment


## ML Pipeline

```
Two CSV datasets (80,000+ students)
        ↓
Merge on student_id
        ↓
Clean + encode categoricals (LabelEncoder)
        ↓
Select 25 behavioral features
        ↓
Scale features (StandardScaler)
        ↓
Find optimal K (Elbow + Silhouette → K=4)
        ↓
Train K-Means clustering
        ↓
Name and interpret clusters
        ↓
Deploy with Streamlit + Groq AI chatbot
```

## Datasets

| Dataset | Source | Size | Features |
|---------|--------|------|----------|
| Student Habits & Academic Performance | Kaggle | 80,000 rows | 31 columns |
| Student Mental Health & Burnout | Kaggle | 150,000 rows | 20 columns |

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10 |
| ML | scikit-learn (KMeans, StandardScaler, PCA) |
| Data | pandas, numpy |
| Visualization | plotly |
| App | Streamlit |
| AI Chatbot | Groq API (Llama 3.3 70B) |
| Deployment | Streamlit Cloud |

## Run Locally

```bash
# Clone the repo
git clone https://github.com/amyaby/student-habits-mental-health-segmentation.git
cd student-habits-mental-health-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Why requirements.txt?**
> Streamlit Cloud starts from a fresh empty server — it has nothing installed.
> It reads `requirements.txt` and installs each library listed there before running your app.
> Your local machine already has everything in your `venv`, but the cloud knows nothing until you tell it.
> Without this file: no pandas, no sklearn, no groq, app crashes immediately.

```bash
# Add your Groq API key
mkdir .streamlit
echo 'GROQ_API_KEY = "gsk_..."' > .streamlit/secrets.toml

# Run the app
streamlit run app/app.py
```

## Results

- **K=4** clusters identified using Elbow method + Silhouette score
- Clusters are visually separated in PCA 2D projection
- Each cluster shows distinct behavioral fingerprint in radar chart
- Exam score distributions differ meaningfully across segments

##  Author

**Imane** — Engineering student, ENSA Berrechid
S8 Data Mining Project
```