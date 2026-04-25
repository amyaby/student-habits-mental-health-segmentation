import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
import streamlit.components.v1 as components
from groq import Groq

# ── Load models ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kmeans        = pickle.load(open(os.path.join(BASE_DIR, 'models/kmeans.pkl'), 'rb'))
scaler        = pickle.load(open(os.path.join(BASE_DIR, 'models/scaler.pkl'), 'rb'))
features      = pickle.load(open(os.path.join(BASE_DIR, 'models/features.pkl'), 'rb'))
cluster_names = pickle.load(open(os.path.join(BASE_DIR, 'models/cluster_names.pkl'), 'rb'))
df_clusters   = pd.read_csv(os.path.join(BASE_DIR, 'data/processed/students_clustered.csv'))

# ── Groq client ──────────────────────────────────────────────
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ── Session state init ───────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 0
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'advice' not in st.session_state:
    st.session_state.advice = None

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="StudyWELL", page_icon="😉", layout="centered")

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0f1117; }
h1, h2, h3 { color: #ffffff; }
.stButton > button {
    border-radius: 20px;
    padding: 10px 30px;
    font-weight: 600;
}
.stProgress > div > div {
    background-color: #7C83FD;
}
</style>
""", unsafe_allow_html=True)

# ── Progress bar ─────────────────────────────────────────────
total_pages = 4
if 0 < st.session_state.page <= total_pages:
    st.progress(st.session_state.page / total_pages)

def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1

# ════════════════════════════════════════════════════════════
# PAGE 0 — Welcome
# ════════════════════════════════════════════════════════════
if st.session_state.page == 0:
    st.title("😉 StudyWELL")
    st.subheader("Discover your learner profile")
    st.markdown("""
    Answer **4 short sections** about your daily habits.

    We'll identify which type of student you are and give you
    a **personalized AI-powered improvement plan**.

    Takes less than 2 minutes.
    """)
    st.divider()
    st.button("Start →", type="primary", on_click=next_page)

# ════════════════════════════════════════════════════════════
# PAGE 1 — Study & Attendance
# ════════════════════════════════════════════════════════════
elif st.session_state.page == 1:
    st.title("📚 Study habits")
    st.caption("Section 1 of 4")
    st.divider()

    study = st.slider("How many hours do you study per day?",
                      0.0, 12.0,
                      st.session_state.answers.get('study', 3.0), 0.5)

    attendance = st.slider("What % of classes do you attend?",
                           0, 100,
                           st.session_state.answers.get('attendance', 75))

    time_mgmt_q = st.radio("How well do you manage your time?",
                            ["Poorly", "Sometimes", "Usually", "Very well"],
                            index=st.session_state.answers.get('time_mgmt_idx', 1))

    motivation_q = st.radio("How motivated are you to study?",
                             ["Not at all", "A little", "Moderately", "Very motivated"],
                             index=st.session_state.answers.get('motivation_idx', 2))

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.button("← Back", on_click=prev_page)
    with col2:
        if st.button("Next →", type="primary"):
            st.session_state.answers['study']         = study
            st.session_state.answers['attendance']    = attendance
            st.session_state.answers['time_mgmt_q']   = time_mgmt_q
            st.session_state.answers['time_mgmt_idx'] = ["Poorly","Sometimes","Usually","Very well"].index(time_mgmt_q)
            st.session_state.answers['motivation_q']  = motivation_q
            st.session_state.answers['motivation_idx']= ["Not at all","A little","Moderately","Very motivated"].index(motivation_q)
            next_page()

# ════════════════════════════════════════════════════════════
# PAGE 2 — Sleep & Health
# ════════════════════════════════════════════════════════════
elif st.session_state.page == 2:
    st.title("😴 Sleep & Health")
    st.caption("Section 2 of 4")
    st.divider()

    sleep = st.slider("How many hours do you sleep per night?",
                      3.0, 10.0,
                      st.session_state.answers.get('sleep', 7.0), 0.5)

    exercise_q = st.radio("How often do you exercise?",
                           ["Never", "1-2x/week", "3-4x/week", "Daily"],
                           index=st.session_state.answers.get('exercise_idx', 1))

    diet_q = st.radio("How would you describe your diet?",
                       ["Poor", "Average", "Good"],
                       index=st.session_state.answers.get('diet_idx', 1))

    sleep_qual_q = st.radio("How is your sleep quality?",
                             ["Poor", "Average", "Good"],
                             index=st.session_state.answers.get('sleep_qual_idx', 1))

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.button("← Back", on_click=prev_page)
    with col2:
        if st.button("Next →", type="primary"):
            st.session_state.answers['sleep']          = sleep
            st.session_state.answers['exercise_q']     = exercise_q
            st.session_state.answers['exercise_idx']   = ["Never","1-2x/week","3-4x/week","Daily"].index(exercise_q)
            st.session_state.answers['diet_q']         = diet_q
            st.session_state.answers['diet_idx']       = ["Poor","Average","Good"].index(diet_q)
            st.session_state.answers['sleep_qual_q']   = sleep_qual_q
            st.session_state.answers['sleep_qual_idx'] = ["Poor","Average","Good"].index(sleep_qual_q)
            next_page()

# ════════════════════════════════════════════════════════════
# PAGE 3 — Screen & Social
# ════════════════════════════════════════════════════════════
elif st.session_state.page == 3:
    st.title("📱 Screen & Social life")
    st.caption("Section 3 of 4")
    st.divider()

    social_media = st.slider("Social media hours per day?",
                              0.0, 10.0,
                              st.session_state.answers.get('social_media', 2.0), 0.5)

    netflix = st.slider("Netflix / YouTube hours per day?",
                         0.0, 8.0,
                         st.session_state.answers.get('netflix', 1.0), 0.5)

    social_act_q = st.radio("How socially active are you?",
                              ["Very isolated", "A little", "Moderately", "Very social"],
                              index=st.session_state.answers.get('social_act_idx', 1))

    stress_q = st.radio("How stressed do you feel lately?",
                          ["Not stressed", "A little", "Quite stressed", "Very stressed"],
                          index=st.session_state.answers.get('stress_idx', 1))

    burnout_q = st.radio("Do you feel burned out from studying?",
                           ["Not at all", "Sometimes", "Often", "Always"],
                           index=st.session_state.answers.get('burnout_idx', 1))

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.button("← Back", on_click=prev_page)
    with col2:
        if st.button("Next →", type="primary"):
            st.session_state.answers['social_media']   = social_media
            st.session_state.answers['netflix']        = netflix
            st.session_state.answers['social_act_q']   = social_act_q
            st.session_state.answers['social_act_idx'] = ["Very isolated","A little","Moderately","Very social"].index(social_act_q)
            st.session_state.answers['stress_q']       = stress_q
            st.session_state.answers['stress_idx']     = ["Not stressed","A little","Quite stressed","Very stressed"].index(stress_q)
            st.session_state.answers['burnout_q']      = burnout_q
            st.session_state.answers['burnout_idx']    = ["Not at all","Sometimes","Often","Always"].index(burnout_q)
            next_page()

# ════════════════════════════════════════════════════════════
# PAGE 4 — Support & Wellbeing
# ════════════════════════════════════════════════════════════
elif st.session_state.page == 4:
    st.title("🤝 Support & Wellbeing")
    st.caption("Section 4 of 4")
    st.divider()

    support_q = st.radio("Do you have people to talk to when stressed?",
                          ["No one", "Sometimes", "Yes, a few", "Strong support"],
                          index=st.session_state.answers.get('support_idx', 1))

    fin_stress_q = st.radio("How much financial stress do you have?",
                              ["None", "A little", "Moderate", "A lot"],
                              index=st.session_state.answers.get('fin_stress_idx', 1))

    parental_q = st.radio("How supportive is your family?",
                           ["Not supportive", "A little", "Quite supportive", "Very supportive"],
                           index=st.session_state.answers.get('parental_idx', 1))

    academic_p_q = st.radio("How much academic pressure do you feel?",
                              ["Very low", "Moderate", "High", "Extreme"],
                              index=st.session_state.answers.get('academic_idx', 1))

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.button("← Back", on_click=prev_page)
    with col2:
        if st.button("🔍 Analyze my profile", type="primary"):
            st.session_state.answers['support_q']      = support_q
            st.session_state.answers['support_idx']    = ["No one","Sometimes","Yes, a few","Strong support"].index(support_q)
            st.session_state.answers['fin_stress_q']   = fin_stress_q
            st.session_state.answers['fin_stress_idx'] = ["None","A little","Moderate","A lot"].index(fin_stress_q)
            st.session_state.answers['parental_q']     = parental_q
            st.session_state.answers['parental_idx']   = ["Not supportive","A little","Quite supportive","Very supportive"].index(parental_q)
            st.session_state.answers['academic_p_q']   = academic_p_q
            st.session_state.answers['academic_idx']   = ["Very low","Moderate","High","Extreme"].index(academic_p_q)
            st.session_state.advice = None
            st.session_state.chat_history = []
            next_page()

# ════════════════════════════════════════════════════════════
# PAGE 5 — Results + Chatbot
# ════════════════════════════════════════════════════════════
elif st.session_state.page == 5:
    a = st.session_state.answers

    def to_scale(answer, options):
        idx = options.index(answer)
        return round(1 + (idx / (len(options) - 1)) * 9, 1)

    study        = a['study']
    sleep        = a['sleep']
    attendance   = a['attendance']
    social_media = a['social_media']
    netflix      = a['netflix']
    screen       = social_media + netflix
    time_mgmt    = to_scale(a['time_mgmt_q'],  ["Poorly","Sometimes","Usually","Very well"])
    motivation   = to_scale(a['motivation_q'],  ["Not at all","A little","Moderately","Very motivated"])
    exercise     = to_scale(a['exercise_q'],    ["Never","1-2x/week","3-4x/week","Daily"]) * 0.7
    diet         = ["Poor","Average","Good"].index(a['diet_q'])
    sleep_qual   = ["Poor","Average","Good"].index(a['sleep_qual_q'])
    social_act   = to_scale(a['social_act_q'],  ["Very isolated","A little","Moderately","Very social"])
    stress       = to_scale(a['stress_q'],      ["Not stressed","A little","Quite stressed","Very stressed"])
    burnout      = ["Not at all","Sometimes","Often","Always"].index(a['burnout_q'])
    social_sup   = to_scale(a['support_q'],     ["No one","Sometimes","Yes, a few","Strong support"])
    fin_stress   = to_scale(a['fin_stress_q'],  ["None","A little","Moderate","A lot"])
    parental_sup = to_scale(a['parental_q'],    ["Not supportive","A little","Quite supportive","Very supportive"])
    academic_p   = to_scale(a['academic_p_q'],  ["Very low","Moderate","High","Extreme"])

    anxiety     = min(10, stress * 1.1 + (10 - sleep) * 0.3)
    depression  = min(10, (10 - motivation) * 0.8 + burnout * 1.5)
    mental_h    = max(1,  10 - (stress * 0.4 + anxiety * 0.3 + burnout * 0.5))
    physical    = exercise * 2
    daily_study = study
    daily_sleep = sleep

    user_input = np.array([[
        study, daily_study, attendance, time_mgmt, motivation,
        sleep, daily_sleep, exercise, physical,
        diet, sleep_qual,
        mental_h,
        stress, anxiety, depression, anxiety,
        burnout, academic_p,
        social_media, netflix, screen, social_act,
        fin_stress, social_sup, parental_sup
    ]])

    user_scaled   = scaler.transform(user_input)
    cluster_id    = kmeans.predict(user_scaled)[0]
    cluster_label = cluster_names[cluster_id]

    cluster_profile = df_clusters[df_clusters['cluster_name'] == cluster_label][
        ['anxiety_score', 'depression_score', 'burnout_level_enc',
         'mental_health_rating', 'academic_pressure_score',
         'financial_stress_score', 'social_support_score']
    ].mean().round(2)
    mh = cluster_profile

    # ── Colored banner ────────────────────────────────────
    colors = {
        "High-Anxiety Low-Motivation Students":  "#FF6B6B",
        "Moderate-Study Low-Stress Students":    "#4ECDC4",
        "High-Screen-Time Burned-Out Students":  "#FFE66D",
        "High-Screen-Time Relaxed Students":     "#A8E6CF"
    }
    color = colors.get(cluster_label, "#7C83FD")

    st.title(" 😉 Your StudyWell Profile")
    st.markdown(f"""
<div style="background:{color}22; border-left:6px solid {color};
            padding:24px; border-radius:12px; margin:10px 0">
    <p style="color:{color}; margin:0; font-size:15px; font-weight:500">
        You belong to this student group:
    </p>
    <h2 style="color:{color}; margin:8px 0">{cluster_label}</h2>
    <p style="color:#aaaaaa; margin:0; font-size:13px">
        Based on your habits, our model placed you in this segment
        out of 4 groups identified from 80,000+ students.
    </p>
</div>
""", unsafe_allow_html=True)

    st.divider()

    st.subheader("📊 Your habits at a glance")
    col1, col2, col3 = st.columns(3)
    col1.metric("📖 Study", f"{study}h/day")
    col2.metric("😴 Sleep", f"{sleep}h/night")
    col3.metric("📱 Screen", f"{screen:.1f}h/day")

    col4, col5, col6 = st.columns(3)
    col4.metric("🎓 Attendance", f"{attendance}%")
    col5.metric("💪 Motivation", f"{motivation:.1f}/10")
    col6.metric("😤 Stress", f"{stress:.1f}/10")

    st.divider()

    st.subheader("🧠 Mental health profile of your group")
    st.caption("Average values for students classified in the same segment as you.")

    col7, col8, col9 = st.columns(3)
    col7.metric("Anxiety (avg)",    f"{mh['anxiety_score']}/10",
                delta="High" if mh['anxiety_score'] > 6 else "Normal",
                delta_color="inverse")
    col8.metric("Depression (avg)", f"{mh['depression_score']}/10",
                delta="High" if mh['depression_score'] > 6 else "Normal",
                delta_color="inverse")
    col9.metric("Burnout (avg)",    f"{mh['burnout_level_enc']}/2",
                delta="High" if mh['burnout_level_enc'] > 1 else "Low",
                delta_color="inverse")

    st.divider()

    st.subheader("📈 Visual analysis — where you stand")
    tab1, tab2, tab3 = st.tabs(["🕸 Cluster radar", "🔵 Student map", "📦 Exam scores"])

    with tab1:
        radar_path = os.path.join(BASE_DIR, 'radar_clusters.html')
        if os.path.exists(radar_path):
            with open(radar_path, 'r') as f:
                components.html(f.read(), height=500, scrolling=False)
        else:
            st.info("Run the radar chart cell in your notebook first.")

    with tab2:
        pca_path = os.path.join(BASE_DIR, 'pca_clusters.html')
        if os.path.exists(pca_path):
            with open(pca_path, 'r') as f:
                components.html(f.read(), height=500, scrolling=False)
        else:
            st.info("Run the PCA cell in your notebook first.")

    with tab3:
        box_path = os.path.join(BASE_DIR, 'boxplot_scores.html')
        if os.path.exists(box_path):
            with open(box_path, 'r') as f:
                components.html(f.read(), height=500, scrolling=False)
        else:
            st.info("Run the boxplot cell in your notebook first.")

    st.divider()

    # ── AI personalized plan ──────────────────────────────
    st.subheader("🤖 Your personalized plan")

    if st.session_state.advice is None:
        prompt = f"""A student completed a lifestyle assessment and was classified as: '{cluster_label}'.

What they reported:
- Study: {study}h/day, Sleep: {sleep}h/night, Attendance: {attendance}%
- Social media: {social_media}h/day, Screen time: {screen:.1f}h/day
- They feel: stress={a['stress_q']}, burnout={a['burnout_q']}, motivation={a['motivation_q']}
- Support system: {a['support_q']}, Family support: {a['parental_q']}
- Academic pressure: {a['academic_p_q']}, Financial stress: {a['fin_stress_q']}

Based on 80,000+ students with the same profile, this group typically shows:
- Anxiety level: {mh['anxiety_score']}/10
- Depression indicators: {mh['depression_score']}/10
- Burnout severity: {mh['burnout_level_enc']}/2
- Mental health rating: {mh['mental_health_rating']}/10

Write a warm, direct, encouraging response with:
1. What this profile means including the mental health pattern (2 sentences max)
2. Their 3 biggest risks if nothing changes (specific to their numbers)
3. A concrete 4-week improvement plan — one goal per week, addressing BOTH habits AND mental wellbeing
4. One powerful motivational closing insight

Use simple language. Be specific, not generic."""

        with st.spinner("🤖 Building your personalized plan..."):
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            st.session_state.advice = response.choices[0].message.content

    st.markdown(st.session_state.advice)
    st.divider()

    # ── Chatbot ───────────────────────────────────────────
    st.subheader("💬 Ask a follow-up question")
    # Chat input FIRST
    followup = st.chat_input("e.g. How do I sleep better? How do I reduce stress before exams?")
    st.caption("Ask anything about your profile, your plan, or how to improve.")
    # Show previous chat messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    if followup:
        with st.chat_message("user"):
            st.markdown(followup)
        st.session_state.chat_history.append({"role": "user", "content": followup})

        chat_prompt = f"""You are an academic wellbeing advisor.

Student profile: '{cluster_label}'
Habits: study={study}h/day, sleep={sleep}h/night, stress={a['stress_q']},
social_media={social_media}h/day, motivation={a['motivation_q']}, burnout={a['burnout_q']}
Cluster mental health avg: anxiety={mh['anxiety_score']}/10, depression={mh['depression_score']}/10

Plan already given: {st.session_state.advice}

Student question: {followup}

Answer in 3-5 sentences. Be specific, practical, and encouraging."""

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_resp = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": chat_prompt}],
                    max_tokens=500
                )
                reply = chat_resp.choices[0].message.content
                st.markdown(reply)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": reply
        })
   
    if st.button("🔄 Start over"):
        st.session_state.page = 0
        st.session_state.answers = {}
        st.session_state.chat_history = []
        st.session_state.advice = None
        st.rerun()