"""
app.py
------
AI Career Advisor - Streamlit application.

A simple, clean Big Data Analytics microproject that helps students
identify skill gaps for a desired job role using a real LinkedIn job
postings dataset.
"""

import streamlit as st
import plotly.express as px

from analysis import (
    load_data,
    get_job_titles,
    filter_jobs_by_title,
    get_job_summary,
    get_top_skills,
    parse_user_skills,
    skill_gap_analysis,
    generate_recommendation,
)

# -----------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Career Advisor",
    page_icon="🎯",
    layout="wide"
)

DATA_PATH = "data/LinkedIn_RDB_three.csv"


# -----------------------------------------------------------------------
# Cache the dataset so it only loads once
# -----------------------------------------------------------------------
@st.cache_data
def get_data():
    return load_data(DATA_PATH)


df = get_data()

# -----------------------------------------------------------------------
# Custom CSS for a cleaner, modern look
# -----------------------------------------------------------------------
st.markdown("""
    <style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .skill-have {
        background-color: #dcfce7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
        margin: 4px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .skill-missing {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
        margin: 4px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .recommendation-box {
        background-color: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        color: #1e3a8a;
        font-size: 1.02rem;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------
st.markdown('<p class="main-title">🎯 AI Career Advisor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyze job market trends and discover the skills you need.</p>',
            unsafe_allow_html=True)

# -----------------------------------------------------------------------
# Sidebar - Project Information
# -----------------------------------------------------------------------
with st.sidebar:
    st.header("📌 Project Information")
    st.markdown("""
    **Project:** AI Career Advisor

    **Subject:** Big Data Analytics

    **Goal:** Help students identify skill gaps for their
    desired job role using real job posting data.

    **Dataset:** LinkedIn job postings
    (`LinkedIn_RDB_three.csv`)

    **Tech Stack:**
    - Python
    - Streamlit
    - Pandas
    - Plotly

    ---
    **Dataset Overview**
    """)
    st.metric("Total Job Postings", f"{len(df):,}")
    st.metric("Unique Job Titles", f"{df['title'].nunique():,}")

# -----------------------------------------------------------------------
# Main input section
# -----------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    job_titles = get_job_titles(df)
    selected_title = st.selectbox("💼 Desired Job Role", job_titles)

with col2:
    user_skills_raw = st.text_area(
        "🛠️ Enter Your Skills",
        placeholder="Example: Python, SQL, Excel",
        height=100
    )

analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=False)

st.divider()

# -----------------------------------------------------------------------
# Analysis results
# -----------------------------------------------------------------------
if analyze_clicked:
    filtered_df = filter_jobs_by_title(df, selected_title)

    if filtered_df.empty:
        st.warning(f"No job postings found for '{selected_title}'. Please try another role.")
    else:
        # ---------------- Job Summary ----------------
        st.subheader("📊 Job Summary")
        summary = get_job_summary(filtered_df)

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Matching Jobs", summary["num_jobs"])
        avg_salary_display = f"${summary['avg_salary']:,.0f}" if summary["avg_salary"] else "N/A"
        s2.metric("Avg. Compensation", avg_salary_display)
        s3.metric("Common Work Type", summary["common_work_type"])
        s4.metric("Common Experience Level", summary["common_level"])
        s5.metric("Common Location", summary["common_location"])

        st.divider()

        # ---------------- Skill Extraction ----------------
        st.subheader("🧠 Top 10 Required Skills")
        top_skills_df = get_top_skills(filtered_df, top_n=10)

        if top_skills_df.empty:
            st.info("No matching skill keywords were found in the job descriptions for this role.")
            required_skills = []
        else:
            required_skills = top_skills_df["Skill"].tolist()

            fig_skills = px.bar(
                top_skills_df,
                x="Count",
                y="Skill",
                orientation="h",
                title="Top 10 Required Skills",
                color="Count",
                color_continuous_scale="Blues",
            )
            fig_skills.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
            st.plotly_chart(fig_skills, use_container_width=True)

        st.divider()

        # ---------------- Skill Gap Analysis ----------------
        st.subheader("🧩 Skill Gap Analysis")

        user_skills = parse_user_skills(user_skills_raw)

        if not user_skills:
            st.info("Enter your skills above and click Analyze again to see your skill gap.")
        elif not required_skills:
            st.info("No required skills were detected for this role to compare against.")
        else:
            gap = skill_gap_analysis(required_skills, user_skills)

            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**✔️ Skills You Already Have**")
                if gap["have"]:
                    badges = "".join(f'<span class="skill-have">✔ {s}</span>' for s in gap["have"])
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.markdown("_None of your entered skills matched the top required skills._")

            with g2:
                st.markdown("**❌ Skills You Need**")
                if gap["missing"]:
                    badges = "".join(f'<span class="skill-missing">✘ {s}</span>' for s in gap["missing"])
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.markdown("_You already cover all the top required skills!_")

            st.divider()

            # ---------------- Learning Recommendation ----------------
            st.subheader("📚 Learning Recommendation")
            recommendation = generate_recommendation(gap["have"], gap["missing"], selected_title)
            st.markdown(f'<div class="recommendation-box">{recommendation}</div>', unsafe_allow_html=True)

        st.divider()

        # ---------------- Visualizations ----------------
        st.subheader("📈 Job Market Visualizations")

        v1, v2 = st.columns(2)

        with v1:
            work_type_counts = filtered_df["work_type"].value_counts().reset_index()
            work_type_counts.columns = ["Work Type", "Count"]
            fig_worktype = px.pie(
                work_type_counts,
                names="Work Type",
                values="Count",
                title="Work Type Distribution",
                hole=0.4,
            )
            st.plotly_chart(fig_worktype, use_container_width=True)

        with v2:
            level_counts = filtered_df["level"].value_counts().reset_index()
            level_counts.columns = ["Experience Level", "Count"]
            fig_level = px.bar(
                level_counts,
                x="Experience Level",
                y="Count",
                title="Experience Level Distribution",
                color="Count",
                color_continuous_scale="Purples",
            )
            fig_level.update_layout(showlegend=False)
            st.plotly_chart(fig_level, use_container_width=True)

        st.divider()

        # ---------------- Sample Matching Jobs ----------------
        st.subheader("🗂️ Sample Matching Jobs")
        display_cols = ["title", "work_type", "level", "job_location", "compensation", "pay_period"]
        sample_jobs = filtered_df[display_cols].head(10).reset_index(drop=True)
        sample_jobs.columns = ["Title", "Work Type", "Level", "Location", "Compensation", "Pay Period"]
        st.dataframe(sample_jobs, use_container_width=True)

else:
    st.info("👈 Select a job role and enter your skills, then click **Analyze** to get started.")