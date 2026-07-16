"""
app.py
------
AI Career Advisor - Streamlit application.

A simple, clean Big Data Analytics microproject that helps students
identify skill gaps for a desired job role using a real LinkedIn job
postings dataset, with an optional Gemini-powered AI recommendation.
"""

import os

import streamlit as st
import plotly.express as px
import plotly.io as pio
from dotenv import load_dotenv

from analysis import (
    load_data,
    get_job_titles,
    filter_jobs_by_title,
    get_job_summary,
    get_salary_stats,
    get_top_skills,
    get_top_hiring_companies,
    parse_user_skills,
    skill_gap_analysis,
    calculate_match_score,
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

# Dark plotly template so charts match the app theme, and default to
# stretching to the full width of their container. (Styling only - the
# underlying chart data/analysis is unchanged.)
pio.templates.default = "plotly_dark"
CHART_COLORWAY = ["#22d3ee", "#3b82f6", "#818cf8", "#06b6d4", "#60a5fa"]


# =========================================================================
# GEMINI AI CONFIGURATION
# -------------------------------------------------------------------------
# Reads the API key from a local .env file (never hardcoded). If the key
# is missing, or the API call fails for any reason (no internet, invalid
# key, quota, etc.), the app degrades gracefully and simply skips the
# AI recommendation section instead of crashing.
# =========================================================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


@st.cache_data(show_spinner=False)
def get_gemini_recommendation(role: str, top_skills: tuple, missing_skills: tuple, job_stats: dict):
    """
    Ask Gemini for a professional recommendation covering a Summary,
    Career Advice, a Suggested Learning Roadmap, and Future Skills.

    Returns (text, error). `text` is None if the recommendation could not
    be generated, in which case `error` holds a short, user-safe reason
    ("no_key", "import_error", or the exception message).
    """
    if not GEMINI_API_KEY:
        return None, "no_key"

    try:
        from google import genai
    except ImportError:
        return None, "import_error"

    prompt = f"""
You are a professional career advisor. Based on the following job market
analysis, write a concise, encouraging, and practical recommendation for a
student.

Desired Job Role: {role}
Top Required Skills (from real job postings): {', '.join(top_skills) if top_skills else 'Not enough data'}
Skills the Student is Missing: {', '.join(missing_skills) if missing_skills else 'None - they already cover the top skills'}
Job Market Statistics: {job_stats}

Structure your response in Markdown with exactly these four sections:
### Summary
### Career Advice
### Suggested Learning Roadmap
### Future Skills

Keep it focused and practical, around 200-300 words total.
"""

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text, None
    except Exception as exc:  # noqa: BLE001 - any failure should degrade gracefully
        return None, str(exc)


# -----------------------------------------------------------------------
# Cache the dataset so it only loads once
# -----------------------------------------------------------------------
@st.cache_data
def get_data():
    return load_data(DATA_PATH)


df = get_data()


# -----------------------------------------------------------------------
# Frontend asset loaders
# -----------------------------------------------------------------------
def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()

# -----------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------
with open("assets/hero.html") as f:
    st.markdown(f.read(), unsafe_allow_html=True)

# -----------------------------------------------------------------------
# Top-level metric cards (dataset overview)
# -----------------------------------------------------------------------
total_jobs = len(df)
unique_titles = df["title"].nunique()
companies_hiring = df["company_id"].nunique()

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">📂</div>
        <div class="metric-label">Total Jobs in Dataset</div>
        <div class="metric-value">{total_jobs:,}</div>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">🧭</div>
        <div class="metric-label">Unique Job Titles</div>
        <div class="metric-value">{unique_titles:,}</div>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">🏢</div>
        <div class="metric-label">Companies Hiring</div>
        <div class="metric-value">{companies_hiring:,}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

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
    - Google Gemini (optional AI recommendations)

    ---
    **Dataset Overview**
    """)
    st.metric("Total Job Postings", f"{len(df):,}")
    st.metric("Unique Job Titles", f"{df['title'].nunique():,}")

    st.markdown("---")
    if GEMINI_API_KEY:
        st.success("🤖 Gemini AI recommendations: **Enabled**")
    else:
        st.info("🤖 Gemini AI recommendations: **Disabled**\n\nAdd `GEMINI_API_KEY` to a `.env` file to enable them.")

# -----------------------------------------------------------------------
# Main input section
# -----------------------------------------------------------------------
st.markdown('<p class="section-heading">🔎 Find Your Skill Gap</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    job_titles = get_job_titles(df)
    selected_title = st.selectbox("💼 Desired Job Role", job_titles)

with col2:
    user_skills_raw = st.text_area(
        "🛠️ Enter Your Skills",
        placeholder="Example: Python, SQL, Excel",
        height=110
    )

analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=False)

st.divider()

# -----------------------------------------------------------------------
# Analysis results
# -----------------------------------------------------------------------
if analyze_clicked:
    # Partial, case-insensitive "contains" match - e.g. selecting
    # "Full Stack Developer" also matches "Senior Full Stack Developer",
    # "Java Full Stack Developer", etc.
    filtered_df = filter_jobs_by_title(df, selected_title)

    if filtered_df.empty:
        st.warning(f"No job postings found for '{selected_title}'. Please try another role.")
    else:
        # ---------------- Job Statistics ----------------
        st.markdown('<p class="section-heading">📊 Job Statistics</p>', unsafe_allow_html=True)
        summary = get_job_summary(filtered_df)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Matching Jobs", summary["num_jobs"])
        s2.metric("Common Work Type", summary["common_work_type"])
        s3.metric("Common Experience Level", summary["common_level"])
        s4.metric("Common Location", summary["common_location"])

        st.divider()

        # ---------------- Salary Analytics ----------------
        st.markdown('<p class="section-heading">💰 Salary Analytics</p>', unsafe_allow_html=True)
        salary_stats = get_salary_stats(filtered_df)

        if salary_stats["average"] is None:
            st.info("No salary data is available for the matching job postings.")
        else:
            sal1, sal2, sal3 = st.columns(3)
            sal1.metric("Average Salary", f"${salary_stats['average']:,.0f}")
            sal2.metric("Highest Salary", f"${salary_stats['highest']:,.0f}")
            sal3.metric("Lowest Salary", f"${salary_stats['lowest']:,.0f}")

        st.divider()

        # ---------------- Top Hiring Companies ----------------
        st.markdown('<p class="section-heading">🏢 Top 5 Hiring Companies</p>', unsafe_allow_html=True)
        top_companies_df = get_top_hiring_companies(filtered_df, top_n=5)

        if top_companies_df.empty:
            st.info("No company data is available for the matching job postings.")
        else:
            st.caption("The dataset identifies companies by ID rather than name.")
            fig_companies = px.bar(
                top_companies_df,
                x="Job Postings",
                y="Company ID",
                orientation="h",
                color="Job Postings",
                color_continuous_scale=["#0f3b47", "#22d3ee"],
            )
            fig_companies.update_layout(
                yaxis=dict(type="category", autorange="reversed"),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                margin=dict(t=20, b=20, l=10, r=10),
            )
            st.plotly_chart(fig_companies, use_container_width=True)

        st.divider()

        # ---------------- Skill Extraction ----------------
        st.markdown('<p class="section-heading">🧠 Top 10 Required Skills</p>', unsafe_allow_html=True)
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
                color_continuous_scale=["#0f3b47", "#22d3ee"],
            )
            fig_skills.update_layout(
                yaxis=dict(autorange="reversed"),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                margin=dict(t=50, b=20, l=10, r=10),
            )
            st.plotly_chart(fig_skills, use_container_width=True)

        st.divider()

        # ---------------- Skill Gap Analysis ----------------
        st.markdown('<p class="section-heading">🧩 Skill Gap Analysis</p>', unsafe_allow_html=True)

        user_skills = parse_user_skills(user_skills_raw)

        if not user_skills:
            st.info("Enter your skills above and click Analyze again to see your skill gap.")
        elif not required_skills:
            st.info("No required skills were detected for this role to compare against.")
        else:
            gap = skill_gap_analysis(required_skills, user_skills)
            match_score = calculate_match_score(required_skills, user_skills)

            st.metric("🎯 Match Score", f"{match_score}%")
            st.progress(match_score / 100)

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

            # ---------------- Learning Recommendation (rule-based) ----------------
            st.markdown('<p class="section-heading">📚 Learning Recommendation</p>', unsafe_allow_html=True)
            recommendation = generate_recommendation(gap["have"], gap["missing"], selected_title)
            st.markdown(f'<div class="recommendation-box">{recommendation}</div>', unsafe_allow_html=True)

            st.divider()

            # ---------------- AI Career Recommendation (Gemini) ----------------
            st.markdown('<p class="section-heading">🤖 AI Career Recommendation (Gemini)</p>', unsafe_allow_html=True)

            if not GEMINI_API_KEY:
                st.info(
                    "AI-powered recommendations are disabled because no `GEMINI_API_KEY` was found. "
                    "Add one to a `.env` file (see `.env.example`) to enable this feature."
                )
            else:
                with st.spinner("Asking Gemini for a personalized recommendation..."):
                    ai_text, ai_error = get_gemini_recommendation(
                        selected_title,
                        tuple(required_skills),
                        tuple(gap["missing"]),
                        summary,
                    )

                if ai_text:
                    st.markdown(f'<div class="ai-recommendation-box">{ai_text}</div>', unsafe_allow_html=True)
                elif ai_error == "import_error":
                    st.warning(
                        "The `google-genai` package is not installed. Run "
                        "`pip install -r requirements.txt` to enable AI recommendations."
                    )
                else:
                    st.warning(
                        "Couldn't generate an AI recommendation right now (network issue or API "
                        "error). The rule-based recommendation above is still available."
                    )

        st.divider()

        # ---------------- Visualizations ----------------
        st.markdown('<p class="section-heading">📈 Job Market Visualizations</p>', unsafe_allow_html=True)

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
                color_discrete_sequence=CHART_COLORWAY,
            )
            fig_worktype.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                margin=dict(t=50, b=20, l=10, r=10),
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
                color_continuous_scale=["#1e3a8a", "#60a5fa"],
            )
            fig_level.update_layout(
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                margin=dict(t=50, b=20, l=10, r=10),
            )
            st.plotly_chart(fig_level, use_container_width=True)

        st.divider()

        # ---------------- Sample Matching Jobs ----------------
        st.markdown('<p class="section-heading">🗂️ Sample Matching Jobs</p>', unsafe_allow_html=True)
        display_cols = ["title", "work_type", "level", "job_location", "compensation", "pay_period"]
        sample_jobs = filtered_df[display_cols].head(10).reset_index(drop=True)
        sample_jobs.columns = ["Title", "Work Type", "Level", "Location", "Compensation", "Pay Period"]
        st.dataframe(sample_jobs, use_container_width=True)

else:
    st.info("👈 Select a job role and enter your skills, then click **Analyze** to get started.")

st.markdown("---")

with open("assets/footer.html", "r", encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)