"""
app.py
------
AI Career Advisor - Streamlit application.

A Big Data Analytics microproject that helps students identify skill gaps
for a desired job role using a real LinkedIn job postings dataset, with an
optional Gemini-powered AI recommendation.

Frontend note: all CSS lives in assets/style.css and all static markup
lives in assets/hero.html / assets/footer.html (loaded below). Python only
supplies the small, unavoidable pieces of dynamic markup (skill badges,
metric values, card labels) as single-line strings - everything else is a
native Streamlit component styled via the external stylesheet.
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

pio.templates.default = "plotly_dark"
CHART_COLORWAY = ["#22d3ee", "#3b82f6", "#818cf8", "#06b6d4", "#60a5fa"]

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


@st.cache_data(show_spinner=False)
def get_gemini_recommendation(role: str, top_skills: tuple, missing_skills: tuple, job_stats: dict):
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
    except Exception as exc:  
        return None, str(exc)


@st.cache_data
def get_data():
    return load_data(DATA_PATH)


df = get_data()


def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def load_html(path: str):
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f.read(), unsafe_allow_html=True)


def section_heading(icon: str, text: str):
    st.markdown(f'<p class="section-heading">{icon} {text}</p>', unsafe_allow_html=True)


load_css()

# Hero
load_html("assets/hero.html")

# Dataset overview cards
total_jobs = len(df)
unique_titles = df["title"].nunique()
companies_hiring = df["company_id"].nunique()

overview_cards = [
    ("📂", "Total Jobs in Dataset", f"{total_jobs:,}"),
    ("🧭", "Unique Job Titles", f"{unique_titles:,}"),
    ("🏢", "Companies Hiring", f"{companies_hiring:,}"),
]

m1, m2, m3 = st.columns(3)
for col, (icon, label, value) in zip((m1, m2, m3), overview_cards):
    with col:
        with st.container(border=True):
            st.markdown(
                f'<div class="metric-card-inner"><div class="metric-icon">{icon}</div>'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

st.write("")

# Sidebar
with st.sidebar:
    st.markdown("### 📌 Project Information")

    with st.container(border=True):
        st.markdown("**🎯 Project**")
        st.caption("AI Career Advisor")
        st.markdown("**📚 Subject**")
        st.caption("Big Data Analytics")
        st.markdown("**🎓 Goal**")
        st.caption("Help students identify skill gaps for their desired job role using real job posting data.")

    with st.container(border=True):
        st.markdown("**🗂️ Dataset**")
        st.caption("LinkedIn job postings (`LinkedIn_RDB_three.csv`)")
        st.markdown("**🛠️ Tech Stack**")
        st.caption("Python · Streamlit · Pandas · Plotly · Google Gemini")

    st.markdown("#### 📊 Dataset Overview")
    st.metric("Total Job Postings", f"{len(df):,}")
    st.metric("Unique Job Titles", f"{df['title'].nunique():,}")

    st.markdown("#### 🤖 AI Recommendations")
    if GEMINI_API_KEY:
        st.success("Gemini AI: **Enabled**")
    else:
        st.info("Gemini AI: **Disabled**\n\nAdd `GEMINI_API_KEY` to a `.env` file to enable them.")

# Main input section
section_heading("🔎", "Find Your Skill Gap")

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

# Analysis results (Restructured visually into tabs and side-by-side components)
if analyze_clicked:
    filtered_df = filter_jobs_by_title(df, selected_title)

    if filtered_df.empty:
        st.warning(f"No job postings found for '{selected_title}'. Please try another role.")
    else:
        # Structured Tab System for streamlined information delivery
        tab_market, tab_skills, tab_ai, tab_raw = st.tabs([
            "📊 Market Insights", 
            "🎯 Skill Gap Analysis", 
            "🤖 AI Career Coach", 
            "🗂️ Sample Jobs"
        ])

        # --- TAB 1: MARKET INSIGHTS ---
        with tab_market:
            with st.container(border=True):
                section_heading("📊", "Job Market Overview")
                summary = get_job_summary(filtered_df)
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total Matching Jobs", summary["num_jobs"])
                s2.metric("Common Work Type", summary["common_work_type"])
                s3.metric("Experience Level", summary["common_level"])
                s4.metric("Common Location", summary["common_location"])

            # Split Grid: Salary Stats left, Hiring Companies right
            grid_col1, grid_col2 = st.columns(2)
            
            with grid_col1:
                with st.container(border=True):
                    section_heading("💰", "Salary Analytics")
                    salary_stats = get_salary_stats(filtered_df)
                    if salary_stats["average"] is None:
                        st.info("No salary data is available for the matching job postings.")
                    else:
                        sal1, sal2, sal3 = st.columns(3)
                        sal1.metric("Average Salary", f"${salary_stats['average']:,.0f}")
                        sal2.metric("Highest", f"${salary_stats['highest']:,.0f}")
                        sal3.metric("Lowest", f"${salary_stats['lowest']:,.0f}")
            
            with grid_col2:
                with st.container(border=True):
                    section_heading("🏢", "Top 5 Hiring Companies")
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
                            font=dict(color="#cbd5e1", family="Inter, sans-serif"),
                            margin=dict(t=10, b=10, l=10, r=10),
                        )
                        st.plotly_chart(fig_companies, use_container_width=True)

            # Split Grid: Distributions side-by-side
            vis_col1, vis_col2 = st.columns(2)
            with vis_col1:
                with st.container(border=True):
                    section_heading("📈", "Work Type Distribution")
                    work_type_counts = filtered_df["work_type"].value_counts().reset_index()
                    work_type_counts.columns = ["Work Type", "Count"]
                    fig_worktype = px.pie(
                        work_type_counts,
                        names="Work Type",
                        values="Count",
                        hole=0.55,
                        color_discrete_sequence=CHART_COLORWAY,
                    )
                    fig_worktype.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
                        margin=dict(t=10, b=10, l=10, r=10),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
                    )
                    st.plotly_chart(fig_worktype, use_container_width=True)

            with vis_col2:
                with st.container(border=True):
                    section_heading("📊", "Experience Level Distribution")
                    level_counts = filtered_df["level"].value_counts().reset_index()
                    level_counts.columns = ["Experience Level", "Count"]
                    fig_level = px.bar(
                        level_counts,
                        x="Experience Level",
                        y="Count",
                        color="Count",
                        color_continuous_scale=["#1e3a8a", "#60a5fa"],
                    )
                    fig_level.update_layout(
                        showlegend=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
                        margin=dict(t=10, b=10, l=10, r=10),
                    )
                    st.plotly_chart(fig_level, use_container_width=True)

        # --- TAB 2: SKILL GAP ANALYSIS ---
        with tab_skills:
            user_skills = parse_user_skills(user_skills_raw)
            top_skills_df = get_top_skills(filtered_df, top_n=10)
            required_skills = top_skills_df["Skill"].tolist() if not top_skills_df.empty else []

            skill_col1, skill_col2 = st.columns(2)

            with skill_col1:
                with st.container(border=True):
                    section_heading("🧩", "Your Personal Breakdown")
                    if not user_skills:
                        st.info("Enter your skills in the main inputs to see your comparison gap.")
                    elif not required_skills:
                        st.info("No required skills were detected for this role to compare against.")
                    else:
                        gap = skill_gap_analysis(required_skills, user_skills)
                        match_score = calculate_match_score(required_skills, user_skills)

                        st.metric("🎯 Match Score", f"{match_score}%")
                        st.progress(match_score / 100)
                        
                        st.write("")
                        st.markdown("**✔️ Skills You Already Have**")
                        if gap["have"]:
                            badges = "".join(f'<span class="skill-have">✔ {s}</span>' for s in gap["have"])
                            st.markdown(badges, unsafe_allow_html=True)
                        else:
                            st.caption("None of your entered skills matched the top required skills.")

                        st.write("")
                        st.markdown("**❌ Skills You Need**")
                        if gap["missing"]:
                            badges = "".join(f'<span class="skill-missing">✘ {s}</span>' for s in gap["missing"])
                            st.markdown(badges, unsafe_allow_html=True)
                        else:
                            st.caption("You already cover all the top required skills!")

            with skill_col2:
                with st.container(border=True):
                    section_heading("🧠", "Top 10 Required Skills")
                    if top_skills_df.empty:
                        st.info("No matching skill keywords were found in the job descriptions.")
                    else:
                        fig_skills = px.bar(
                            top_skills_df,
                            x="Count",
                            y="Skill",
                            orientation="h",
                            color="Count",
                            color_continuous_scale=["#0f3b47", "#22d3ee"],
                        )
                        fig_skills.update_layout(
                            yaxis=dict(autorange="reversed"),
                            showlegend=False,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#cbd5e1", family="Inter, sans-serif"),
                            margin=dict(t=10, b=10, l=10, r=10),
                        )
                        st.plotly_chart(fig_skills, use_container_width=True)

        # --- TAB 3: AI RECOMMENDATIONS ---
        with tab_ai:
            if not user_skills or not required_skills:
                st.info("Please fill in your current skills on the left panel to activate recommendations.")
            else:
                gap = skill_gap_analysis(required_skills, user_skills)
                
                with st.container(border=True):
                    section_heading("📚", "Standard Learning Roadmap")
                    recommendation = generate_recommendation(gap["have"], gap["missing"], selected_title)
                    st.markdown(f'<div class="recommendation-box">{recommendation}</div>', unsafe_allow_html=True)

                with st.container(border=True):
                    section_heading("🤖", "Personalized Gemini Insights")
                    if not GEMINI_API_KEY:
                        st.info("AI-powered recommendations are disabled because no `GEMINI_API_KEY` was found.")
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
                            st.warning("The `google-genai` package is not installed.")
                        else:
                            st.warning("Couldn't generate an AI recommendation right now.")

        # --- TAB 4: SAMPLE JOBS TABLE ---
        with tab_raw:
            with st.container(border=True):
                section_heading("🗂️", "Sample Matching Postings")
                display_cols = ["title", "work_type", "level", "job_location", "compensation", "pay_period"]
                sample_jobs = filtered_df[display_cols].head(10).reset_index(drop=True)
                sample_jobs.columns = ["Title", "Work Type", "Level", "Location", "Compensation", "Pay Period"]
                st.dataframe(sample_jobs, use_container_width=True)

else:
    st.info("👈 Select a job role and enter your skills, then click **Analyze** to get started.")

# Footer
load_html("assets/footer.html")