"""
app.py
------
AI Career Advisor - Streamlit application.

A Big Data Analytics microproject that helps students identify skill gaps
for a desired job role using a real LinkedIn job postings dataset, with an
optional Gemini-powered AI recommendation.

Frontend note: all CSS lives in assets/style.css and all static markup
lives in assets/hero.html / assets/footer.html (loaded below). Python only
supplies small, unavoidable pieces of dynamic markup (skill badges, metric
values, priority/readiness badges) as single-line strings - everything
else is a native Streamlit component styled via the external stylesheet.
"""

import os
import time

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
    assign_skill_priority,
    calculate_career_readiness,
    group_skills,
    generate_learning_roadmap,
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

# Dark plotly template so charts match the app theme (styling only - the
# underlying chart data/analysis is unchanged).
pio.templates.default = "plotly_dark"
CHART_COLORWAY = ["#22d3ee", "#3b82f6", "#818cf8", "#06b6d4", "#60a5fa"]


# =========================================================================
# GEMINI AI CONFIGURATION
# -------------------------------------------------------------------------
# Reads the API key from a local .env file (never hardcoded). Network
# calls are wrapped with retry logic for transient 503s and clear error
# classification for quota / auth / connectivity failures, so the app
# always degrades gracefully to the rule-based recommendation instead of
# crashing or showing a raw stack trace.
# =========================================================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

GEMINI_MAX_RETRIES = 2          # extra attempts after the first, for transient errors only
GEMINI_RETRY_BACKOFF_SECONDS = 2  # multiplied by attempt number

# User-facing messages for each error category the Gemini call can hit.
GEMINI_ERROR_MESSAGES = {
    "no_key": "AI recommendations are disabled because no `GEMINI_API_KEY` was found.",
    "import_error": "The `google-genai` package is not installed. Run `pip install -r requirements.txt`.",
    "server_unavailable": "Gemini's servers are temporarily overloaded (503). Showing the rule-based plan instead.",
    "quota_exceeded": "The Gemini API quota/rate limit was reached. Showing the rule-based plan instead.",
    "invalid_key": "The Gemini API key appears to be invalid. Check `GEMINI_API_KEY` in your `.env` file.",
    "network_error": "Couldn't reach Gemini (network issue). Showing the rule-based plan instead.",
    "empty_response": "Gemini returned an empty response. Showing the rule-based plan instead.",
}


def _classify_gemini_error(exc: Exception) -> str:
    """Map a raw Gemini/network exception to a short, user-safe category key."""
    message = str(exc).lower()
    if "503" in message or "unavailable" in message or "overloaded" in message:
        return "server_unavailable"
    if "429" in message or "quota" in message or "rate limit" in message or "resource_exhausted" in message:
        return "quota_exceeded"
    if "api key" in message or "401" in message or "403" in message or "permission" in message or "unauthenticated" in message:
        return "invalid_key"
    if "timeout" in message or "connection" in message or "network" in message or "dns" in message:
        return "network_error"
    return str(exc)  # unknown - surface the raw (short) message for debugging


def _build_gemini_prompt(role, top_skills, missing_skills, job_stats, readiness, grouped_missing):
    """
    Build the Gemini prompt. Gemini is asked to act as a senior career
    mentor / hiring manager / industry expert and return a fully
    structured, practical recommendation (not a generic paragraph).
    """
    grouped_text = "\n".join(
        f"- {group}: {', '.join(skills)}" for group, skills in grouped_missing.items()
    ) or "None - the student already covers the top required skills."

    return f"""
You are acting as three experts at once for this student:
1. A Senior Career Mentor who gives honest, motivating guidance.
2. A Senior Hiring Manager who knows exactly what makes a candidate stand out.
3. An Industry Expert who tracks where this field is heading.

Analyze this student's situation and respond as those experts would - concrete
and practical, never generic filler.

ROLE THE STUDENT WANTS: {role}

TOP SKILLS REQUIRED BY REAL JOB POSTINGS: {', '.join(top_skills) if top_skills else 'Not enough data'}

SKILLS THE STUDENT IS MISSING, GROUPED BY AREA:
{grouped_text}

CALCULATED CAREER READINESS: {readiness['score']}% - {readiness['level']} ({readiness['explanation']})

JOB MARKET STATISTICS FOR THIS ROLE: {job_stats}

Respond in Markdown using EXACTLY these section headers, in this order:

### Career Summary
### Current Strengths
### Missing Skills
### Why These Skills Matter
### Career Readiness Score
### Learning Roadmap
Structure this part as Week 1, Week 2, Week 3, Week 4, then Projects, then Interview Prep - not a paragraph.
### Projects to Build
### Recommended Certifications
### Interview Preparation
### Resume Advice
### Estimated Learning Time
### Future Industry Trends
### Final Advice

Keep every section short and practical (2-5 bullet points or sentences each).
Avoid generic AI phrases like "In today's competitive job market". Be specific
to this role and this student's actual skill gap.
"""


@st.cache_data(show_spinner=False)
def get_gemini_recommendation(role: str, top_skills: tuple, missing_skills: tuple,
                               job_stats: dict, readiness: dict, grouped_missing: dict):
    """
    Ask Gemini for a structured career recommendation.

    Returns (text, error). `text` is None if the recommendation could not
    be generated, in which case `error` is one of the keys in
    GEMINI_ERROR_MESSAGES (or a raw short message for truly unexpected
    errors). Transient server-unavailable (503) errors are retried with
    a short backoff before giving up.
    """
    if not GEMINI_API_KEY:
        return None, "no_key"

    try:
        from google import genai
    except ImportError:
        return None, "import_error"

    prompt = _build_gemini_prompt(role, top_skills, missing_skills, job_stats, readiness, grouped_missing)

    last_error = "unknown_error"
    for attempt in range(GEMINI_MAX_RETRIES + 1):
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            if response and getattr(response, "text", None):
                return response.text, None
            last_error = "empty_response"
            break
        except Exception as exc:  # noqa: BLE001 - any failure should degrade gracefully
            category = _classify_gemini_error(exc)
            last_error = category
            if category == "server_unavailable" and attempt < GEMINI_MAX_RETRIES:
                time.sleep(GEMINI_RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            break

    return None, last_error


# -----------------------------------------------------------------------
# Cache the dataset so it only loads once
# -----------------------------------------------------------------------
@st.cache_data
def get_data():
    return load_data(DATA_PATH)


df = get_data()


# -----------------------------------------------------------------------
# Frontend asset loaders - all CSS/HTML lives in assets/, not in Python
# -----------------------------------------------------------------------
def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def load_html(path: str):
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f.read(), unsafe_allow_html=True)


def section_heading(icon: str, text: str):
    """Single-line HTML card title - safe for Streamlit's markdown parser."""
    st.markdown(f'<p class="section-heading">{icon} {text}</p>', unsafe_allow_html=True)


def render_skill_badges(skills: list, css_class: str, symbol: str):
    """Render a list of skills as single-line HTML pill badges."""
    badges = "".join(f'<span class="{css_class}">{symbol} {s}</span>' for s in skills)
    st.markdown(badges, unsafe_allow_html=True)


def render_grouped_skills(grouped: dict, css_class: str, symbol: str):
    """Render skills grouped by category (Frontend/Backend/...), each with a header."""
    for group, skills in grouped.items():
        st.markdown(f'<div class="group-header">{group}</div>', unsafe_allow_html=True)
        render_skill_badges(skills, css_class, symbol)


def render_priority_badges(prioritized_df):
    """Render each required skill as a badge colored by its priority."""
    priority_class = {"High": "priority-high", "Medium": "priority-medium", "Low": "priority-low"}
    badges = "".join(
        f'<span class="priority-badge {priority_class.get(row.Priority, "priority-low")}">'
        f'{row.Skill} &middot; {row.Priority}</span>'
        for row in prioritized_df.itertuples()
    )
    st.markdown(badges, unsafe_allow_html=True)


def render_readiness_badge(readiness: dict):
    """Render the career readiness level as a colored badge with its score and explanation."""
    level_class = {
        "Beginner": "readiness-beginner",
        "Intermediate": "readiness-intermediate",
        "Advanced": "readiness-advanced",
        "Job Ready": "readiness-job-ready",
    }.get(readiness["level"], "readiness-beginner")
    st.markdown(
        f'<span class="readiness-badge {level_class}">{readiness["level"]} &middot; {readiness["score"]}%</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<p class="readiness-explanation">{readiness["explanation"]}</p>', unsafe_allow_html=True)


def render_roadmap(steps: list):
    """Render a list of (label, content) tuples as a simple timeline."""
    for label, content in steps:
        st.markdown(
            f'<div class="roadmap-step"><div class="roadmap-label">{label}</div>'
            f'<div class="roadmap-content">{content}</div></div>',
            unsafe_allow_html=True,
        )


load_css()

# -----------------------------------------------------------------------
# Hero
# -----------------------------------------------------------------------
load_html("assets/hero.html")

# -----------------------------------------------------------------------
# Dataset overview - premium metric cards (native bordered containers,
# single-line HTML only for the icon/label/value inside)
# -----------------------------------------------------------------------
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

# -----------------------------------------------------------------------
# Sidebar - Project Information
# -----------------------------------------------------------------------
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

# -----------------------------------------------------------------------
# Main input section
# -----------------------------------------------------------------------
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
        tab_market, tab_skills, tab_ai, tab_raw = st.tabs([
            "📊 Market Insights",
            "🎯 Skill Gap Analysis",
            "🤖 AI Career Coach",
            "🗂️ Sample Jobs",
        ])

        # ================= TAB 1: MARKET INSIGHTS =================
        with tab_market:
            with st.container(border=True):
                section_heading("📊", "Job Market Overview")
                summary = get_job_summary(filtered_df)
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total Matching Jobs", summary["num_jobs"])
                s2.metric("Common Work Type", summary["common_work_type"])
                s3.metric("Experience Level", summary["common_level"])
                s4.metric("Common Location", summary["common_location"])

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

        # ================= TAB 2: SKILL GAP ANALYSIS =================
        with tab_skills:
            user_skills = parse_user_skills(user_skills_raw)
            top_skills_df = get_top_skills(filtered_df, top_n=10)
            required_skills = top_skills_df["Skill"].tolist() if not top_skills_df.empty else []
            prioritized_skills_df = assign_skill_priority(top_skills_df)
            readiness = calculate_career_readiness(top_skills_df, user_skills)

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

                        st.markdown("**🚦 Career Readiness**")
                        render_readiness_badge(readiness)

                        st.metric("🎯 Match Score", f"{match_score}%")
                        st.progress(match_score / 100)

                        st.write("")
                        st.markdown("**✔️ Skills You Already Have**")
                        if gap["have"]:
                            render_skill_badges(gap["have"], "skill-have", "✔")
                        else:
                            st.caption("None of your entered skills matched the top required skills.")

                        st.write("")
                        st.markdown("**❌ Skills You Need**")
                        if gap["missing"]:
                            grouped_missing = group_skills(gap["missing"])
                            render_grouped_skills(grouped_missing, "skill-missing", "✘")
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

                        st.markdown("**⭐ Priority**")
                        st.caption("How often each skill appears across matching postings.")
                        render_priority_badges(prioritized_skills_df)

        # ================= TAB 3: AI RECOMMENDATIONS =================
        with tab_ai:
            if not user_skills or not required_skills:
                st.info("Please fill in your current skills on the left panel to activate recommendations.")
            else:
                gap = skill_gap_analysis(required_skills, user_skills)
                grouped_missing = group_skills(gap["missing"])

                with st.container(border=True):
                    section_heading("🤖", "Personalized Gemini Insights")

                    if not GEMINI_API_KEY:
                        st.info(GEMINI_ERROR_MESSAGES["no_key"])
                        ai_text, ai_error = None, "no_key"
                    else:
                        with st.spinner("Asking Gemini for a personalized recommendation..."):
                            ai_text, ai_error = get_gemini_recommendation(
                                selected_title,
                                tuple(required_skills),
                                tuple(gap["missing"]),
                                summary,
                                readiness,
                                grouped_missing,
                            )

                    if ai_text:
                        st.markdown(f'<div class="ai-recommendation-box">{ai_text}</div>', unsafe_allow_html=True)
                    elif ai_error and ai_error != "no_key":
                        friendly_message = GEMINI_ERROR_MESSAGES.get(
                            ai_error, "Couldn't generate an AI recommendation right now. Showing the rule-based plan instead."
                        )
                        if ai_error == "invalid_key":
                            st.error(friendly_message)
                        else:
                            st.warning(friendly_message)

                # Rule-based recommendation + structured roadmap - always shown as a
                # reliable fallback, so the user always gets a usable recommendation
                # even when Gemini is disabled or fails.
                with st.container(border=True):
                    section_heading("📚", "Standard Learning Roadmap")
                    st.caption("A rule-based recommendation, always available even if Gemini is unreachable.")
                    recommendation = generate_recommendation(gap["have"], gap["missing"], selected_title)
                    st.markdown(f'<div class="recommendation-box">{recommendation}</div>', unsafe_allow_html=True)

                    st.write("")
                    roadmap_steps = generate_learning_roadmap(gap["missing"])
                    render_roadmap(roadmap_steps)

        # ================= TAB 4: SAMPLE JOBS TABLE =================
        with tab_raw:
            with st.container(border=True):
                section_heading("🗂️", "Sample Matching Postings")
                display_cols = ["title", "work_type", "level", "job_location", "compensation", "pay_period"]
                sample_jobs = filtered_df[display_cols].head(10).reset_index(drop=True)
                sample_jobs.columns = ["Title", "Work Type", "Level", "Location", "Compensation", "Pay Period"]
                st.dataframe(sample_jobs, use_container_width=True)

else:
    st.info("👈 Select a job role and enter your skills, then click **Analyze** to get started.")

# -----------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------
load_html("assets/footer.html")