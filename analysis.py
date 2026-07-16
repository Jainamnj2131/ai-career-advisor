"""
analysis.py
------------
Reusable data-analysis functions for the AI Career Advisor app.

The dataset (LinkedIn_RDB_three.csv) has the following columns:
job_id, title, description, pay_period, work_type, job_location,
applies, remote_allowed, views, level, sponsored, compensation,
job_domain, company_id, ben_pack_id

There is NO dedicated "skills" column, so skills are extracted from the
`description` column using a predefined keyword list.
"""

import re
import pandas as pd

# ---------------------------------------------------------------------
# Predefined skill keywords to search for inside job descriptions,
# organized by category. SKILL_KEYWORDS (the flat list used by the
# extraction functions) is built automatically from these categories so
# the function signatures below never need to change.
# ---------------------------------------------------------------------
SKILL_CATEGORIES = {
    "Programming Languages": [
        "Python", "Java", "C++", "C#", "JavaScript", "TypeScript", "Golang",
        "Rust", "Kotlin", "Swift", "PHP", "Ruby", "Scala", "MATLAB",
        "Perl", "Dart",
    ],
    "Frontend": [
        "React", "Angular", "Vue", "Next.js", "Redux", "HTML", "CSS",
        "Sass", "Tailwind CSS", "Bootstrap", "jQuery", "Webpack",
    ],
    "Backend": [
        "Node.js", "Express", "Spring Boot", "Django", "Flask", "FastAPI",
        ".NET", "Laravel", "Ruby on Rails", "REST API", "GraphQL",
        "Microservices", "gRPC",
    ],
    "Databases": [
        "SQL", "MySQL", "PostgreSQL", "Oracle", "MongoDB", "Redis",
        "Cassandra", "DynamoDB", "SQLite", "MariaDB", "Elasticsearch",
        "Firebase",
    ],
    "Cloud": [
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform",
        "CloudFormation", "Serverless", "Lambda",
    ],
    "DevOps": [
        "Git", "CI/CD", "Jenkins", "GitHub Actions", "GitLab CI",
        "Ansible", "Linux", "Bash", "Prometheus", "Grafana", "Nginx",
    ],
    "Data Analytics": [
        "Power BI", "Tableau", "Excel", "Pandas", "NumPy", "SQL Server",
        "ETL", "Data Warehousing", "Spark", "Hadoop", "Looker",
        "Data Visualization", "A/B Testing",
    ],
    "AI": [
        "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
        "Scikit-learn", "NLP", "Computer Vision", "LLM", "Generative AI",
        "MLOps", "Keras", "OpenCV",
    ],
    "Soft Skills": [
        "Communication", "Problem Solving", "Teamwork", "Leadership",
        "Time Management", "Critical Thinking", "Collaboration",
        "Adaptability", "Project Management", "Agile", "Scrum",
    ],
    "Computer Science Fundamentals": [
        "Data Structures", "Algorithms", "Statistics", "OOP",
        "System Design", "Distributed Systems",
    ],
}

# Flat list used by the extraction functions (order preserved,
# duplicates removed just in case).
SKILL_KEYWORDS = list(dict.fromkeys(
    skill for category_skills in SKILL_CATEGORIES.values() for skill in category_skills
))


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load the LinkedIn job postings dataset from a CSV file."""
    df = pd.read_csv(path)

    # Basic cleanup - make sure text columns are strings and free of NaN issues
    text_columns = ["title", "description", "work_type", "job_location",
                     "level", "job_domain", "compensation", "pay_period"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Not specified").astype(str).str.strip()

    return df


# ---------------------------------------------------------------------
# Job role list
# ---------------------------------------------------------------------
def get_job_titles(df: pd.DataFrame) -> list:
    """Return a sorted list of unique job titles for the dropdown."""
    titles = df["title"].dropna().unique().tolist()
    return sorted(titles)


# ---------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------
def filter_jobs_by_title(df: pd.DataFrame, selected_title: str) -> pd.DataFrame:
    """
    Return all job postings whose title contains the selected job title text.

    Matching is case-insensitive and partial ("contains"), so selecting
    "Full Stack Developer" also matches "Senior Full Stack Developer",
    "Lead Full Stack Engineer" (via shared words), ".NET Full Stack
    Developer", etc. - not just an exact title match.
    """
    if not selected_title or not selected_title.strip():
        return df.iloc[0:0].copy()

    search_term = selected_title.strip().lower()
    return df[df["title"].str.lower().str.contains(re.escape(search_term), na=False)].copy()


# ---------------------------------------------------------------------
# Salary handling
# ---------------------------------------------------------------------
def _parse_compensation_value(value: str) -> float:
    """Convert a compensation string like '$42,000.00' into a float."""
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"[^\d.]", "", value)
    if cleaned == "":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def get_average_salary(filtered_df: pd.DataFrame):
    """
    Return the average compensation for the filtered jobs, or None.

    NOTE: In this dataset the `compensation` column does not behave like a
    true rate tied to `pay_period` - rows labelled "HOURLY" still contain
    values in the tens/hundreds of thousands (i.e. already a salary-scale
    figure, not a per-hour rate). Multiplying those by working hours/year
    would produce meaningless multi-million-dollar averages. So the
    `compensation` value is used as-is, exactly as it appears in the data.
    """
    if filtered_df.empty:
        return None

    values = filtered_df["compensation"].apply(_parse_compensation_value)
    values = values.dropna()

    if values.empty:
        return None

    return values.mean()


def get_salary_stats(filtered_df: pd.DataFrame) -> dict:
    """
    Return a dict with 'average', 'highest', and 'lowest' compensation
    for the filtered jobs. Any value is None if salary data is unavailable,
    so the caller can handle the "no salary data" case gracefully.
    """
    if filtered_df.empty:
        return {"average": None, "highest": None, "lowest": None}

    values = filtered_df["compensation"].apply(_parse_compensation_value).dropna()

    if values.empty:
        return {"average": None, "highest": None, "lowest": None}

    return {
        "average": values.mean(),
        "highest": values.max(),
        "lowest": values.min(),
    }


# ---------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------
def _most_common(series: pd.Series):
    """Return the most frequent non-empty value in a pandas Series."""
    cleaned = series[(series.notna()) & (series != "Not specified") & (series != "")]
    if cleaned.empty:
        return "Not available"
    return cleaned.mode().iloc[0]


def get_job_summary(filtered_df: pd.DataFrame) -> dict:
    """Return summary statistics for the filtered job postings."""
    summary = {
        "num_jobs": len(filtered_df),
        "avg_salary": get_average_salary(filtered_df),
        "common_work_type": _most_common(filtered_df["work_type"]) if not filtered_df.empty else "Not available",
        "common_level": _most_common(filtered_df["level"]) if not filtered_df.empty else "Not available",
        "common_location": _most_common(filtered_df["job_location"]) if not filtered_df.empty else "Not available",
    }
    return summary


# ---------------------------------------------------------------------
# Top hiring companies
# ---------------------------------------------------------------------
def get_top_hiring_companies(filtered_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Return the top N companies posting the most jobs in the filtered
    dataset. The dataset does not include a company name column, only
    `company_id`, so companies are identified by that ID.
    Returns a DataFrame with columns ['Company ID', 'Job Postings'].
    """
    if filtered_df.empty or "company_id" not in filtered_df.columns:
        return pd.DataFrame(columns=["Company ID", "Job Postings"])

    counts = (
        filtered_df["company_id"]
        .dropna()
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    counts.columns = ["Company ID", "Job Postings"]
    # Company IDs are numeric identifiers - display them as clean integers/strings
    counts["Company ID"] = counts["Company ID"].apply(
        lambda x: str(int(x)) if isinstance(x, (int, float)) and not pd.isna(x) else str(x)
    )
    return counts


# ---------------------------------------------------------------------
# Skill extraction
# ---------------------------------------------------------------------
def extract_skills_from_text(text: str, keyword_list: list = SKILL_KEYWORDS) -> list:
    """Return the list of keywords found inside a single description."""
    if not isinstance(text, str) or text.strip() == "":
        return []

    text_lower = text.lower()
    found = []
    for keyword in keyword_list:
        # Use word-boundary style matching so "R" doesn't match inside other words, etc.
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(keyword.lower()) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            found.append(keyword)
    return found


def get_top_skills(filtered_df: pd.DataFrame, keyword_list: list = SKILL_KEYWORDS, top_n: int = 10) -> pd.DataFrame:
    """
    Scan all job descriptions in the filtered dataframe and count how often
    each skill keyword appears. Returns a DataFrame with columns
    ['Skill', 'Count'] sorted by frequency, limited to top_n rows.
    """
    if filtered_df.empty:
        return pd.DataFrame(columns=["Skill", "Count"])

    skill_counts = {keyword: 0 for keyword in keyword_list}

    for description in filtered_df["description"]:
        found_skills = extract_skills_from_text(description, keyword_list)
        for skill in found_skills:
            skill_counts[skill] += 1

    skill_df = pd.DataFrame(list(skill_counts.items()), columns=["Skill", "Count"])
    skill_df = skill_df[skill_df["Count"] > 0]
    skill_df = skill_df.sort_values(by="Count", ascending=False).head(top_n).reset_index(drop=True)
    return skill_df


# ---------------------------------------------------------------------
# User skills parsing
# ---------------------------------------------------------------------
def parse_user_skills(raw_text: str) -> list:
    """Convert a comma-separated skills string into a clean list of skills."""
    if not raw_text or not raw_text.strip():
        return []
    skills = [skill.strip() for skill in raw_text.split(",")]
    skills = [skill for skill in skills if skill != ""]
    return skills


# ---------------------------------------------------------------------
# Skill gap analysis
# ---------------------------------------------------------------------
def skill_gap_analysis(required_skills: list, user_skills: list) -> dict:
    """
    Compare the required skills (from job postings) against the user's
    entered skills. Matching is case-insensitive.
    Returns a dict with 'have' and 'missing' skill lists.
    """
    user_skills_lower = {skill.lower() for skill in user_skills}

    have = [skill for skill in required_skills if skill.lower() in user_skills_lower]
    missing = [skill for skill in required_skills if skill.lower() not in user_skills_lower]

    return {"have": have, "missing": missing}


# ---------------------------------------------------------------------
# Match score
# ---------------------------------------------------------------------
def calculate_match_score(required_skills: list, user_skills: list) -> int:
    """
    Return a 0-100 match score representing what percentage of the
    required skills the user already has.
    Returns 0 if there are no required skills to compare against.
    """
    if not required_skills:
        return 0

    user_skills_lower = {skill.lower() for skill in user_skills}
    matched = sum(1 for skill in required_skills if skill.lower() in user_skills_lower)

    score = (matched / len(required_skills)) * 100
    return round(score)


# ---------------------------------------------------------------------
# Learning recommendation
# ---------------------------------------------------------------------
def generate_recommendation(have: list, missing: list, job_title: str) -> str:
    """Generate a simple, dynamic learning recommendation sentence."""
    if not have and not missing:
        return (f"No common skill keywords were found for '{job_title}' in the analyzed "
                f"job descriptions. Try selecting a different role or check back with a "
                f"more detailed skill list.")

    if not missing:
        return (f"Great news! Based on the analyzed job postings for '{job_title}', you "
                f"already possess all the top required skills. Keep sharpening them and "
                f"consider exploring advanced topics in your field.")

    have_text = ", ".join(have) if have else "no matching skills yet"
    missing_text = ", ".join(missing)

    return (f"Based on the analyzed job postings for '{job_title}', you already possess "
            f"{have_text}. To improve your chances of getting this role, focus on learning "
            f"{missing_text}.")