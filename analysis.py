"""
analysis.py
------------
Reusable data-analysis functions for the AI Career Advisor app.

The app reads linkedin_jobs_cleaned.csv, produced by
Data_Cleaning_and_NLP.ipynb from the original LinkedIn_RDB_three.csv
export. It has the same columns as the raw file, plus two extra ones
added by that notebook: `cleaned_description` and `extracted_skills`.

job_id, title, description, pay_period, work_type, job_location,
applies, remote_allowed, views, level, sponsored, compensation,
job_domain, company_id, ben_pack_id, cleaned_description, extracted_skills

Note on `extracted_skills`: the notebook's own extraction uses a small
demo list of ~24 skills. This app intentionally keeps using its own
richer, on-the-fly extraction (SKILL_KEYWORDS below, 100+ terms across
10 categories) against the `description` column instead, since it
detects far more skills than the notebook's list would. The
`cleaned_description` / `extracted_skills` columns are left in the
DataFrame but unused, so nothing is lost if a future iteration wants to
build on them.
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
# Skill normalization
# ---------------------------------------------------------------------
# Maps common variant spellings to the single canonical name used in
# SKILL_KEYWORDS, so "ReactJS", "React.js", and "react" are all treated
# as the same skill when comparing user input against required skills.
SKILL_ALIASES = {
    "reactjs": "React", "react.js": "React", "react js": "React",
    "nodejs": "Node.js", "node": "Node.js", "node.js": "Node.js", "node js": "Node.js",
    "js": "JavaScript", "javascript": "JavaScript",
    "ts": "TypeScript", "typescript": "TypeScript",
    "vuejs": "Vue", "vue.js": "Vue", "vue js": "Vue",
    "nextjs": "Next.js", "next": "Next.js", "next.js": "Next.js",
    "postgres": "PostgreSQL", "postgresql": "PostgreSQL",
    "mongo": "MongoDB", "mongodb": "MongoDB",
    "k8s": "Kubernetes", "kubernetes": "Kubernetes",
    "csharp": "C#", "c sharp": "C#", "c#": "C#",
    "golang": "Golang", "go": "Golang",
    "sklearn": "Scikit-learn", "scikit learn": "Scikit-learn", "scikit-learn": "Scikit-learn",
    "cicd": "CI/CD", "ci cd": "CI/CD", "ci/cd": "CI/CD",
    "restapi": "REST API", "rest api": "REST API", "rest": "REST API",
    "gcp": "GCP", "google cloud": "GCP", "google cloud platform": "GCP",
    "aws": "AWS", "amazon web services": "AWS",
    "powerbi": "Power BI", "power bi": "Power BI",
    "tailwindcss": "Tailwind CSS", "tailwind": "Tailwind CSS",
    "dotnet": ".NET", ".net": ".NET",
}


def normalize_skill(skill: str) -> str:
    """
    Normalize a skill string to a canonical form so that variants like
    'ReactJS', 'React.js', and 'react' all compare equal. Falls back to
    the original (trimmed) text when no canonical match is found, so
    unrecognized skills still compare consistently against themselves.
    """
    if not skill or not isinstance(skill, str):
        return ""

    raw = skill.strip().lower()
    compact = re.sub(r"[.\-_]", "", raw)

    if raw in SKILL_ALIASES:
        return SKILL_ALIASES[raw]
    if compact in SKILL_ALIASES:
        return SKILL_ALIASES[compact]

    # Fall back to a case-insensitive match against the known keyword list
    for canonical in SKILL_KEYWORDS:
        canonical_lower = canonical.lower()
        if canonical_lower == raw or re.sub(r"[.\-_]", "", canonical_lower) == compact:
            return canonical

    return skill.strip()


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """
    Load the LinkedIn job postings dataset from a CSV file.

    Works with either the original raw CSV or the cleaned CSV produced by
    Data_Cleaning_and_NLP.ipynb (linkedin_jobs_cleaned.csv). The cleaned
    file already has whitespace trimmed and missing values filled in, so
    this mostly acts as a safety net for whichever file is passed in.
    """
    df = pd.read_csv(path)

    # Text/categorical columns: make sure they're clean strings with no
    # NaNs. `compensation` is deliberately excluded - it's a numeric
    # amount, not a category, and forcing it to text just to parse it
    # back to a number later was unnecessary complexity (see
    # _parse_compensation_value, which now reads numbers directly).
    text_columns = ["title", "description", "work_type", "job_location",
                     "level", "job_domain", "pay_period"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Not Specified").astype(str).str.strip()
            # Normalize case so a differently-cased placeholder from either
            # dataset (e.g. "not specified") still matches downstream
            # filtering in _most_common().
            df.loc[df[col].str.lower() == "not specified", col] = "Not Specified"

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
def _parse_compensation_value(value) -> float:
    """
    Convert a compensation value into a plain float, regardless of which
    dataset it came from:
    - The cleaned dataset (linkedin_jobs_cleaned.csv) already stores
      compensation as a numeric column (e.g. 42000.0).
    - The original raw dataset stores it as a string with symbols
      (e.g. '$42,000.00').
    Returns None for missing/unparseable values so callers can drop them.
    """
    if value is None:
        return None

    # Already numeric (the normal case for the cleaned dataset) - just
    # guard against NaN, which is a float but not a valid amount.
    if isinstance(value, (int, float)):
        return None if pd.isna(value) else float(value)

    if isinstance(value, str):
        cleaned = re.sub(r"[^\d.]", "", value)
        if cleaned == "":
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None

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
    """
    Return the most frequent non-empty, non-placeholder value in a
    pandas Series. The "not specified" check is case-insensitive so it
    correctly excludes the placeholder regardless of which dataset it
    came from (the cleaned CSV uses "Not Specified"; older data may use
    other casing).
    """
    non_null = series[series.notna()]
    is_placeholder = non_null.astype(str).str.strip().str.lower().isin(["not specified", ""])
    cleaned = non_null[~is_placeholder]

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
    entered skills. Matching is case-insensitive AND normalized, so
    variants like "ReactJS" / "React.js" / "react" are recognized as the
    same skill as "React" (see normalize_skill / SKILL_ALIASES).
    Returns a dict with 'have' and 'missing' skill lists (using the
    original, human-readable required-skill labels).
    """
    normalized_user = {normalize_skill(skill).lower() for skill in user_skills}

    have = [skill for skill in required_skills if normalize_skill(skill).lower() in normalized_user]
    missing = [skill for skill in required_skills if normalize_skill(skill).lower() not in normalized_user]

    return {"have": have, "missing": missing}


# ---------------------------------------------------------------------
# Match score
# ---------------------------------------------------------------------
def calculate_match_score(required_skills: list, user_skills: list) -> int:
    """
    Return a 0-100 match score representing what percentage of the
    required skills the user already has. Uses normalized matching
    (see normalize_skill) so skill variants are recognized correctly.
    Returns 0 if there are no required skills to compare against.
    """
    if not required_skills:
        return 0

    normalized_user = {normalize_skill(skill).lower() for skill in user_skills}
    matched = sum(1 for skill in required_skills if normalize_skill(skill).lower() in normalized_user)

    score = (matched / len(required_skills)) * 100
    return round(score)


# ---------------------------------------------------------------------
# Skill grouping (Frontend / Backend / Database / Cloud / Tools / Soft Skills)
# ---------------------------------------------------------------------
# Consolidates the finer-grained SKILL_CATEGORIES into the broader
# buckets used for the skill-gap breakdown view.
_CATEGORY_TO_GROUP = {
    "Frontend": "Frontend",
    "Backend": "Backend",
    "Databases": "Database",
    "Cloud": "Cloud",
    "DevOps": "Tools",
    "Data Analytics": "Tools",
    "AI": "Tools",
    "Computer Science Fundamentals": "Tools",
    "Soft Skills": "Soft Skills",
}
# Programming languages are split by how they're typically used.
_FRONTEND_LANGUAGES = {"JavaScript", "TypeScript"}

SKILL_GROUPS = ["Frontend", "Backend", "Database", "Cloud", "Tools", "Soft Skills"]


def _skill_to_group(skill: str) -> str:
    """Map a single skill name to one of the SKILL_GROUPS buckets."""
    for category, skills in SKILL_CATEGORIES.items():
        if skill in skills:
            if category == "Programming Languages":
                return "Frontend" if skill in _FRONTEND_LANGUAGES else "Backend"
            return _CATEGORY_TO_GROUP.get(category, "Tools")
    return "Tools"


def group_skills(skills: list) -> dict:
    """
    Group a flat list of skill names into Frontend / Backend / Database /
    Cloud / Tools / Soft Skills buckets. Only non-empty buckets are
    returned, in a fixed, sensible display order.
    """
    buckets = {group: [] for group in SKILL_GROUPS}
    for skill in skills:
        buckets[_skill_to_group(skill)].append(skill)
    return {group: items for group, items in buckets.items() if items}


# ---------------------------------------------------------------------
# Skill priority (High / Medium / Low)
# ---------------------------------------------------------------------
def assign_skill_priority(top_skills_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'Priority' column (High / Medium / Low) to a top-skills
    DataFrame, based on how frequently each skill appears relative to
    the most-requested skill for the role. Returns a new DataFrame -
    the input is not modified.
    """
    result = top_skills_df.copy()
    if result.empty:
        result["Priority"] = pd.Series(dtype="object")
        return result

    max_count = result["Count"].max()

    def _priority(count):
        ratio = (count / max_count) if max_count else 0
        if ratio >= 0.66:
            return "High"
        if ratio >= 0.33:
            return "Medium"
        return "Low"

    result["Priority"] = result["Count"].apply(_priority)
    return result


# ---------------------------------------------------------------------
# Career readiness score
# ---------------------------------------------------------------------
_PRIORITY_WEIGHTS = {"High": 3, "Medium": 2, "Low": 1}


def calculate_career_readiness(top_skills_df: pd.DataFrame, user_skills: list) -> dict:
    """
    Calculate a weighted career-readiness score that gives more credit
    for matching high-priority (frequently required) skills than
    low-priority ones, and maps the score to a readiness level with a
    short explanation.

    Returns a dict: {"score": int 0-100, "level": str, "explanation": str}
    """
    if top_skills_df.empty:
        return {
            "score": 0,
            "level": "Beginner",
            "explanation": "Not enough job data was found for this role to assess readiness.",
        }

    prioritized = assign_skill_priority(top_skills_df)
    normalized_user = {normalize_skill(skill).lower() for skill in user_skills}

    total_weight = 0
    matched_weight = 0
    for _, row in prioritized.iterrows():
        weight = _PRIORITY_WEIGHTS.get(row["Priority"], 1)
        total_weight += weight
        if normalize_skill(row["Skill"]).lower() in normalized_user:
            matched_weight += weight

    score = round((matched_weight / total_weight) * 100) if total_weight else 0

    if score >= 85:
        level = "Job Ready"
        explanation = "You already cover most of the high-priority skills employers are asking for in this role."
    elif score >= 65:
        level = "Advanced"
        explanation = "You have a strong foundation, with a few high-priority skills left to pick up."
    elif score >= 40:
        level = "Intermediate"
        explanation = "You have some relevant skills, but several high-priority requirements are still missing."
    else:
        level = "Beginner"
        explanation = "Most of the key skills employers want for this role are not yet in your profile."

    return {"score": score, "level": level, "explanation": explanation}


# ---------------------------------------------------------------------
# Structured fallback learning roadmap
# ---------------------------------------------------------------------
def generate_learning_roadmap(missing_skills: list) -> list:
    """
    Build a simple, structured multi-week learning roadmap from a list of
    missing skills. Used as a fallback when the AI (Gemini) recommendation
    is unavailable, so the user still gets a structured plan instead of a
    single paragraph. Returns a list of (label, content) tuples.
    """
    if not missing_skills:
        return [(
            "Overview",
            "You already cover the top required skills for this role - focus on "
            "building projects that demonstrate them well.",
        )]

    weeks = [missing_skills[i:i + 2] for i in range(0, len(missing_skills), 2)][:4]
    roadmap = []
    for i, chunk in enumerate(weeks, start=1):
        roadmap.append((f"Week {i}", f"Focus on: {', '.join(chunk)}."))

    covered = sum(len(chunk) for chunk in weeks)
    remaining = missing_skills[covered:]
    if remaining:
        roadmap.append(("Ongoing", f"Continue building familiarity with: {', '.join(remaining)}."))

    roadmap.append((
        "Projects",
        "Build 1-2 small projects that combine your existing skills with the "
        "new ones above, and add them to your portfolio.",
    ))
    roadmap.append((
        "Interview Prep",
        "Review common interview questions for this role and practice explaining "
        "your projects and skill choices out loud.",
    ))
    return roadmap


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