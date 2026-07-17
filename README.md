# AI Career Advisor

**Subject:** Big Data Analytics — College Microproject
**Project Title:** AI Career Advisor for Job Market Trend Analysis and Skill Demand Prediction Using Big Data Analytics

## Project Overview

AI Career Advisor is a Streamlit web app that helps students identify skill
gaps for a desired job role. It analyzes a real-world LinkedIn job postings
dataset to find the most in-demand skills for a chosen role, compares them
against the skills the user already has, and optionally asks Google Gemini
for a personalized career recommendation.

The app runs on `data/linkedin_jobs_cleaned.csv` - a cleaned version of the
original `LinkedIn_RDB_three.csv` export, produced by the
`Data_Cleaning_and_NLP.ipynb` notebook included in this repo (whitespace
trimmed, duplicates removed, missing values filled in, and `compensation`
converted from a `"$42,000.00"`-style string to a proper number). See
[Data Cleaning & NLP Pipeline](#data-cleaning--nlp-pipeline) below for details.

The dataset does **not** contain a dedicated "skills" column, so skills are
extracted directly from each job's `description` text using a large,
categorized keyword list (Programming Languages, Frameworks, Cloud,
Databases, Soft Skills, AI, Data Analytics, Backend, Frontend, DevOps).

## Dataset Columns Used

| Column | Used For |
|---|---|
| `title` | Job role dropdown, partial/case-insensitive filtering |
| `description` | Skill keyword extraction |
| `pay_period` | Displayed alongside compensation in the sample jobs table |
| `compensation` | Average / highest / lowest salary calculation (numeric; see note below) |
| `work_type` | Work type distribution chart |
| `job_location` | Most common location |
| `level` | Most common experience level, level distribution chart |
| `company_id` | Top 5 Hiring Company IDs (the dataset has no company name column, only numeric IDs) |

Two extra columns exist in the cleaned CSV but are **not** used by the app:
`cleaned_description` and `extracted_skills` (see
[Data Cleaning & NLP Pipeline](#data-cleaning--nlp-pipeline) for why).

> **Data note:** `compensation` doesn't consistently scale with `pay_period`
> in this dataset - rows labelled `HOURLY` still contain values in the
> tens/hundreds of thousands, not a true per-hour rate. Because of this,
> the app reports `compensation` values as-is rather than annualizing them,
> to avoid producing misleading multi-million-dollar averages. This is a
> property of the underlying source data, not something the cleaning step
> changes - cleaning only converts the column from a `"$42,000.00"`-style
> string to a proper number.

## Project Features

- **Smart job search** - select a role and get partial, case-insensitive
  matches (e.g. "Full Stack Developer" also matches "Senior Full Stack
  Developer", "Java Full Stack Developer", ".NET Full Stack Developer")
- **Job Statistics** - total matching jobs, most common work type,
  experience level, and location
- **Salary Analytics** - average, highest, and lowest salary for the
  matched postings, handled gracefully when salary data is unavailable
- **Top 5 Hiring Company IDs** for the selected role (the dataset has no
  company name column)
- **Top 10 Required Skills**, extracted from a large categorized keyword
  list covering programming languages, frameworks, cloud, databases, soft
  skills, AI, data analytics, and DevOps
- **Skill Gap Analysis** - skills you already have (✔) vs. skills you need
  (❌), grouped by area (Frontend/Backend/Database/Cloud/Tools/Soft
  Skills) and prioritized (High/Medium/Low) by how often each skill
  appears across matching postings
- Skill matching is **normalized**, so variants like "ReactJS" / "React.js"
  / "react" are all recognized as the same skill
- A **Career Readiness score** (Beginner / Intermediate / Advanced / Job
  Ready) weighted by skill priority, plus a simple **Match Score** percentage
- A structured, rule-based **Learning Roadmap** (Week 1-4, Projects,
  Interview Prep), always available even without an API key
- An optional **AI Career Coach powered by Google Gemini** - a fully
  structured recommendation (career summary, strengths, missing skills,
  learning roadmap, certifications, interview prep, resume advice, and
  more), with automatic retry on transient errors and a graceful fallback
  to the rule-based roadmap if Gemini is unavailable
- Interactive, styled Plotly visualizations (dark theme, full-width):
  Top 10 Required Skills, Top Hiring Company IDs, Work Type Distribution,
  Experience Level Distribution
- A table of sample matching job postings

## Data Cleaning & NLP Pipeline

`Data_Cleaning_and_NLP.ipynb` is a standalone Jupyter/Colab notebook that
takes the original `LinkedIn_RDB_three.csv` export and produces
`data/linkedin_jobs_cleaned.csv`, the file the Streamlit app actually reads.
It:

1. Strips whitespace from text columns and removes exact duplicate rows.
2. Converts `compensation` from a `"$42,000.00"`-style string into a clean
   numeric column.
3. Fills missing `level` / `job_domain` values with `"Not Specified"`.
4. Runs a basic NLP pass over `description` (lowercasing, stopword
   removal, and light normalization of a few tech-term variants) into a
   new `cleaned_description` column, then extracts skills from that text
   into an `extracted_skills` column using a small demo keyword list.

**Why the app doesn't use `extracted_skills` directly:** the notebook's
extraction list has about two dozen skills for demonstration purposes. The
app's own extraction (`SKILL_KEYWORDS` in `analysis.py`) covers 100+ skills
across 10 categories and already does its own case-insensitive, word-boundary
matching directly against `description`. Using the notebook's narrower list
for the live app would detect fewer skills than the app already finds on its
own, so `analysis.py` keeps its existing extraction and simply leaves
`cleaned_description` / `extracted_skills` unused in the DataFrame. If the
NLP pipeline grows a richer skill list in the future, swapping it in is a
one-function change in `get_top_skills`.

To re-run the notebook yourself, it needs its own dependencies
(`nltk`, `matplotlib`, `seaborn`) - these are separate from `requirements.txt`
since the Streamlit app itself doesn't need them at runtime.

## Installation

### 1. Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install the requirements

```bash
pip install -r requirements.txt
```

## How to Add Your Gemini API Key

The AI recommendation feature is optional - the rest of the app works
fully without it.

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Copy `.env.example` to a new file named `.env` in the project root:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` and paste your key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```
4. Never commit your real `.env` file - it's already listed in `.gitignore`.

If no key is set, the app will simply show a message and skip the AI
recommendation section - everything else keeps working.

## How to Run

From inside the `AI-Career-Advisor` folder, run:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`)
in your browser.

Make sure `data/linkedin_jobs_cleaned.csv` is present relative to `app.py` —
the app reads the dataset from that path. If you only have the raw
`LinkedIn_RDB_three.csv`, run `Data_Cleaning_and_NLP.ipynb` first to
generate the cleaned file.

## Folder Structure

```
AI-Career-Advisor/
│
├── app.py                        # Streamlit UI (loads assets, wires up analysis + Gemini)
├── analysis.py                    # Data processing, filtering, and analysis functions
├── Data_Cleaning_and_NLP.ipynb    # Notebook: raw CSV -> cleaned CSV (see above)
├── requirements.txt
├── README.md
├── .env.example                   # Template for your Gemini API key - copy to .env
│
├── assets/
│   ├── style.css                   # All custom CSS (dark theme, blue/cyan accents)
│   ├── hero.html                   # Static hero section markup
│   └── footer.html                 # Footer markup
│
└── data/
    ├── linkedin_jobs_cleaned.csv    # Used by the app (output of the notebook)
    └── LinkedIn_RDB_three.csv       # Original raw export (input to the notebook)
```

## Libraries Used

- **Streamlit** – Web app UI framework
- **Pandas** – Data loading, filtering, and aggregation
- **Plotly Express** – Interactive charts
- **python-dotenv** – Loads the Gemini API key from `.env`
- **google-genai** – Official Google Gen AI SDK, used for the optional
  Gemini-powered career recommendation

## Future Improvements

- Allow multi-role comparison
- Add trend analysis over time if a posting-date column becomes available
- Support resume upload for automatic skill extraction
- Resolve `company_id` values to real company names if that data becomes
  available
- Expand the notebook's `extracted_skills` keyword list and consider using
  it as a second signal alongside the app's own keyword extraction
- Deploy the app online for public access