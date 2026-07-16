# AI Career Advisor

**Subject:** Big Data Analytics — College Microproject
**Project Title:** AI Career Advisor for Job Market Trend Analysis and Skill Demand Prediction Using Big Data Analytics

## Project Overview

AI Career Advisor is a Streamlit web app that helps students identify skill
gaps for a desired job role. It analyzes a real-world LinkedIn job postings
dataset (`LinkedIn_RDB_three.csv`) to find the most in-demand skills for a
chosen role, compares them against the skills the user already has, and
optionally asks Google Gemini for a personalized career recommendation.

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
| `compensation` | Average / highest / lowest salary calculation (used as-is; see note below) |
| `work_type` | Work type distribution chart |
| `job_location` | Most common location |
| `level` | Most common experience level, level distribution chart |
| `company_id` | Top 5 Hiring Companies (the dataset has no company name column, only numeric IDs) |

> **Data note:** The `compensation` column does not consistently scale with
> `pay_period` in this dataset (rows labelled `HOURLY` still contain values
> in the tens/hundreds of thousands, not a true per-hour rate). Because of
> this, the app reports `compensation` values as-is rather than
> annualizing them, to avoid producing misleading multi-million-dollar
> averages.

## Project Features

- **Smart job search** - select a role and get partial, case-insensitive
  matches (e.g. "Full Stack Developer" also matches "Senior Full Stack
  Developer", "Java Full Stack Developer", ".NET Full Stack Developer")
- **Job Statistics** - total matching jobs, most common work type,
  experience level, and location
- **Salary Analytics** - average, highest, and lowest salary for the
  matched postings, handled gracefully when salary data is unavailable
- **Top 5 Hiring Companies** for the selected role
- **Top 10 Required Skills**, extracted from a large categorized keyword
  list covering programming languages, frameworks, cloud, databases, soft
  skills, AI, data analytics, and DevOps
- **Skill Gap Analysis** - skills you already have (✔) vs. skills you need
  (❌), plus a **Match Score** (0-100%) showing how well your skills fit
  the role
- A dynamically generated, rule-based **Learning Recommendation**
- An optional **AI Career Recommendation powered by Google Gemini** -
  summary, career advice, a suggested learning roadmap, and future skills
  to watch, generated from your specific role and skill gap
- Interactive, styled Plotly visualizations (dark theme, full-width):
  Top 10 Required Skills, Top Hiring Companies, Work Type Distribution,
  Experience Level Distribution
- A table of sample matching job postings

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

Make sure `data/LinkedIn_RDB_three.csv` is present relative to `app.py` —
the app reads the dataset from that path.

## Folder Structure

```
AI-Career-Advisor/
│
├── app.py                 # Streamlit UI (loads assets, wires up analysis + Gemini)
├── analysis.py             # Data processing, filtering, and analysis functions
├── requirements.txt
├── README.md
├── .env.example            # Template for your Gemini API key - copy to .env
│
├── assets/
│   ├── style.css            # All custom CSS (dark theme, blue/cyan accents)
│   ├── hero.html            # Static hero section markup
│   └── footer.html          # Footer markup
│
└── data/
    └── LinkedIn_RDB_three.csv
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
- Deploy the app online for public access