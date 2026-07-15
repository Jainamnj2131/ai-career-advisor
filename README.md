# AI Career Advisor

**Subject:** Big Data Analytics — College Microproject
**Project Title:** AI Career Advisor for Job Market Trend Analysis and Skill Demand Prediction Using Big Data Analytics

## Project Overview

AI Career Advisor is a simple, clean Streamlit web app that helps students
identify skill gaps for a desired job role. It analyzes a real-world
LinkedIn job postings dataset (`LinkedIn_RDB_three.csv`) to find the most
in-demand skills for a chosen role and compares them against the skills the
user already has.

The dataset does **not** contain a dedicated "skills" column, so skills are
extracted directly from each job's `description` text using a predefined
keyword list (Python, SQL, AWS, Power BI, Machine Learning, etc.).

## Dataset Columns Used

| Column | Used For |
|---|---|
| `title` | Job role dropdown, filtering |
| `description` | Skill keyword extraction |
| `pay_period` | Displayed alongside compensation in the sample jobs table |
| `compensation` | Average compensation calculation (used as-is; see note below) |
| `work_type` | Work type distribution chart |
| `job_location` | Most common location |
| `level` | Most common experience level, level distribution chart |

> **Data note:** The `compensation` column does not consistently scale with
> `pay_period` in this dataset (rows labelled `HOURLY` still contain values
> in the tens/hundreds of thousands, not a true per-hour rate). Because of
> this, the app reports the average `compensation` value as-is rather than
> annualizing it, to avoid producing misleading multi-million-dollar
> averages.

## Features

- Select a desired job role from real job posting titles
- Enter your current skills as free text
- View a job summary: matching job count, estimated average yearly salary,
  most common work type, experience level, and location
- See the Top 10 Required Skills for the selected role (extracted from job
  descriptions)
- Skill Gap Analysis: see which required skills you already have (✔) and
  which you're missing (❌)
- Get a dynamically generated learning recommendation
- Interactive Plotly visualizations:
  - Top 10 Required Skills (bar chart)
  - Work Type Distribution (pie chart)
  - Experience Level Distribution (bar chart)
- View a table of sample matching job postings

## Installation

1. Make sure Python 3.8+ is installed.
2. Clone or download this project folder.
3. Install the required libraries:

```bash
pip install -r requirements.txt
```

## How to Run

From inside the `AI-Career-Advisor` folder, run:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`)
in your browser.

Make sure `data/LinkedIn_RDB_three.csv` is present relative to `app.py` —
the app reads the dataset from that path.

## Project Structure

```
AI-Career-Advisor/
│
├── data/
│   └── LinkedIn_RDB_three.csv
│
├── app.py             # Streamlit UI
├── analysis.py         # Data processing & analysis functions
├── requirements.txt
└── README.md
```

## Libraries Used

- **Streamlit** – Web app UI framework
- **Pandas** – Data loading, filtering, and aggregation
- **Plotly Express** – Interactive charts

## Future Improvements

- Add real AI-powered recommendations using an LLM API
- Allow multi-role comparison
- Add trend analysis over time if a posting-date column becomes available
- Support resume upload for automatic skill extraction
- Deploy the app online for public access