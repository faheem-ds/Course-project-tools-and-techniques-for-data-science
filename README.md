# Uber & Lyft Data Analysis Project

## Project Overview
This project analyzes Uber and Lyft ride data for Boston, MA, to extract insights and build predictive models. The goal is to understand ride patterns, explore trends, and potentially predict demand or pricing using Python-based data analysis. This project will help in having hand-on-experience of data loading, eda, cleaning, visualizing and used that cleaned data in ML modeling.

## Project Structure
```
Course_project_Tools_and_Techniques/
│
├─ .venv/                # Virtual environment (ignored in Git)
├─ data/                 # CSV datasets
├─ src/                  # Python scripts
│   ├─ all python scripts           
│   └─ utils.py          # Helper functions
├─  main.py              # Entry point for analysis
├─ .gitignore
├─ requirements.txt
├─ README.md
```

- data/ → Stores Uber/Lyft CSV datasets
- src/ → Python scripts for analysis
- .venv/ → Virtual environment for project dependencies (ignored in Git)

# Getting Started

1. Clone the repository
git clone https://github.com/faheem-ds/Course-project-tools-and-techniques-for-data-science.git
cd Course-project-tools-and-techniques-for-data-science

2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1   # PowerShell

3. Install dependencies
pip install -r requirements.txt

## How to Run
All analysis is done using Python files in the src/ folder.
To run the main script:

python main.py

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

All dependencies are listed in requirements.txt for reproducibility.

## Notes
- Keep all datasets in the data/ folder.
- .venv/ is ignored in Git to keep the repository clean.
- Use Python files in src/ for all scripts; Jupyter notebooks are not used.

## Author
Faheem
- GitHub: https://github.com/faheem-ds
- Email: msds25016@itu.edu.pk


