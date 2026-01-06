"""Learning goal: handle missing values and do my first groupby aggregation.

Real data is messy, so today I practice:
- detecting missing values with isna().sum()
- two ways to deal with them: drop rows, OR fill the gaps
- grouping rows by a category (department) and aggregating
  (mean salary + how many people per department)

Still a beginner, so heavy comments + lots of prints to see what happens.
"""

import os

import numpy as np  # noqa: F401  (allowed today; keeping it imported)
import pandas as pd


def load_data():
    """Read the same CSV used by dataframe_intro.py."""
    here = os.path.dirname(__file__)
    csv_path = os.path.join(here, "sample_data.csv")
    print("Reading CSV from:", csv_path)
    df = pd.read_csv(csv_path)
    return df


def detect_missing(df):
    """Show where the holes in the data are."""
    # isna() returns a DataFrame of True/False (True == this cell is missing).
    # Summing booleans counts the Trues, so .sum() per column = missing count.
    print("\n--- missing values per column ---")
    print(df.isna().sum())

    # Total number of missing cells across the whole table.
    total_missing = int(df.isna().sum().sum())
    print("\nTotal missing cells:", total_missing)


def handle_missing(df):
    """Two strategies: drop vs fill. I'll keep the FILLED version going forward."""
    # Strategy 1, DROP any row that has at least one missing value.
    # Good when missing data is rare and you can afford to lose rows.
    dropped = df.dropna()
    print("\n--- after dropna(): rows kept ---")
    print("rows before:", len(df), "| rows after dropna:", len(dropped))

    # Strategy 2, FILL the gaps instead of throwing rows away.
    # For 'age' (numeric) I fill with the column's mean, rounded to a whole year.
    # For 'salary' (numeric) I fill with the column's median.
    # I work on a copy so the original df stays untouched.
    filled = df.copy()

    age_mean = round(filled["age"].mean())
    filled["age"] = filled["age"].fillna(age_mean)

    salary_median = filled["salary"].median()
    filled["salary"] = filled["salary"].fillna(salary_median)

    print("\n--- after filling (age=mean, salary=median) ---")
    print("filled age missing :", int(filled["age"].isna().sum()))
    print("filled salary missing:", int(filled["salary"].isna().sum()))
    print("(used age_mean =", age_mean, ", salary_median =", salary_median, ")")

    # Return the cleaned (filled) DataFrame so groupby works on complete data.
    return filled


def summarize_by_department(df):
    """Group rows by department and build a tiny summary table."""
    # groupby("department") buckets the rows by their department value.
    # Then .agg(...) lets me compute several things at once per bucket.
    summary = df.groupby("department").agg(
        avg_salary=("salary", "mean"),
        head_count=("id", "count"),
    )

    # Round the average salary so the table reads cleanly.
    summary["avg_salary"] = summary["avg_salary"].round(0)

    # Sort departments by average salary, highest paid on top.
    summary = summary.sort_values("avg_salary", ascending=False)

    print("\n--- aggregated summary by department ---")
    print(summary)

    return summary


def main():
    df = load_data()
    detect_missing(df)
    clean = handle_missing(df)
    summarize_by_department(clean)
    print("\nDone with cleaning + groupby practice!")


if __name__ == "__main__":
    main()
