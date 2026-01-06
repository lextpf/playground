"""Learning goal: get comfortable with the basic pandas DataFrame.

Today I want to actually *touch* a DataFrame for the first time:
- load a CSV from disk into a DataFrame
- peek at it (head / info / describe)
- select a single column and a few columns
- filter rows with a boolean condition
- sort by a column
- add a new "derived" column computed from existing ones

I'm a total beginner, so I'm printing a LOT and commenting everything
so future-me understands what each line is doing.
"""

import os

import numpy as np  # noqa: F401  (imported so I remember it's allowed today)
import pandas as pd


def load_data():
    """Read the CSV that lives right next to this script."""
    # __file__ is the path to THIS .py file. os.path.dirname gives its folder.
    # Then I join that folder with the csv name so it works no matter where
    # I run python from. (I kept running it from the wrong folder before!)
    here = os.path.dirname(__file__)
    csv_path = os.path.join(here, "sample_data.csv")
    print("Reading CSV from:", csv_path)

    # pd.read_csv turns the file into a DataFrame (like a spreadsheet/table).
    df = pd.read_csv(csv_path)
    return df


def explore(df):
    """Print the standard 'first look' summaries of a DataFrame."""
    # .head() shows the first 5 rows. Great sanity check that loading worked.
    print("\n--- df.head() : first rows ---")
    print(df.head())

    # .info() tells me column names, how many non-null values, and dtypes.
    # Notice age/salary show fewer non-null counts because some cells are empty!
    print("\n--- df.info() : structure + dtypes ---")
    df.info()

    # .describe() gives summary stats (count, mean, std, min, max, quartiles)
    # but ONLY for numeric columns by default.
    print("\n--- df.describe() : numeric summary ---")
    print(df.describe())

    # .shape is a (rows, columns) tuple. Handy quick fact.
    print("\n--- df.shape (rows, cols) ---")
    print(df.shape)


def select_and_filter(df):
    """Pick columns, then keep only rows matching a condition."""
    # Selecting ONE column with df["name"] gives a Series (a single column).
    print("\n--- select one column: name ---")
    print(df["name"])

    # Selecting MULTIPLE columns needs a list of names -> still a DataFrame.
    print("\n--- select a few columns: name, city, salary ---")
    print(df[["name", "city", "salary"]])

    # Boolean filtering: df["salary"] > 60000 makes a True/False Series,
    # and df[<that mask>] keeps only the True rows.
    print("\n--- boolean filter: salary > 60000 ---")
    high_earners = df[df["salary"] > 60000]
    print(high_earners[["name", "salary", "department"]])

    # I can combine conditions with & (and) / | (or). Each condition needs ().
    print("\n--- filter: Engineering AND age under 40 ---")
    young_eng = df[(df["department"] == "Engineering") & (df["age"] < 40)]
    print(young_eng[["name", "age", "department"]])


def sort_and_derive(df):
    """Sort rows, then create a new column from existing ones."""
    # Sort by salary, biggest first. This returns a NEW sorted DataFrame.
    print("\n--- sort by salary (descending) ---")
    by_salary = df.sort_values("salary", ascending=False)
    print(by_salary[["name", "salary"]])

    # Adding a derived column: monthly salary = yearly / 12.
    # I use .copy() so I don't accidentally mutate the original df in place.
    enriched = df.copy()
    enriched["monthly_salary"] = enriched["salary"] / 12

    # Round to whole euros just so it prints nicely.
    enriched["monthly_salary"] = enriched["monthly_salary"].round(0)
    print("\n--- new derived column: monthly_salary ---")
    print(enriched[["name", "salary", "monthly_salary"]])


def main():
    df = load_data()
    explore(df)
    select_and_filter(df)
    sort_and_derive(df)
    print("\nDone exploring the DataFrame basics!")


if __name__ == "__main__":
    main()
