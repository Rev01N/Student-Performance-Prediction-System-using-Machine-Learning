"""
generate_dataset.py
-------------------
Generates a realistic synthetic dataset of 500 students for the
Student Performance Prediction project.

Columns:
    Student_ID       - Unique student identifier
    Study_Hours      - Daily study hours (1.0 – 10.0)
    Attendance       - Attendance percentage (50.0 – 100.0)
    Previous_Marks   - Past exam score (30.0 – 100.0)
    Assignments      - Assignment completion score (0.0 – 100.0)
    Internal_Marks   - Internal assessment score (0.0 – 50.0)
    Final_Result     - Target label: Pass / Fail
"""

import numpy as np
import pandas as pd
import os

def generate_dataset(n_students: int = 500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    # ── Base features (independent distributions) ────────────────────────────
    study_hours    = np.round(np.random.uniform(1.0, 10.0, n_students), 1)
    attendance     = np.round(np.random.uniform(50.0, 100.0, n_students), 1)
    previous_marks = np.round(np.random.uniform(30.0, 100.0, n_students), 1)
    assignments    = np.round(np.random.uniform(40.0, 100.0, n_students), 1)
    internal_marks = np.round(np.random.uniform(10.0, 50.0, n_students), 1)

    # ── Derive Final_Result from a weighted performance score ─────────────────
    # Normalise each feature to [0, 1] range
    def norm(arr, lo, hi):
        return (arr - lo) / (hi - lo)

    score = (
        0.30 * norm(study_hours,    1.0,  10.0) +
        0.25 * norm(attendance,    50.0, 100.0) +
        0.25 * norm(previous_marks, 30.0, 100.0) +
        0.10 * norm(assignments,   40.0, 100.0) +
        0.10 * norm(internal_marks, 10.0,  50.0)
    )

    # Add Gaussian noise so the boundary isn't perfectly linear
    noise = np.random.normal(0, 0.08, n_students)
    score = np.clip(score + noise, 0, 1)

    # Threshold: score >= 0.50 → Pass  (≈ 65 % pass rate)
    final_result = np.where(score >= 0.50, "Pass", "Fail")

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "Student_ID"    : range(1, n_students + 1),
        "Study_Hours"   : study_hours,
        "Attendance"    : attendance,
        "Previous_Marks": previous_marks,
        "Assignments"   : assignments,
        "Internal_Marks": internal_marks,
        "Final_Result"  : final_result,
    })

    return df


if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "student_data.csv")
    df = generate_dataset(n_students=500)
    df.to_csv(output_path, index=False)

    print("=" * 55)
    print("  Dataset Generated Successfully!")
    print("=" * 55)
    print(f"  Saved  : {output_path}")
    print(f"  Rows   : {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  Class Distribution:")
    print(df["Final_Result"].value_counts().to_string())
    print("=" * 55)
