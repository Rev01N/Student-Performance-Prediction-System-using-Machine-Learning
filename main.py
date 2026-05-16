"""
main.py
-------
Master pipeline runner for the Student Performance Prediction System.

Execution order:
  1. Generate synthetic dataset (data/student_data.csv)
  2. Run Exploratory Data Analysis  → outputs/figures/
  3. Train & evaluate all 5 models  → outputs/figures/ + outputs/reports/
  4. Print final summary

Usage:
    python main.py
"""

import os, sys, time

# Ensure src/ is on the path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_DIR)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
sys.path.insert(0, DATA_DIR)

BANNER = r"""
  ╔══════════════════════════════════════════════════════════════╗
  ║       STUDENT PERFORMANCE PREDICTION SYSTEM                  ║
  ║       Machine Learning Project  |  Python · Scikit-learn     ║
  ╚══════════════════════════════════════════════════════════════╝
"""

def section(title):
    print(f"\n{'='*65}")
    print(f"  STEP: {title}")
    print(f"{'='*65}")

def main():
    print(BANNER)
    total_start = time.time()

    # ── Step 1: Generate Dataset ──────────────────────────────────────────────
    section("1 — Generate Synthetic Dataset")
    t = time.time()
    from data.generate_dataset import generate_dataset
    output_path = os.path.join(DATA_DIR, "student_data.csv")
    df = generate_dataset(n_students=500)
    df.to_csv(output_path, index=False)
    print(f"  ✔ Dataset created  ({len(df)} rows) in {time.time()-t:.2f}s")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Pass/Fail: {df['Final_Result'].value_counts().to_dict()}")

    # ── Step 2: EDA ────────────────────────────────────────────────────────────
    section("2 — Exploratory Data Analysis")
    t = time.time()
    from src.eda import run_eda
    run_eda()
    print(f"  ✔ EDA complete in {time.time()-t:.2f}s")

    # ── Step 3: Train & Evaluate Models ───────────────────────────────────────
    section("3 — Model Training & Evaluation (5 Models)")
    t = time.time()
    from src.train_models import train_and_evaluate
    results = train_and_evaluate()
    print(f"  ✔ Training complete in {time.time()-t:.2f}s")

    # ── Final Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    print(f"\n{'='*65}")
    print(f"  PIPELINE COMPLETE  ·  Total time: {elapsed:.2f}s")
    print(f"{'='*65}")
    print(f"\n  Output files:")
    print(f"    Plots   → outputs/figures/   (11 PNG files)")
    print(f"    Reports → outputs/reports/   (5 TXT files)")
    print(f"    Model   → models/best_model.pkl")
    print(f"\n  To predict a new student:")
    print(f"    python src/predict.py\n")

if __name__ == "__main__":
    main()
