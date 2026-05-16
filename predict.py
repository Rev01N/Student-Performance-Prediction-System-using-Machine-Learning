import os, sys
import numpy as np
import joblib

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(BASE_DIR, "models")
MODEL_PATH   = os.path.join(MODELS_DIR, "best_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

FEAT_COLS = [
    ("Study_Hours",    "Daily study hours",            1.0, 10.0),
    ("Attendance",     "Attendance percentage",        50.0,100.0),
    ("Previous_Marks", "Previous exam marks",          30.0,100.0),
    ("Assignments",    "Assignment score",             40.0,100.0),
    ("Internal_Marks", "Internal assessment marks",   10.0, 50.0),
]

BANNER = """
╔══════════════════════════════════════════════════════╗
║   Student Performance Prediction System              ║
╚══════════════════════════════════════════════════════╝
"""

def get_float(prompt, lo, hi):
    while True:
        try:
            val = float(input(f"  {prompt} [{lo}–{hi}]: ").strip())
            if lo <= val <= hi:
                return val
            print(f"  ⚠  Please enter a value between {lo} and {hi}.")
        except ValueError:
            print("  ⚠  Invalid input. Please enter a number.")

def predict_student():
    print(BANNER)

    # ── Load model & encoder ─────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print("  ✖ Model not found. Please run main.py first to train models.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    le    = joblib.load(ENCODER_PATH)
    print("  ✔ Model loaded successfully.\n")
    print("  Enter the following student details:\n")

    # ── Collect input ────────────────────────────────────────────────────────
    values = []
    for col, desc, lo, hi in FEAT_COLS:
        val = get_float(desc, lo, hi)
        values.append(val)

    # ── Predict ──────────────────────────────────────────────────────────────
    X_new = np.array(values).reshape(1, -1)
    pred_encoded  = model.predict(X_new)[0]
    pred_proba    = model.predict_proba(X_new)[0]
    pred_label    = le.inverse_transform([pred_encoded])[0]

    confidence    = max(pred_proba) * 100
    classes       = list(le.classes_)
    class_probs   = {cls: f"{prob*100:.1f}%" for cls, prob in zip(classes, pred_proba)}

    # ── Display result ────────────────────────────────────────────────────────
    print("\n" + "─" * 54)
    print("  PREDICTION RESULT")
    print("─" * 54)
    print(f"  Final Result  : {'PASS' if pred_label == 'Pass' else 'FAIL'}")
    print(f"  Confidence    : {confidence:.1f}%")
    print(f"\n  Class Probabilities:")
    for cls, prob in class_probs.items():
        print(f"{cls:<6}: {prob:>6}")
    print("─" * 54)

    if pred_label == "Fail":
        print("\n  ⚠ Recommendation: This student is AT RISK of failing.")
        print("    Suggested actions:")
        study_h, attend, prev = values[0], values[1], values[2]
        if study_h < 4.0:
            print(f"    • Increase study hours (currently {study_h}h/day → target ≥5h)")
        if attend < 75.0:
            print(f"    • Improve attendance (currently {attend:.1f}% → target ≥75%)")
        if prev < 50.0:
            print(f"    • Provide extra tutoring support (Previous marks: {prev:.1f})")
    else:
        print("\nStudent is on track to PASS. Keep it up!")
    print()

if __name__ == "__main__":
    predict_student()
