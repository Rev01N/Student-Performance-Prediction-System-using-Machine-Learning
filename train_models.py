"""
train_models.py — Trains 5 ML classifiers, evaluates, saves plots & best model.
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.svm             import SVC
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     confusion_matrix, classification_report)
from sklearn.preprocessing   import LabelEncoder

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "student_data.csv")
FIG_DIR    = os.path.join(BASE_DIR, "outputs", "figures")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")
for d in [FIG_DIR, REPORT_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

BG    = "#0F1117"
PANEL = "#1A1D27"
TXT   = "#E8EAF0"
GRID  = "#2A2D3A"
ACC   = "#7B68EE"
FEAT_COLS = ["Study_Hours","Attendance","Previous_Marks","Assignments","Internal_Marks"]

def _theme():
    plt.rcParams.update({"figure.facecolor":BG,"axes.facecolor":PANEL,
        "axes.edgecolor":GRID,"axes.labelcolor":TXT,"axes.titlecolor":TXT,
        "xtick.color":TXT,"ytick.color":TXT,"text.color":TXT,
        "grid.color":GRID,"font.size":11,"axes.titlesize":13})

def _save(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {p}")

def plot_cm(cm, labels, title, fname):
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor=BG, ax=ax,
                annot_kws={"size":14,"fontweight":"bold"})
    ax.set_title(f"Confusion Matrix — {title}", pad=12)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); _save(fig, fname)

def train_and_evaluate():
    _theme()
    df = pd.read_csv(DATA_PATH)
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["Final_Result"])
    class_names = list(le.classes_)

    X = df[FEAT_COLS].values
    y = df["Label"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=42, stratify=y)
    print(f"\n[Train] Train={len(X_tr)} | Test={len(X_te)} | Classes={class_names}\n")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree"      : DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest"      : RandomForestClassifier(n_estimators=150, random_state=42),
        "Naive Bayes"        : GaussianNB(),
        "SVM"                : SVC(kernel="rbf", probability=True, random_state=42),
    }

    results = {}
    for name, m in models.items():
        print(f"─── {name}")
        m.fit(X_tr, y_tr)
        yp = m.predict(X_te)
        acc  = accuracy_score(y_te, yp)
        prec = precision_score(y_te, yp, average="weighted", zero_division=0)
        rec  = recall_score(y_te, yp, average="weighted", zero_division=0)
        f1   = f1_score(y_te, yp, average="weighted", zero_division=0)
        cv   = cross_val_score(m, X, y, cv=5, scoring="accuracy").mean()
        results[name] = dict(Accuracy=acc, Precision=prec,
                             Recall=rec, F1=f1, CV=cv, model=m)
        print(f"    Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  CV={cv:.4f}")

        # Save report
        rpt = classification_report(y_te, yp, target_names=class_names, zero_division=0)
        rpt_path = os.path.join(REPORT_DIR, f"{name.replace(' ','_')}_report.txt")
        with open(rpt_path, "w") as f:
            f.write(f"Model: {name}\nAccuracy={acc:.4f}\n\n{rpt}")

        # Confusion matrix
        cm = confusion_matrix(y_te, yp)
        plot_cm(cm, class_names, name,
                f"09_cm_{name.replace(' ','_').lower()}.png")

    # Save best model
    joblib.dump(results["Random Forest"]["model"],
                os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    print("\n  Best model saved → models/best_model.pkl")

    # Comparison chart
    names = list(results.keys())
    accs  = [results[n]["Accuracy"] for n in names]
    f1s   = [results[n]["F1"]       for n in names]
    cvs   = [results[n]["CV"]       for n in names]
    x, w  = np.arange(len(names)), 0.26
    fig, ax = plt.subplots(figsize=(13,6))
    for i,(vals,lbl,clr) in enumerate([(accs,"Accuracy","#7B68EE"),
                                        (f1s,"F1-Score","#4CAF82"),
                                        (cvs,"CV Acc","#FFD166")]):
        bars = ax.bar(x+(i-1)*w, vals, w, label=lbl,
                      color=clr, alpha=0.88, edgecolor=BG)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                    f"{b.get_height():.2f}", ha="center", va="bottom",
                    fontsize=8.5, color=TXT, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0.5,1.06); ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", pad=15, fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.4)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout(); _save(fig, "10_model_comparison.png")

    # Feature importance
    rf = results["Random Forest"]["model"]
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar([FEAT_COLS[i].replace("_"," ") for i in idx],
                  imp[idx], color=ACC, alpha=0.85, edgecolor=BG)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
                f"{b.get_height():.3f}", ha="center", va="bottom",
                fontsize=10, color=TXT, fontweight="bold")
    ax.set_title("Feature Importance — Random Forest", pad=14)
    ax.set_ylabel("Importance Score"); ax.set_xlabel("Feature")
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout(); _save(fig, "11_feature_importance.png")

    # Summary
    print("\n" + "="*65)
    print(f"  {'Model':<25} {'Accuracy':>9} {'F1-Score':>9} {'CV Acc':>9}")
    print("  " + "─"*63)
    for n in names:
        r = results[n]
        print(f"  {n:<25} {r['Accuracy']:>9.4f} {r['F1']:>9.4f} {r['CV']:>9.4f}")
    best = max(results, key=lambda n: results[n]["Accuracy"])
    print(f"\n  Best Model: {best}  |  Accuracy: {results[best]['Accuracy']*100:.2f}%")
    print("="*65 + "\n")
    return results

if __name__ == "__main__":
    train_and_evaluate()
