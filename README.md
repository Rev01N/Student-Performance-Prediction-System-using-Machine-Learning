# Student Performance Prediction System

> Machine Learning project to predict student academic outcomes (Pass/Fail) using Python and Scikit-learn.

---

## Project Structure

```
Student Performance Prediction/
├── data/
│   ├── generate_dataset.py     # Synthetic dataset generator
│   └── student_data.csv        # Generated after running main.py
├── src/
│   ├── eda.py                  # Exploratory Data Analysis (8 plots)
│   ├── preprocess.py           # Data cleaning & normalisation
│   ├── train_models.py         # Train 5 ML models + evaluation
│   └── predict.py              # Interactive CLI prediction tool
├── outputs/
│   ├── figures/                # All generated plots (PNG)
│   └── reports/                # Classification reports (TXT)
├── models/
│   ├── best_model.pkl          # Saved Random Forest model
│   └── label_encoder.pkl       # Saved LabelEncoder
├── main.py                     # Master pipeline runner
├── requirements.txt
└── README.md
```

---

## Dataset Features

| Feature | Description | Range |
|---|---|---|
| Student_ID | Unique identifier | 1 – 500 |
| Study_Hours | Daily study hours | 1.0 – 10.0 |
| Attendance | Attendance percentage | 50.0 – 100.0 |
| Previous_Marks | Past exam score | 30.0 – 100.0 |
| Assignments | Assignment completion score | 40.0 – 100.0 |
| Internal_Marks | Internal assessment marks | 10.0 – 50.0 |
| **Final_Result** | **Target: Pass / Fail** | — |

---
