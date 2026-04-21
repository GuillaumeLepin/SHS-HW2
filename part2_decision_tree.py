"""
HW2-2 Part 2: Decision Tree Classifier on drugs_dataset.csv
Covers preprocessing, baseline training, hyperparameter tuning via
GridSearchCV, evaluation (accuracy / precision / recall / F1 /
confusion matrix), and visualization of the final tree.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix,
                             ConfusionMatrixDisplay,
                             classification_report)

HERE = Path(__file__).parent
DATA_PATH = HERE / "drugs_dataset.csv"

RANDOM_STATE = 0


# ---------------------------------------------------------------------------
# Part 1: Preprocessing
# ---------------------------------------------------------------------------
def preprocess():
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)
    print("\nHead:\n", df.head())
    print("\nMissing values per column:\n", df.isnull().sum())

    # Handle missing values: impute numeric with median, categorical with mode.
    # (No missing values in this particular dataset, but we keep the logic
    # so the pipeline is robust.)
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    y = df["Drug"]
    X = df.drop(columns=["Drug"])

    # Categorical encoding:
    #   - Sex (2 levels, no ordering)            -> one-hot (drop first)
    #   - BP (HIGH/NORMAL/LOW, ordered)          -> ordinal mapping
    #   - Cholesterol (HIGH/NORMAL, ordered)     -> ordinal mapping
    bp_order = {"LOW": 0, "NORMAL": 1, "HIGH": 2}
    chol_order = {"NORMAL": 0, "HIGH": 1}
    X = X.copy()
    X["BP"] = X["BP"].map(bp_order)
    X["Cholesterol"] = X["Cholesterol"].map(chol_order)
    X = pd.get_dummies(X, columns=["Sex"], drop_first=True)

    # Numerical attributes: decision trees are invariant to monotonic
    # transforms, so normalization is not strictly needed. We still
    # standardize here to keep the features comparable in case we want to
    # inspect them or swap in another model later.
    scaler = StandardScaler()
    num_cols = ["Age", "Na_to_K"]
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")
    print(f"Features: {list(X.columns)}")
    return X_train, X_val, y_train, y_val


# ---------------------------------------------------------------------------
# Part 2: Baseline decision tree
# ---------------------------------------------------------------------------
def evaluate(model, X_val, y_val, name):
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_val, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    print(f"\n=== {name} ===")
    print(f"  accuracy           = {acc:.4f}")
    print(f"  precision (weight) = {prec:.4f}")
    print(f"  recall    (weight) = {rec:.4f}")
    print(f"  f1        (weight) = {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_val, y_pred, zero_division=0))

    cm = confusion_matrix(y_val, y_pred, labels=sorted(y_val.unique()))
    disp = ConfusionMatrixDisplay(cm, display_labels=sorted(y_val.unique()))
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix — {name}")
    fig.tight_layout()
    fig.savefig(HERE / f"part2_cm_{name.replace(' ', '_')}.png", dpi=120)
    plt.close(fig)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def train_baseline(X_train, y_train):
    clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    return clf


# ---------------------------------------------------------------------------
# Part 3: Hyperparameter tuning
# ---------------------------------------------------------------------------
def tune(X_train, y_train):
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
    }
    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=5,
        n_jobs=1,
    )
    gs.fit(X_train, y_train)
    print("\nBest CV score:", gs.best_score_)
    print("Best params  :", gs.best_params_)
    return gs.best_estimator_, gs.best_params_


# ---------------------------------------------------------------------------
# Part 4: Visualize the tree
# ---------------------------------------------------------------------------
def visualize_tree(clf, feature_names, class_names, out_path):
    fig, ax = plt.subplots(figsize=(20, 12))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("\nText representation of the tree:\n")
    print(export_text(clf, feature_names=list(feature_names)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    X_train, X_val, y_train, y_val = preprocess()

    # Baseline
    baseline = train_baseline(X_train, y_train)
    baseline_metrics = evaluate(baseline, X_val, y_val, "baseline")

    # Hyperparameter tuning
    best, best_params = tune(X_train, y_train)
    tuned_metrics = evaluate(best, X_val, y_val, "tuned")

    print("\n=== Summary ===")
    print("baseline :", baseline_metrics)
    print("tuned    :", tuned_metrics)
    print("best params:", best_params)

    # Tree visualization (the tuned, best-performing one)
    classes = sorted(y_train.unique())
    visualize_tree(best, X_train.columns, classes,
                   HERE / "part2_best_tree.png")


if __name__ == "__main__":
    main()
