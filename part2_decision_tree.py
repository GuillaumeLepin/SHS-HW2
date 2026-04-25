import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

DATA_PATH = Path(__file__).parent / "drugs_dataset.csv"


# load and clean the data
def preprocess():
    df = pd.read_csv(DATA_PATH)

    print("shape:", df.shape)
    print(df.head())
    print("\nmissing values:")
    print(df.isnull().sum())

    # fill missing values (there are none but just to be safe)
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    X = df.drop(columns=["Drug"])
    y = df["Drug"]

    # encode BP as ordinal: LOW=0, NORMAL=1, HIGH=2
    X = X.copy()
    X["BP"] = X["BP"].map({"LOW": 0, "NORMAL": 1, "HIGH": 2})

    # encode cholesterol
    X["Cholesterol"] = X["Cholesterol"].map({"NORMAL": 0, "HIGH": 1})

    # one hot encode sex
    X = pd.get_dummies(X, columns=["Sex"], drop_first=True)

    # standardize age and Na_to_K
    sc = StandardScaler()
    X[["Age", "Na_to_K"]] = sc.fit_transform(X[["Age", "Na_to_K"]])

    print("\nfeatures:", list(X.columns))

    # 80/20 split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"train size: {len(X_train)}  val size: {len(X_val)}")
    return X_train, X_val, y_train, y_val


def evaluate(model, X_val, y_val, name):
    preds = model.predict(X_val)

    a  = accuracy_score(y_val,  preds)
    p  = precision_score(y_val, preds, average="weighted", zero_division=0)
    r  = recall_score(y_val,    preds, average="weighted", zero_division=0)
    f  = f1_score(y_val,        preds, average="weighted", zero_division=0)

    print(f"\n=== {name} ===")
    print(f"accuracy  = {a:.4f}")
    print(f"precision = {p:.4f}")
    print(f"recall    = {r:.4f}")
    print(f"f1        = {f:.4f}")
    print(classification_report(y_val, preds, zero_division=0))

    # confusion matrix
    labels = sorted(y_val.unique())
    cm   = confusion_matrix(y_val, preds, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix - " + name)
    plt.tight_layout()
    out = Path(__file__).parent / f"part2_cm_{name.replace(' ', '_')}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)

    return {"accuracy": a, "precision": p, "recall": r, "f1": f}


def tune_hyperparams(X_train, y_train):
    grid = {
        "criterion":         ["gini", "entropy"],
        "max_depth":         [2, 3, 4, 5, 6, 8, 10, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf":  [1, 2, 5, 10],
    }
    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        grid,
        scoring="f1_weighted",
        cv=5,
        n_jobs=1
    )
    gs.fit(X_train, y_train)
    print("best cv score:", gs.best_score_)
    print("best params:",   gs.best_params_)
    return gs.best_estimator_, gs.best_params_


if __name__ == "__main__":

    X_train, X_val, y_train, y_val = preprocess()

    # baseline tree with default settings
    print("\n--- baseline ---")
    baseline = DecisionTreeClassifier(random_state=42)
    baseline.fit(X_train, y_train)
    m_base = evaluate(baseline, X_val, y_val, "baseline")

    # tuned tree
    print("\n--- grid search ---")
    best_tree, best_params = tune_hyperparams(X_train, y_train)
    m_tuned = evaluate(best_tree, X_val, y_val, "tuned")

    print("\n--- summary ---")
    print("baseline:", m_base)
    print("tuned   :", m_tuned)
    print("params  :", best_params)

    # visualize the best tree
    classes = sorted(y_train.unique())
    fig, ax = plt.subplots(figsize=(20, 12))
    plot_tree(best_tree,
              feature_names=list(X_train.columns),
              class_names=list(classes),
              filled=True, rounded=True, fontsize=8, ax=ax)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "part2_best_tree.png", dpi=120)
    plt.close()

    print("tree saved")
    print(export_text(best_tree, feature_names=list(X_train.columns)))
