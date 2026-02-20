import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)

# =========================================================
# CONFIG
# =========================================================
INPUT_CSV = "data/Class5data_test_clean.csv"
TARGET_COL = "ProdTaken"

MODEL_PATH = "examples/class5_catboost_model.pkl"
THRESHOLD_PATH = "examples/class5_catboost_best_threshold.json"
META_PATH = "examples/class5_catboost_meta.json"

OUTPUT_PREDICTIONS_CSV = "examples/class5_catboost_predictions.csv"
CONFUSION_MATRIX_PNG = "examples/confusion_matrix_catboost.png"
CONFUSION_MATRIX_NORM_PNG = "examples/confusion_matrix_normalized_catboost.png"
ROC_CURVE_PNG = "examples/roc_curve_catboost.png"


def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        threshold = json.load(f)["best_threshold"]

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return model, threshold, meta


def save_confusion_matrix_plots(y_true, y_pred):
    os.makedirs("examples", exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]).plot(ax=ax, values_format="d")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PNG, dpi=300)
    plt.close(fig)

    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=[0, 1]).plot(ax=ax, values_format=".2f")
    ax.set_title("Normalized Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_NORM_PNG, dpi=300)
    plt.close(fig)


def save_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ROC_CURVE_PNG, dpi=300)
    plt.close(fig)

    return auc_score


def main():
    model, threshold, meta = load_artifacts()
    print(f"Loaded threshold: {threshold:.3f}")

    df = pd.read_csv(INPUT_CSV)

    y_true = None
    if TARGET_COL in df.columns:
        y_true = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int).values
        X = df.drop(columns=[TARGET_COL]).copy()
    else:
        X = df.copy()

    # align columns to training
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = X[c].astype(str).str.strip()

    feature_cols = meta["feature_columns"]
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feature_cols]  # exact order

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    output_df = df.copy()
    output_df["Predicted_Probability"] = y_prob
    output_df["Predicted_Class"] = y_pred

    os.makedirs("examples", exist_ok=True)
    output_df.to_csv(OUTPUT_PREDICTIONS_CSV, index=False)
    print(f"Saved predictions: {OUTPUT_PREDICTIONS_CSV}")

    if y_true is not None:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=4))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

        save_confusion_matrix_plots(y_true, y_pred)
        auc_score = save_roc_curve(y_true, y_prob)

        print(f"\nROC AUC: {auc_score:.4f}")
        print("\nSaved plots:")
        print(f"- {CONFUSION_MATRIX_PNG}")
        print(f"- {CONFUSION_MATRIX_NORM_PNG}")
        print(f"- {ROC_CURVE_PNG}")

    print("\nPreview:")
    preview_cols = ["Predicted_Probability", "Predicted_Class"]
    if TARGET_COL in output_df.columns:
        preview_cols = [TARGET_COL] + preview_cols
    print(output_df[preview_cols].head(10))


if __name__ == "__main__":
    main()