import os
import json
import pickle
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# =========================================================
# CONFIG
# =========================================================
TRAIN_CSV = "data/Class5data_train_clean.csv"
VAL_CSV   = "data/Class5data_validation_clean.csv"
TEST_CSV  = "data/Class5data_test_clean.csv"

TARGET_COL = "ProdTaken"

MODEL_PATH = "examples/class5_catboost_model.pkl"
THRESHOLD_PATH = "examples/class5_catboost_best_threshold.json"
META_PATH = "examples/class5_catboost_meta.json"

RANDOM_SEED = 42


def load_split_data():
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if TARGET_COL not in df.columns:
            raise ValueError(f"{TARGET_COL} not found in {name} file.")

    return train_df, val_df, test_df


def prepare_xy(df):
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)
    X = df.drop(columns=[TARGET_COL]).copy()

    # Clean whitespace in object columns (helps category consistency)
    obj_cols = X.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        X[c] = X[c].astype(str).str.strip()

    return X, y


def get_cat_features(X):
    # CatBoost accepts names of categorical columns
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    return cat_cols


def find_best_threshold(y_true, y_prob):
    best_thr = 0.5
    best_acc = 0.0

    # Search a bit wider
    for thr in np.arange(0.20, 0.81, 0.005):
        y_pred = (y_prob >= thr).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)

    return best_thr, best_acc


def main():
    os.makedirs("examples", exist_ok=True)

    # 1) Load
    train_df, val_df, test_df = load_split_data()
    X_train, y_train = prepare_xy(train_df)
    X_val, y_val = prepare_xy(val_df)
    X_test, y_test = prepare_xy(test_df)

    cat_cols = get_cat_features(X_train)
    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
    print("Shapes:")
    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)

    # 2) Class imbalance handling
    class_counts = y_train.value_counts().to_dict()
    # safer weights if classes are [0,1]
    neg_count = class_counts.get(0, 1)
    pos_count = class_counts.get(1, 1)
    pos_weight = neg_count / max(pos_count, 1)
    print(f"Class counts: {class_counts}")
    print(f"scale_pos_weight: {pos_weight:.4f}")

    # 3) CatBoost model (strong baseline for tabular data)
    model = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_SEED,
        verbose=100,
        scale_pos_weight=pos_weight,
        l2_leaf_reg=5,
        subsample=0.8,
        colsample_bylevel=0.8
    )

    # 4) Train with validation + early stopping
    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=100
    )

    # 5) Threshold tuning on validation
    val_prob = model.predict_proba(X_val)[:, 1]
    best_thr, best_val_acc = find_best_threshold(y_val.values, val_prob)
    print(f"\nBest validation threshold: {best_thr:.3f}")
    print(f"Best validation accuracy:  {best_val_acc:.4f}")

    # 6) Final test evaluation with tuned threshold
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_thr).astype(int)

    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_prob)

    print("\n================ TEST RESULTS ================")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test ROC AUC:  {test_auc:.4f}")
    print(f"Threshold:     {best_thr:.3f}")

    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, digits=4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_pred))

    # 7) Save model + threshold + metadata
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
        json.dump({"best_threshold": best_thr}, f, indent=2)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_col": TARGET_COL,
                "categorical_columns": cat_cols,
                "feature_columns": X_train.columns.tolist()
            },
            f,
            indent=2
        )

    print("\nSaved artifacts:")
    print(f"- {MODEL_PATH}")
    print(f"- {THRESHOLD_PATH}")
    print(f"- {META_PATH}")


if __name__ == "__main__":
    main()