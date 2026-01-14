import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
TEST_IMG_DIR = r"datasets/processed/test"
TEST_CLIN_CSV = r"datasets/clinical_data/test_clinical.csv"
MODEL_PATH = "miliarytb_best.h5"
BATCH_SIZE = 32
# ----------------------------------------


def load_npy_image(path):
    arr = np.load(path)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    return arr


def make_test_dataframe():
    clin = pd.read_csv(TEST_CLIN_CSV).reset_index(drop=True)

    # ensure clinical data has NO 'label' column to avoid duplicates
    if "label" in clin.columns:
        clin = clin.drop(columns=["label"])

    rows = []
    for cls in ["normal", "tuberculosis"]:
        cls_dir = os.path.join(TEST_IMG_DIR, cls)
        label = 0 if cls == "normal" else 1
        for f in os.listdir(cls_dir):
            if f.endswith(".npy"):
                rows.append(
                    {
                        "npy_path": os.path.join(cls_dir, f),
                        "label": label,
                    }
                )

    df_img = pd.DataFrame(rows)

    # align lengths (simple trim to min)
    n = min(len(df_img), len(clin))
    df_img = df_img.iloc[:n].reset_index(drop=True)
    clin = clin.iloc[:n].reset_index(drop=True)

    merged = pd.concat([df_img, clin], axis=1)
    merged["label"] = merged["label"].astype("int32")
    return merged


def main():
    df = make_test_dataframe()
    print("Test samples:", len(df))

    model = keras.models.load_model(MODEL_PATH)

    X_img = []
    X_clin = []
    y_true = []

    for _, row in df.iterrows():
        img = load_npy_image(row["npy_path"])
        gender_val = 1.0 if row["gender"] == "F" else 0.0
        clin = np.array(
            [
                row["age"],
                row["leukocyte_count"],
                row["alt_level"],
                row["lymphocyte_pct"],
                gender_val,
            ],
            dtype="float32",
        )
        X_img.append(img)
        X_clin.append(clin)
        y_true.append(row["label"])

    X_img = np.array(X_img, dtype="float32")
    X_clin = np.array(X_clin, dtype="float32")
    y_true = np.array(y_true, dtype="int32")

    y_prob = model.predict(
        {"image_input": X_img, "clinical_input": X_clin},
        batch_size=BATCH_SIZE,
        verbose=1,
    )
    y_prob = y_prob.ravel()
    y_pred = (y_prob >= 0.5).astype("int32")

    # Confusion matrix and report
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)

    print(
        "\nClassification report:\n",
        classification_report(y_true, y_pred, target_names=["Normal", "TB"]),
    )

    # ROC-AUC
    try:
        auc = roc_auc_score(y_true, y_prob)
        print("ROC-AUC:", auc)
    except ValueError as e:
        print("ROC-AUC could not be computed:", e)
        auc = None

    # Plot ROC curve if possible
    if auc is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - MiliaryTB-Net")
        plt.legend()
        plt.tight_layout()
        plt.savefig("roc_curve.png", dpi=300)
        plt.close()
        print("Saved ROC curve to roc_curve.png")


if __name__ == "__main__":
    main()
