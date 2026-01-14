import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
# Adjust this if your path is different
DATASET_DIR = r"datasets\TB_Chest_Radiography_Database"   # contains NORMAL and TB folders
CLINICAL_CSV = "clinical_data_complete.csv"

PROCESSED_ROOT = r"datasets\processed"
IMG_SIZE = (224, 224)
RANDOM_SEED = 42
# ----------------------------------------


def load_image_paths():
    rows = []
    # Walk through all subfolders
    for root, _, files in os.walk(DATASET_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                full = os.path.join(root, f)
                folder_name = os.path.basename(root).lower()
                # label: 1 = TB, 0 = Normal
                label = 1 if "tb" in folder_name else 0
                rows.append({"filepath": full, "filename": f, "label": label})

    df = pd.DataFrame(rows)
    print("Total images:", len(df))
    if len(df) == 0:
        print("WARNING: No images found. Check DATASET_DIR and folder names (NORMAL, TB).")
        return df

    print("Normal:", (df["label"] == 0).sum(), "TB:", (df["label"] == 1).sum())
    return df


def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    return img


def main():
    # Create output folders
    for split in ["train", "val", "test"]:
        for cls in ["normal", "tuberculosis"]:
            os.makedirs(os.path.join(PROCESSED_ROOT, split, cls), exist_ok=True)

    # 1) Load all images
    df_img = load_image_paths()
    if df_img is None or len(df_img) == 0:
        print("No images to process. Exiting.")
        return

    # 2) Stratified split: 70 / 15 / 15
    train_df, temp_df = train_test_split(
        df_img,
        test_size=0.30,
        stratify=df_img["label"],
        random_state=RANDOM_SEED,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_SEED,
    )

    splits = [("train", train_df), ("val", val_df), ("test", test_df)]

    # 3) Preprocess and save as .npy
    for split_name, df in splits:
        print(f"\nProcessing {split_name} ({len(df)})...")
        for _, row in df.iterrows():
            img = preprocess_image(row["filepath"])
            if img is None:
                continue
            cls = "tuberculosis" if row["label"] == 1 else "normal"
            base = os.path.splitext(row["filename"])[0]
            out_name = base + ".npy"
            out_path = os.path.join(PROCESSED_ROOT, split_name, cls, out_name)
            np.save(out_path, img)
        print("Done", split_name)

    # 4) Split clinical CSV to train/val/test (simple row-wise split)
    clinical = pd.read_csv(CLINICAL_CSV)
    clinical = clinical.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    n_train = len(train_df)
    n_val = len(val_df)

    train_clin = clinical.iloc[:n_train]
    val_clin = clinical.iloc[n_train:n_train + n_val]
    test_clin = clinical.iloc[n_train + n_val:]

    os.makedirs(r"datasets\clinical_data", exist_ok=True)
    train_clin.to_csv(r"datasets\clinical_data\train_clinical.csv", index=False)
    val_clin.to_csv(r"datasets\clinical_data\val_clinical.csv", index=False)
    test_clin.to_csv(r"datasets\clinical_data\test_clinical.csv", index=False)

    print("\nSaved clinical splits:")
    print(" train:", len(train_clin), "val:", len(val_clin), "test:", len(test_clin))


if __name__ == "__main__":
    main()
