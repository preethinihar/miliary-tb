import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------- CONFIG ----------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

TRAIN_IMG_DIR = r"datasets/processed/train"
VAL_IMG_DIR = r"datasets/processed/val"

TRAIN_CLIN_CSV = r"datasets/clinical_data/train_clinical.csv"
VAL_CLIN_CSV = r"datasets/clinical_data/val_clinical.csv"
# ----------------------------------------


def load_npy_image(path):
    arr = np.load(path)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    return arr


def make_dataframe(img_root, clin_csv):
    clin = pd.read_csv(clin_csv).reset_index(drop=True)

    # ensure clinical data has NO 'label' column to avoid duplicates
    if "label" in clin.columns:
        clin = clin.drop(columns=["label"])

    # collect all npy paths and labels from processed folders
    rows = []
    for cls in ["normal", "tuberculosis"]:
        cls_dir = os.path.join(img_root, cls)
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

    # side‑by‑side merge -> only one clean 'label' column (from df_img)
    merged = pd.concat([df_img, clin], axis=1)

    # ensure label is int 0/1 (1D Series)
    merged["label"] = merged["label"].astype("int32")
    return merged


def df_to_tf_dataset(df, shuffle=True):
    # clinical features: age, leukocyte_count, alt_level, lymphocyte_pct, gender (M/F -> 0/1)
    def gen():
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
            label = np.float32(row["label"])
            # one scalar label, not a vector
            yield {"image_input": img, "clinical_input": clin}, label

    output_signature = (
        {
            "image_input": tf.TensorSpec(shape=(224, 224, 1), dtype=tf.float32),
            "clinical_input": tf.TensorSpec(shape=(5,), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model():
    # image branch (ResNet50 backbone)
    img_in = keras.Input(shape=(224, 224, 1), name="image_input")
    x = layers.Conv2D(3, (1, 1), padding="same")(img_in)  # convert 1ch -> 3ch
    base = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg",
    )
    base.trainable = False  # freeze backbone initially for mini‑project
    x = base(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # clinical branch
    clin_in = keras.Input(shape=(5,), name="clinical_input")
    y = layers.Dense(32, activation="relu")(clin_in)
    y = layers.Dropout(0.3)(y)

    # fusion
    z = layers.Concatenate()([x, y])
    z = layers.Dense(64, activation="relu")(z)
    z = layers.Dropout(0.4)(z)
    out = layers.Dense(1, activation="sigmoid", name="diagnosis")(z)

    model = keras.Model(inputs=[img_in, clin_in], outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.Precision(name="precision"),
        ],
    )
    return model


def main():
    print("Loading metadata...")
    train_df = make_dataframe(TRAIN_IMG_DIR, TRAIN_CLIN_CSV)
    val_df = make_dataframe(VAL_IMG_DIR, VAL_CLIN_CSV)

    print("Train samples:", len(train_df), "Val samples:", len(val_df))

    train_ds = df_to_tf_dataset(train_df, shuffle=True)
    val_ds = df_to_tf_dataset(val_df, shuffle=False)

    model = build_model()
    model.summary()

    # fixed class weights: TB is minority (approx 5x)
    class_weight = {0: 1.0, 1: 2.0}

    print("Using fixed class weights:", class_weight)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=5, mode="max", restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            "miliarytb_best.h5", monitor="val_auc", mode="max", save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    model.save("miliarytb_final.h5")
    print("Saved models: miliarytb_best.h5, miliarytb_final.h5")


if __name__ == "__main__":
    main()
