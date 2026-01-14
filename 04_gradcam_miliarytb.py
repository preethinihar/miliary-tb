import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import cv2

# ---------------- CONFIG ----------------
MODEL_PATH = "miliarytb_best.h5"
TEST_IMG_DIR = r"datasets/processed/test"
TEST_CLIN_CSV = r"datasets/clinical_data/test_clinical.csv"

OUT_DIR = "gradcam_outputs"
# ----------------------------------------


def load_npy_image(path):
    arr = np.load(path)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    return arr


def make_test_dataframe():
    clin = pd.read_csv(TEST_CLIN_CSV).reset_index(drop=True)
    if "label" in clin.columns:
        clin = clin.drop(columns=["label"])

    rows = []
    for cls in ["normal", "tuberculosis"]:
        cls_dir = os.path.join(TEST_IMG_DIR, cls)
        label = 0 if cls == "normal" else 1
        for f in os.listdir(cls_dir):
            if f.endswith(".npy"):
                rows.append({"npy_path": os.path.join(cls_dir, f), "label": label})

    df_img = pd.DataFrame(rows)
    n = min(len(df_img), len(clin))
    df_img = df_img.iloc[:n].reset_index(drop=True)
    clin = clin.iloc[:n].reset_index(drop=True)
    merged = pd.concat([df_img, clin], axis=1)
    merged["label"] = merged["label"].astype("int32")
    return merged


def make_gradcam_heatmap(img_array, clinical_array, model, pred_index=None):
    """
    img_array: (1, 224, 224, 1)
    clinical_array: (1, 5)
    """

    # get the internal ResNet50 submodel and its last conv layer
    base_model = model.get_layer("resnet50")
    last_conv_layer = base_model.get_layer("conv5_block3_out")  # inside ResNet50[web:166]

    # build a model that outputs both conv maps and final prediction
    # need outputs from base_model and from full model
    conv_output = last_conv_layer.output
    # map: [image_input, clinical_input] -> [conv_output, final_output]
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[conv_output, model.output],
    )

    # build input dict exactly like during training
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    clin_tensor = tf.convert_to_tensor(clinical_array, dtype=tf.float32)
    input_dict = {
        "image_input": img_tensor,
        "clinical_input": clin_tensor,
    }

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_dict)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)  # same shape as conv_outputs
    # average gradients over spatial dims
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    conv_outputs = conv_outputs[0]  # (H, W, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (H, W, 1)
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def save_gradcam_overlay(img_arr, heatmap, out_path, alpha=0.4):
    # img_arr: (224,224,1) in [0,1]
    img = (img_arr * 255).astype("uint8")
    img = np.repeat(img, 3, axis=-1)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    cv2.imwrite(out_path, superimposed_img)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading model...")
    model = keras.models.load_model(MODEL_PATH)

    print("Preparing test dataframe...")
    df = make_test_dataframe()
    print("Total test samples for Grad-CAM:", len(df))

    # pick a few examples: 2 Normal + 2 TB
    subset = pd.concat(
        [
            df[df["label"] == 0].head(2),
            df[df["label"] == 1].head(2),
        ],
        axis=0,
    ).reset_index(drop=True)

    for idx, row in subset.iterrows():
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

        img_batch = np.expand_dims(img, axis=0)
        clin_batch = np.expand_dims(clin, axis=0)

        heatmap = make_gradcam_heatmap(img_batch, clin_batch, model, pred_index=None)

        label = "Normal" if row["label"] == 0 else "TB"
        base_name = os.path.splitext(os.path.basename(row["npy_path"]))[0]
        out_path = os.path.join(OUT_DIR, f"{base_name}_gradcam_{label}.png")

        save_gradcam_overlay(img, heatmap, out_path)
        print("Saved Grad-CAM:", out_path)


if __name__ == "__main__":
    main()
