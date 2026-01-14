import numpy as np
import tensorflow as tf
from tensorflow import keras
import gradio as gr

MODEL_PATH = "miliarytb_best.h5"
IMG_SIZE = (224, 224)

print("Loading model...")
model = keras.models.load_model(MODEL_PATH)


def preprocess_image(image):
    img = np.array(image).astype("float32")

    if img.ndim == 3 and img.shape[-1] == 3:
        img = img.mean(axis=-1, keepdims=True)
    elif img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    img = tf.image.resize(img, IMG_SIZE).numpy()
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_miliary_tb(image):
    if image is None:
        return "Please upload a chest X-ray image."

    img_batch = preprocess_image(image)
    clin_batch = np.zeros((1, 5), dtype="float32")

    prob_tb = model.predict(
        {"image_input": img_batch, "clinical_input": clin_batch},
        verbose=0,
    ).ravel()[0]

    prob_percent = prob_tb * 100

    THRESHOLD = 0.62

    # >>> Correct direction <<<
    if prob_tb < THRESHOLD:
        label = "Miliary TB"
    else:
        label = "Normal TB"

    text = (
        f"Predicted category: {label}\n"
        
        f"Probability (%): {prob_percent:.1f}%"
    )

    return text


iface = gr.Interface(
    fn=predict_miliary_tb,
    inputs=gr.Image(type="pil", label="Upload Chest X-ray"),
    outputs=gr.Textbox(label="Prediction"),
    title="MiliaryTB-Net Prototype Interface",
    description=(
        "Upload a chest X-ray. For demo, the binary TB probability is mapped to "
        "Normal TB or Miliary TB using a tuned threshold."
    ),
)

if __name__ == "__main__":
    iface.launch(share=True)
