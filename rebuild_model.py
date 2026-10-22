from pathlib import Path

import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best_model_60f_gru64_stage3.keras"
)

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False
)

print("Model dimuat:", MODEL_PATH.name)
print("Input shape:", model.input_shape)