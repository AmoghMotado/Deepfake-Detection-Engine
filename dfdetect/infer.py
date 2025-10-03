import numpy as np, cv2, os, json
from tensorflow import keras

def load_model(weights_path):
    return keras.models.load_model(weights_path)

def preprocess_bgr(img_bgr, image_size=160):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, (image_size, image_size)).astype("float32") / 255.0
    return np.expand_dims(img, 0)

def predict_image(model, img_bgr, image_size=160):
    x = preprocess_bgr(img_bgr, image_size)
    p = float(model.predict(x, verbose=0)[0][0])
    return p  # probability of FAKE if trained that way

def aggregate(scores, mode="topk_mean", topk=0.3):
    s = np.array(scores, dtype=float)
    if mode == "mean":
        return float(s.mean())
    if mode == "max":
        return float(s.max())
    if mode == "topk_mean":
        k = max(1, int(len(s)*topk))
        return float(np.sort(s)[-k:].mean())
    return float(s.mean())