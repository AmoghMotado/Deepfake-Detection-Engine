from tensorflow.keras import layers, models
from tensorflow import keras
import tensorflow as tf

def build_model(input_shape=(160,160,3), backbone="mobilenet_v2"):
    if backbone == "mobilenet_v2":
        base = keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=None)
    else:
        base = keras.applications.EfficientNetV2B0(input_shape=input_shape, include_top=False, weights=None)

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(base.input, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc"), "accuracy"])
    return model