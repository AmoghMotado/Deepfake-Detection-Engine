from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf

def make_image_datasets(train_dir, val_dir, image_size=160, batch_size=32, seed=42):
    train_ds = image_dataset_from_directory(train_dir, labels="inferred", label_mode="binary",
                                            image_size=(image_size, image_size), batch_size=batch_size, seed=seed)
    val_ds = image_dataset_from_directory(val_dir, labels="inferred", label_mode="binary",
                                          image_size=(image_size, image_size), batch_size=batch_size, seed=seed)
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1024).prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds