import argparse, yaml, os, cv2, numpy as np, glob
from dfdetect.models import build_model
from tensorflow import keras

def iter_video_frames_as_images(root, split, image_size, frames_per_video, face_detection=True):
    # expects directory structure: root/split/{real,fake}/*.mp4
    for label_name in ["real","fake"]:
        lbl = 0 if label_name=="real" else 1
        for vpath in glob.glob(os.path.join(root, split, label_name, "*")):
            cap = cv2.VideoCapture(vpath)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            idxs = np.linspace(0, total-1, num=frames_per_video, dtype=int)
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok: continue
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (image_size, image_size)).astype("float32")/255.0
                yield img, lbl
            cap.release()

def tf_dataset_from_generator(root, split, image_size, frames_per_video, batch_size):
    import tensorflow as tf
    def gen():
        for x,y in iter_video_frames_as_images(root, split, image_size, frames_per_video):
            yield x,y
    ds = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=(image_size,image_size,3), dtype=tf.float32),
                                                               tf.TensorSpec(shape=(), dtype=tf.int32)))
    ds = ds.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def main(cfg):
    model = build_model(input_shape=(cfg["image_size"], cfg["image_size"], 3), backbone=cfg.get("model","mobilenet_v2"))
    os.makedirs(cfg["save_dir"], exist_ok=True)
    train_ds = tf_dataset_from_generator(cfg["train_dir"], "train", cfg["image_size"], cfg["frames_per_video"], cfg["batch_size"])
    val_ds = tf_dataset_from_generator(cfg["val_dir"], "val", cfg["image_size"], cfg["frames_per_video"], cfg["batch_size"])
    ckpt = keras.callbacks.ModelCheckpoint(os.path.join(cfg["save_dir"], "best.keras"), monitor="val_auc", mode="max", save_best_only=True)
    early = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_auc", mode="max")
    hist = model.fit(train_ds, validation_data=val_ds, epochs=cfg["epochs"], callbacks=[ckpt, early])
    model.save(os.path.join(cfg["save_dir"], "final.keras"))
    print("Saved to", cfg["save_dir"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    main(cfg)