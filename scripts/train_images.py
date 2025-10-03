import argparse, yaml, os
from dfdetect.models import build_model
from dfdetect.data_images import make_image_datasets
from tensorflow import keras

def main(cfg):
    model = build_model(input_shape=(cfg["image_size"], cfg["image_size"], 3), backbone=cfg.get("model","mobilenet_v2"))
    train_ds, val_ds = make_image_datasets(cfg["train_dir"], cfg["val_dir"], cfg["image_size"], cfg["batch_size"], cfg["seed"])
    ckpt = keras.callbacks.ModelCheckpoint(os.path.join(cfg["save_dir"], "best.keras"), monitor="val_auc", mode="max", save_best_only=True)
    early = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_auc", mode="max")
    os.makedirs(cfg["save_dir"], exist_ok=True)
    hist = model.fit(train_ds, validation_data=val_ds, epochs=cfg["epochs"], callbacks=[ckpt, early])
    model.save(os.path.join(cfg["save_dir"], "final.keras"))
    print("Saved to", cfg["save_dir"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    main(cfg)