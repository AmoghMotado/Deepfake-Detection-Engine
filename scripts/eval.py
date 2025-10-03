import argparse, glob, os, json, numpy as np, cv2
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
from dfdetect.infer import load_model, preprocess_bgr

def eval_images(weights, real_dir, fake_dir, image_size=160, threshold=0.5):
    m = load_model(weights)
    y_true = []; y_score = []
    for root, lbl in [(real_dir,0),(fake_dir,1)]:
        for p in glob.glob(os.path.join(root, "*")):
            img = cv2.imread(p)
            if img is None: continue
            x = preprocess_bgr(img, image_size)
            s = float(m.predict(x, verbose=0)[0][0])
            y_true.append(lbl); y_score.append(s)
    auc = roc_auc_score(y_true, y_score)
    # choose thresh by Youden
    fpr, tpr, thr = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    best_thr = float(thr[idx])
    ap = average_precision_score(y_true, y_score)
    y_pred = (np.array(y_score) >= best_thr).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {"auc": float(auc), "ap": float(ap), "best_threshold": best_thr, "confusion_matrix": cm}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--fake_dir", required=True)
    ap.add_argument("--image_size", type=int, default=160)
    args = ap.parse_args()
    out = eval_images(args.weights, args.real_dir, args.fake_dir, args.image_size)
    print(json.dumps(out, indent=2))