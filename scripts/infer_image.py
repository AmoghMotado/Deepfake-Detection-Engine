import argparse, cv2, json
from dfdetect.infer import load_model, preprocess_bgr

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--image_size", type=int, default=160)
    args = ap.parse_args()

    m = load_model(args.weights)
    img = cv2.imread(args.image)

    # model outputs P(real) -> invert for fake probability
    x = preprocess_bgr(img, args.image_size)
    p_real = float(m.predict(x, verbose=0)[0][0])
    p_fake = 1.0 - p_real

    print(json.dumps({
        "path": args.image,
        "real_prob": p_real,
        "fake_prob": p_fake
    }))
