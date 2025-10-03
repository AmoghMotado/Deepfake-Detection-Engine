import argparse, cv2, json, numpy as np
from dfdetect.infer import load_model, preprocess_bgr, aggregate
from dfdetect.data_videos import frames_from_video

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--image_size", type=int, default=160)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--agg", type=str, default="topk_mean")
    ap.add_argument("--topk", type=float, default=0.3)
    args = ap.parse_args()
    m = load_model(args.weights)
    frames = frames_from_video(args.video, frames_per_video=args.frames, strategy="uniform", face=True)
    scores = []
    for f in frames:
        x = preprocess_bgr(f, args.image_size)
        p = float(m.predict(x, verbose=0)[0][0])
        scores.append(p)
    vid_score = aggregate(scores, mode=args.agg, topk=args.topk)
    print(json.dumps({"path": args.video, "frame_scores": scores, "video_score": vid_score}))