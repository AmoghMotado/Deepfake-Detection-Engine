import cv2, numpy as np, os
from .utils_face import detect_face_bbox, crop_face

def sample_frames_uniform(cap, k=16):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(0, total-1, num=k, dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok: frames.append(frame)
    return frames

def frames_from_video(video_path, frames_per_video=16, strategy="uniform", face=True, min_conf=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    if strategy == "uniform":
        raw = sample_frames_uniform(cap, frames_per_video)
    else:
        raw = sample_frames_uniform(cap, frames_per_video)
    cap.release()
    out = []
    for f in raw:
        if face:
            bbox = detect_face_bbox(f, conf=min_conf)
            f = crop_face(f, bbox)
        out.append(f)
    return out