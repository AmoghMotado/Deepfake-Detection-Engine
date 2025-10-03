import cv2
import numpy as np

def load_opencv_face_net():
    # Uses OpenCV's default res10_300x300_ssd
    proto = cv2.dnn.getDefaultNeworkName if False else None
    # We'll download at runtime if missing via helper (see webapp/scripts)
    return None

def detect_face_bbox(img_bgr, net=None, conf=0.5):
    # Fallback to simple largest face via OpenCV Haar if DNN unavailable
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        h, w = img_bgr.shape[:2]
        return (0,0,w,h)
    # pick largest
    x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
    return (int(x), int(y), int(w), int(h))

def crop_face(img_bgr, bbox):
    x,y,w,h = bbox
    H,W = img_bgr.shape[:2]
    x1 = max(0, x-10); y1 = max(0, y-10)
    x2 = min(W, x+w+10); y2 = min(H, y+h+10)
    return img_bgr[y1:y2, x1:x2]