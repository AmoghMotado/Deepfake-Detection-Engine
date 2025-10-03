Deepfake Detection (with Web UI)
📖 Project Overview

Deepfake Detection Engine leverages deep learning to classify images and videos as AI-generated (fake) or authentic (real).
The project combines a TensorFlow/Keras-based detection model with a FastAPI-powered full-stack web application, enabling end-to-end workflows — from training and testing models to user-friendly real-time inference.

Deepfakes pose growing risks in areas like misinformation, cyber-security, and online identity theft. DFDetect helps identify manipulated media to enhance digital trust and security.

✨ Key Features

🔍 Deepfake Detection for Images – Classifies single images as “Original” or “AI-Generated”.

🎥 Deepfake Detection for Videos – Extracts frames → crops faces → scores per-frame → aggregates results with Top-K Mean logic for robust predictions.

📈 Unified Training Pipeline – Train models for both images and videos with YAML-based configurations.

📊 Metrics & Evaluation – Provides ROC, AUC, Accuracy, and calibrated thresholds for reliable decision-making.

🌐 Interactive Web Dashboard – User-friendly interface built with FastAPI + Jinja2 Templates.

🔐 Authentication – Supports standard Login/Signup and Google OAuth 2.0.

💾 MongoDB Integration – Stores user profiles and prediction history (images/videos).

📤 Media Upload & Compare – Upload and test individual images/videos, or compare two media files side-by-side.

⚡ Real-Time Inference – Drag-and-drop uploads with immediate classification and smooth progress animations.

💻 GPU-Ready – Optimized for faster model training and inference when using supported NVIDIA GPUs.

🚀 Tech Stack

Deep Learning: TensorFlow / Keras, OpenCV, NumPy

Backend: FastAPI, Pydantic, Uvicorn

Frontend: Jinja2 Templates, HTML5, CSS3, JS

Database: MongoDB (Motor async driver)

Authentication: OAuth 2.0 (Authlib), Session Middleware

Deployment-Ready: Uvicorn / Gunicorn compatible

Visualization & Metrics: Matplotlib, scikit-image

Environment Management: Python venv, dotenv

## 1. Pre-requisites

| Requirement | Recommended |
|-------------|-------------|
| **OS** | Windows 10/11 (64-bit) or Linux |
| **Python** | 3.10 × 64-bit |
| **Git** | for pulling code / version control |
| **MongoDB** | ≥ 5 .x running locally at `mongodb://localhost:27017` |
| **GPU** *(optional)* | Any NVIDIA GPU ≥ 2 GB VRAM (otherwise CPU will be used) |

> ⚠️ On Windows keep the project in a **short path with no spaces** (e.g. `D:\dfdetect`) to avoid long-path issues during TensorFlow install.

---

## 2. Clone / Unzip the Project

```bash
# if you have the ZIP
unzip dfdetect_updated.zip -d D:/dfdetect
cd D:/dfdetect

# or if you host on GitHub
git clone https://github.com/you/dfdetect.git
cd dfdetect
```

---

## 3. Create and Activate a Virtual Environment

### Windows (PowerShell)
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Linux / macOS or Git-Bash on Windows
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

---

## 4. Install Dependencies

```bash
pip install --no-cache-dir -r requirements.txt
```

> If you hit a TensorFlow long-path error on Windows, move the project to a shorter folder (e.g. `D:\dfdetect`) and retry.

---

## 5. (Recommended) GPU / CPU Thread Tweaks

```bash
# optional but helps prevent OOM on small GPUs
export TF_FORCE_GPU_ALLOW_GROWTH=true            # bash / Linux
$env:TF_FORCE_GPU_ALLOW_GROWTH="true"            # PowerShell

# tune CPU threads if training on CPU
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2
```

---

## 6. Dataset Preparation

Project expects:

```
data/
 ├─ images/
 │   ├─ train/{real,fake}/
 │   └─ val/{real,fake}/
 └─ videos/
     ├─ train/{real,fake}/
     └─ val/{real,fake}/
```

---

### 6-A. Images → **140k Real vs Fake Faces**

1. Download the dataset from Kaggle:  
   <https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces>

2. Unzip so you have:  
   `data_raw/140k_faces/real_vs_fake/{real,fake}`

3. Split into train/val and copy:

```bash
python scripts/prepare_140k_faces.py        --src data_raw/140k_faces/real_vs_fake        --out data/images        --val_split 0.2 --copy
```

---

### 6-B. Videos → **SDFVD**

1. Extract the downloaded SDFVD dataset somewhere.

2. Prepare it:

```bash
python scripts/prepare_sdfvd.py        --sdfvd_dir /path/to/SDFVD        --out_root data/videos        --val_split 0.2 --copy
```

---

## 7. Train Models

### 7-A. Train **Image Model**

Edit **`configs/baseline.yaml`** (suggested defaults):

```yaml
image_size: 160
batch_size: 16        # reduce to 8 on low-RAM laptops
epochs: 12            # raise/lower as desired
```

Train:
```bash
python scripts/train_images.py --config configs/baseline.yaml
```

Artifacts → `runs/images_baseline/{best.keras, final.keras}`

Evaluate & get optimal threshold:
```bash
python scripts/eval.py        --weights runs/images_baseline/best.keras        --real_dir data/images/val/real        --fake_dir data/images/val/fake
```

---

### 7-B. Train **Video Model**

Edit **`configs/videos.yaml`** (suggested defaults):

```yaml
image_size: 160
batch_size: 8          # reduce to 4 on low-RAM laptops
epochs: 6
frames_per_video: 12   # sample frames per clip
aggregation: topk_mean
topk: 0.3
```

Train:
```bash
python scripts/train_videos.py --config configs/videos.yaml
```

Artifacts → `runs/videos_baseline/{best.keras, final.keras}`

---

## 8. Quick Inference Tests

**Single Image**
```bash
python scripts/infer_image.py        --weights runs/images_baseline/best.keras        --image path/to/test.jpg --image_size 160
```

**Single Video**
```bash
python scripts/infer_video.py        --weights runs/videos_baseline/best.keras        --video path/to/test.mp4 --frames 12        --agg topk_mean --topk 0.3
```

---

## 9. Configure Environment Variables for Web App

Create a file **`.env`** in the project root:

```env
MONGO_URL=mongodb://localhost:27017
DB_NAME=dfdetect
SECRET_KEY=change-this-secret
WEIGHTS_PATH=runs/images_baseline/best.keras   # or use the video model
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
```

> The app auto-loads `.env` via `python-dotenv`.

---

## 10. Run the Web App

```bash
uvicorn webapp.main:app --reload
```

Open <http://localhost:8000>  

Log in with Google → **Dashboard** →  
**Upload Photo • Upload Video • Compare Photos • Compare Videos**

---

## 11. Project Structure Recap
```
dfdetect/
 ├─ configs/…yaml
 ├─ data/images/{train,val}/{real,fake}
 ├─ data/videos/{train,val}/{real,fake}
 ├─ runs/…
 ├─ scripts/…
 ├─ webapp/main.py (+ templates, static)
 ├─ .env
 └─ requirements.txt
```

---

## 12. Troubleshooting

| Issue | Fix |
|-------|-----|
| `OSError … envoy … upb_minitable.h` on TF install | Move project to short path (e.g. `D:\dfdetect`), re-create venv |
| `MemoryError / OOM` during training | Lower `batch_size` (8→4), reduce `image_size` (160→128), close other apps |
| `ModuleNotFoundError` for CUDA libs | TF will fall back to CPU; install GPU-TF only if you have compatible NVIDIA CUDA |
| Google OAuth login fails | Check **GOOGLE_CLIENT_ID / SECRET** in `.env` match console credentials & callback URL is `http://localhost:8000/auth` |
| Long training time | Use GPU if available; otherwise increase CPU threads as in step 5 |

---

## 13. License / Credits
- Training/inference code: this repo (MIT-style license)
- Datasets: respect original dataset licenses (Kaggle 140k Real vs Fake Faces, SDFVD, etc.)