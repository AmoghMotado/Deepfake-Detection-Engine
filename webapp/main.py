import os, io, uuid, json, cv2, numpy as np
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from motor.motor_asyncio import AsyncIOMotorClient
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from dfdetect.infer import load_model, preprocess_bgr, aggregate
from dfdetect.data_videos import frames_from_video

load_dotenv()
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "dfdetect")
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "runs/images_baseline/best.keras")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")
templates = Jinja2Templates(directory="webapp/templates")

mongo = AsyncIOMotorClient(MONGO_URL)
db = mongo[DB_NAME]

oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID", ""),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET", ""),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"}
)

_model = None
def get_model():
    global _model
    if _model is None:
        from tensorflow import keras
        _model = keras.models.load_model(WEIGHTS_PATH)
    return _model

def current_user(request: Request):
    return request.session.get("user")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    user = current_user(request)
    if user:
        return RedirectResponse("/dashboard")
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for("auth")
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth")
async def auth(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user = token.get("userinfo")
    request.session["user"] = dict(user)
    await db.users.update_one({"sub": user["sub"]}, {"$set": dict(user)}, upsert=True)
    return RedirectResponse("/dashboard")

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = current_user(request)
    if not user:
        return RedirectResponse("/")
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": user})

# ---------------- IMAGE INFERENCE -----------------
@app.post("/api/predict-image")
async def predict_image(request: Request, file: UploadFile = File(...)):
    """
    For the current image model:
      output p = P(real)
      convert to fake_prob = 1 - p
    """
    user = current_user(request)
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    m = get_model()
    x = preprocess_bgr(img, 160)
    p_real = float(m.predict(x, verbose=0)[0][0])
    p_fake = 1.0 - p_real

    rec = {
        "kind": "image",
        "filename": file.filename,
        "prob_fake": p_fake,
        "prob_real": p_real,
        "user": user.get("email") if user else None
    }
    await db.predictions.insert_one(rec)

    label = "AI Generated" if p_fake >= 0.5 else "Original"
    return {"fake_prob": p_fake, "label": label}

# ---------------- VIDEO INFERENCE -----------------
@app.post("/api/predict-video")
async def predict_video(request: Request, file: UploadFile = File(...)):
    user = current_user(request)
    buf = await file.read()
    tmp_path = f"/tmp/{uuid.uuid4()}.mp4"
    with open(tmp_path, "wb") as f:
        f.write(buf)

    frames = frames_from_video(tmp_path, frames_per_video=16,
                               strategy="uniform", face=True)
    m = get_model()
    scores = []
    for fr in frames:
        x = preprocess_bgr(fr, 160)
        p = float(m.predict(x, verbose=0)[0][0])   # video model outputs P(fake)
        scores.append(p)

    video_score = aggregate(scores, mode="topk_mean", topk=0.3)
    rec = {
        "kind": "video",
        "filename": file.filename,
        "video_score": video_score,
        "frame_scores": scores,
        "user": user.get("email") if user else None
    }
    await db.predictions.insert_one(rec)

    label = "AI Generated" if video_score >= 0.5 else "Original"
    os.remove(tmp_path)
    return {"video_score": video_score, "label": label}

# ---------------- COMPARE IMAGES -----------------
@app.post("/api/compare-images")
async def compare_images(request: Request, file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    Both outputs p = P(real) -> convert to fake_prob
    """
    m = get_model()
    def score(contents):
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        x = preprocess_bgr(img, 160)
        p_real = float(m.predict(x, verbose=0)[0][0])
        return 1.0 - p_real

    s1 = score(await file1.read())
    s2 = score(await file2.read())
    return {"image1_fake_prob": s1, "image2_fake_prob": s2}

# ---------------- COMPARE VIDEOS -----------------
@app.post("/api/compare-videos")
async def compare_videos(request: Request, file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    Video model already outputs P(fake), no inversion needed.
    """
    import tempfile
    def score(buf):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(buf); tmp.flush()
            frames = frames_from_video(tmp.name, frames_per_video=16,
                                       strategy="uniform", face=True)
        m = get_model()
        scores = [float(m.predict(preprocess_bgr(fr, 160), verbose=0)[0][0])
                  for fr in frames]
        return float(np.mean(sorted(scores)[-max(1, int(0.3*len(scores))):]))

    s1 = score(await file1.read())
    s2 = score(await file2.read())
    return {"video1_fake_prob": s1, "video2_fake_prob": s2}
