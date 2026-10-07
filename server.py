"""FastAPI wrapper for the Bilingual SER models. Run: uvicorn server:app --reload"""
import os, subprocess, tempfile, time
import imageio_ffmpeg
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

app = FastAPI(title="Bilingual SER API")
HERE = os.path.dirname(os.path.abspath(__file__))


def to_wav(src, sr):
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", src, "-ar", str(sr), "-ac", "1", out]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError("FFmpeg could not read this audio file.")
    return out


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "frontend", "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), language: str = Form("Kannada")):
    src = wav = None
    try:
        ext = os.path.splitext(file.filename or "")[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            f.write(await file.read())
            src = f.name
        t0 = time.time()
        if language.lower() == "english":
            from ser_english import predict_english  # lazy: loads only English model
            wav = to_wav(src, 16000)
            emotion, probs = predict_english(wav)
            probs = {str(k): float(v) for k, v in probs.items()}
        else:
            from inference import predict_emotion, CLASSES  # lazy: loads only Kannada model
            wav = to_wav(src, 22050)
            emotion, p = predict_emotion(wav)
            probs = {str(c): float(v) for c, v in zip(CLASSES, p)}
        return {"emotion": str(emotion), "probs": probs,
                "latency": round(time.time() - t0, 2), "language": language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in (src, wav):
            if p and os.path.exists(p):
                try: os.remove(p)
                except OSError: pass