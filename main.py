from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import requests
import os
import io

app = FastAPI()

MODEL_URL = "https://huggingface.co/Sahil200217/pneumonia-detection-model/resolve/main/hybrid_model.h5"
MODEL_PATH = "hybrid_model.h5"

if not os.path.exists(MODEL_PATH):
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

model = load_model(MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>PneumoScan AI</h2>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file"/>
        <button type="submit">Predict</button>
    </form>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        return {"result": "Pneumonia Detected", "confidence": float(pred)}
    else:
        return {"result": "Normal", "confidence": float(1 - pred)}
