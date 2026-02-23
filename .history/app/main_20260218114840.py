import os
import io

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from PIL import Image

import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Put it in .env")

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-flash-latest"
model = genai.GenerativeModel(MODEL_NAME)

app = FastAPI(title="tandangi-ai", version="0.1.0")


class GeminiRawResponse(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME}


@app.post("/analyze", response_model=GeminiRawResponse)
async def analyze(
    image: UploadFile = File(...),
    prompt: str = Form("이 사진에 있는 음식 메뉴를 한국어로 간단히 나열해줘.")
):
    content = await image.read()

    # 이미지 유효성 체크
    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Gemini 호출
    try:
        resp = model.generate_content([prompt, pil_img])
        text = getattr(resp, "text", None) or str(resp)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")

    return GeminiRawResponse(text=text)
