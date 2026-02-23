import os
import io
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from PIL import Image
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Put it in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-flash-latest"

app = FastAPI(title="tandangi-ai", version="0.1.0")

class GeminiRawResponse(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME}

@app.post("/analyze", response_model=GeminiRawResponse)
async def analyze(
    image: UploadFile = File(...),
    prompt: str = Form("이 사진 속 음식 메뉴를 한국어로 5개 이하로 추정해줘. 각 항목은 한 줄로.")
):
    content = await image.read()
    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    ex400, detail="Invalid image file")

    try:
        resp = client.models.generate_content(model=MODEL_NAME, contents=[prompt, pil_img])
        text = getattr(resp, "text", None) or str(resp)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")
