import os
import io
import json
import re
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Put it in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

# ✅ 너가 성공했던 모델명으로 유지
MODEL_NAME = "gemini-flash-latest"

app = FastAPI(title="tandangi-ai", version="0.1.0")


class MenuItem(BaseModel):
    name: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class MenuItemsResponse(BaseModel):
    items: List[MenuItem]


JSON_PROMPT = """
너는 음식 인식 도우미야.
사용자가 보낸 사진을 보고 음식 메뉴 후보를 최대 8개까지 한국어로 추정해줘.
반드시 아래 JSON만 출력해. 다른 텍스트(설명e는 추정 확신 정도(0~1).
- items가 비어있으면 [].
""".strip()


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME}


@app.post("/analyze", response_model=MenuItemsResponse)
async def analyze(
    image: UploadFile = File(...),
    prompt: str = Form(JSON_PROMPT),
):
    content = await image.read()

    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, pil_img],
        )
        text = getattr(resp, "text", None) or str(resp)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")

    try:
        data = json.loads(text)
    except Exception:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini returned non-JSON: {text[:300]}",
        )

    try:
        odel_validate(data)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Invalid JSON schema from Gemini: {e}. raw={text[:300]}",
        )
