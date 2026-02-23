import os
import io

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from PIL import Image
from typing import List
from pydantic import BaseModel, Field
import json
from fastapi import Body
import google.generativeai as genai

class MenuItem(BaseModel):
    name: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

class MenuItemsResponse(BaseModel):
    items: List[MenuItem]

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


JSON_PROMPT = """
너는 음식 인식 도우미야.
사용자가 보낸 사진을 보고 음식 메뉴 후보를 최대 8개까지 한국어로 추정해줘.
반드시 아래 JSON만 출력해. 다른 텍스트(설명, 코드블록, 문장) 절대 금지.

스키마:
{
  "items": [
    {"name": string, "confidence": number (0.0~1.0)}
  ]
}

규칙:
- name은 음식명만. 불필요한 수식어 제거.
- confidence는 추정 확신 정도(0~1).
- items가 비어있으면 [].
"""

@app.post("/analyze", response_model=MenuItemsResponse)
async def analyze(
    image: UploadFile = File(...),
    prompt: str = Form(JSON_PROMPT)
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

    # Gemini가 준 text를 JSON으로 파싱
    try:
        data = json.loads(text)
        return data
    except Exception:
        raise HTTPException(status_code=502, detail=f"Gemini returned non-JSON: {text[:200]}")
