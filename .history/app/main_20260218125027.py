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

# ✅ 너가 이전에 성공했던 모델명으로 설정
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
반드시 아래 JSON만 출력해. 다른 텍스트(설명, 코드블록, 문장) 절대 금지.

스키마:
{
  "items": [
    {"name": string, "confidence": number (0.0~1.0)}
  ]
}

name 규칙:
- 음식명만. 불필요한 수식어 제거.
- 너무 유사한 항목(예: 곱창전골/소곱창전골/대창전골)은 가능하면 하나로 통합하거나, 여러 개를 넣을 경우 confidence를 분산해.

confidence 규칙(중요):
- 1.0 사용 금지. 범위는 0.20~0.95만 사용해.
- 0.90 이상은 최대 1개만 허용해.
- items는 confidence 내림차순으로 정렬해.
- 서로 매우 유사한 항목이 여러 개면(대체 후보) 각각 0.90 이상이 되면 안 되고, 0.80 이하로 분산해.
- 확신이 낮으면 과감히 0.30~0.60대로 내려.

items가 없으면 {"items": []} 를 출력해.
""".strip()



def try_parse_json(s: str):
    # 1) 그대로 파싱
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) ```json ... ``` 코드블록 제거 후 파싱
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 3) 문자열 안에서 { ... } 구간만 추출해서 파싱
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return None


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME}


@app.post("/analyze", response_model=MenuItemsResponse)
async def analyze(
    image: UploadFile = File(...),
    prompt: str = Form(JSON_PROMPT),
):
    content = await image.read()

    # 이미지 유효성 체크
    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 1차 Gemini 호출 (이미지 포함)
    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, pil_img],
        )
        text = getattr(resp, "text", None) or str(resp)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")

    data = try_parse_json(text)

    # JSON이 아니면 2차 "변환기" 호출 (텍스트만)
    if data is None:
        repair_prompt = f"""
너는 변환기야. 아래 텍스트에서 음식 메뉴 후보를 추출해서 반드시 JSON으로만 출력해.
다른 문장/설명 절대 금지.

출력 JSON 스키마:
{{"items":[{{"name":string,"confidence":number(0~1)}}]}}

텍스트:
{text}
""".strip()

        try:
            resp2 = client.models.generate_content(
                model=MODEL_NAME,
                contents=[repair_prompt],
            )
            text2 = getattr(resp2, "text", None) or str(resp2)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Gemini repair call failed: {e}")

        data = try_parse_json(text2)

    if data is None:
        raise HTTPException(status_code=502, detail=f"Gemini returned non-JSON: {text[:300]}")

    # 스키마 검증
    try:
        return MenuItemsResponse.model_validate(data)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Invalid JSON schema from Gemini: {e}. raw={str(data)[:300]}")
