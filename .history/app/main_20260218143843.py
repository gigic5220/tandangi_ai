import os
import io
import json
import re
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

from google import genai

load_dotenv()

# ======================
# ENV
# ======================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Put it in .env")

# 너가 성공했던 모델명 유지/수정 가능
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip()

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="tandangi-ai", version="0.3.0")


# ======================
# Response Models
# ======================
class NutrientEstimate(BaseModel):
    value: float
    range: List[float] = Field(default_factory=list, description="[min, max]")
    unit: str


class VisionNutrients(BaseModel):
    kcal: NutrientEstimate
    carb_g: NutrientEstimate
    protein_g: NutrientEstimate
    fat_g: NutrientEstimate
    sugar_g: NutrientEstimate
    sodium_mg: NutrientEstimate


class IngredientsBlock(BaseModel):
    visible: List[str] = Field(default_factory=list, description="사진에서 '보이는' 재료/구성")
    assumed: List[str] = Field(default_factory=list, description="레시피/관행 기반으로 '추정'한 재료/양념")


class VisionBlock(BaseModel):
    ingredients: IngredientsBlock
    estimated_nutrients: VisionNutrients
    assumptions: List[str] = Field(default_factory=list)


class FinalBlock(BaseModel):
    source: str  # always "vision_estimate" in this version
    nutrients: Dict[str, Any]
    notes: List[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    items: List[str] = Field(default_factory=list)
    vision: VisionBlock
    final: FinalBlock


# ======================
# Prompt
# ======================
JSON_PROMPT = """
너는 음식/영양 분석 도우미야.
사용자가 보낸 "음식 사진"을 보고, 아래 JSON 스키마를 정확히 따라 결과를 만들어.

반드시 JSON만 출력해. 다른 문장/설명/코드블록 절대 금지.

스키마:
{
  "items": [string],  // 음식 후보명(가장 가능성 높은 순서)
  "vision": {
    "ingredients": {
      "visible": [string], // 사진에서 실제로 보이는 재료/구성(최대 10개)
      "assumed": [string]  // 사진만으로 확정은 어렵지만 레시피/관행상 들어갈 가능성이 높은 재료/양념(최대 8개)
    },
    "estimated_nutrients": {
      "kcal":      {"value": number, "range": [number, number], "unit": "kcal"},
      "carb_g":    {"value": number, "range": [number, number], "unit": "g"},
      "protein_g": {"value": number, "range": [number, number], "unit": "g"},
      "fat_g":     {"value": number, "range": [number, number], "unit": "g"},
      "sugar_g":   {"value": number, "range": [number, number], "unit": "g"},
      "sodium_mg": {"value": number, "range": [number, number], "unit": "mg"}
    },
    "assumptions": [string] // 추정에 사용한 가정 2~5개
  }
}

규칙:
- items는 "가장 가능성이 높은 순서"로 정렬하고, 최대 8개만.
- 음식명이 애매하면 더 일반적인 이름을 우선(예: '곱창전골' 우선, '소곱창전골'은 후보).
- visible에는 사진에서 실제로 확인 가능한 재료/구성만 넣어. 보이지 않으면 넣지 마.
- assumed에는 보이지 않더라도 관행상 들어갈 가능성이 높은 재료/양념만 넣어(예: 다진마늘/간장/고추장 등). 너무 많이 넣지 마.
- estimated_nutrients는 "사진 속 1인분(1인 기준)"을 가정해서 추정해.
- range는 [최소, 최대]로, 최소<최대가 되게. value는 그 사이의 대표값.
- 확신이 낮으면 range를 넓혀.
- assumptions 예: "국물 포함", "1인분 기준", "기름 사용량 보통", "밥 제외" 등.
""".strip()


def try_parse_json(s: str) -> Optional[dict]:
    # 1) raw json
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) remove ```json ... ```
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 3) extract first {...}
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return None


def _clean_str_list(xs: Any, limit: int) -> List[str]:
    if not isinstance(xs, list):
        return []
    out: List[str] = []
    seen = set()
    for x in xs:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if not s:
            continue
        key = re.sub(r"\s+", "", s)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= limit:
            break
    return out


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    image: UploadFile = File(...),
    prompt: str = Form(JSON_PROMPT),
):
    content = await image.read()

    # Validate image
    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 1) Gemini call (vision)
    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, pil_img],
        )
        text = getattr(resp, "text", None) or str(resp)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")

    data = try_parse_json(text)

    # 2) Repair if non-JSON
    if data is None:
        repair_prompt = f"""
너는 JSON 변환기야. 아래 출력물을 "반드시" 지정 스키마의 JSON으로만 변환해.
다른 문장/설명 절대 금지.

스키마:
{{
  "items":[string],
  "vision": {{
    "ingredients": {{
      "visible":[string],
      "assumed":[string]
    }},
    "estimated_nutrients": {{
      "kcal":      {{"value": number, "range":[number, number], "unit":"kcal"}},
      "carb_g":    {{"value": number, "range":[number, number], "unit":"g"}},
      "protein_g": {{"value": number, "range":[number, number], "unit":"g"}},
      "fat_g":     {{"value": number, "range":[number, number], "unit":"g"}},
      "sugar_g":   {{"value": number, "range":[number, number], "unit":"g"}},
      "sodium_mg": {{"value": number, "range":[number, number], "unit":"mg"}}
    }},
    "assumptions":[string]
  }}
}}

원문:
{text}
""".strip()

        try:
            resp2 = client.models.generate_content(model=MODEL_NAME, contents=[repair_prompt])
            text2 = getattr(resp2, "text", None) or str(resp2)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Gemini repair call failed: {e}")

        data = try_parse_json(text2)

    if data is None:
        raise HTTPException(status_code=502, detail=f"Gemini returned non-JSON: {text[:400]}")

    # 3) Validate schema (strict)
    try:
        # Start with minimum skeleton then overwrite with validated vision
        validated = AnalyzeResponse.model_validate(
            {
                "items": data.get("items", []),
                "vision": data.get("vision"),
                "final": {"source": "vision_estimate", "nutrients": {}, "notes": []},
            }
        )
        validated.items = data.get("items", []) or []
        validated.vision = VisionBlock.model_validate(data.get("vision"))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Invalid JSON schema from Gemini: {e}")

    # 4) Server-side cleanup / caps (MVP 안정화)
    validated.items = _clean_str_list(validated.items, limit=8)

    # ingredients caps
    ing = validated.vision.ingredients
    ing.visible = _clean_str_list(ing.visible, limit=10)
    ing.assumed = _clean_str_list(ing.assumed, limit=8)

    # assumptions cap
    validated.vision.assumptions = _clean_str_list(validated.vision.assumptions, limit=5)

    # 5) Final: always use vision estimate in this version
    validated.final = FinalBlock(
        source="vision_estimate",
        nutrients=validated.vision.estimated_nutrients.model_dump(),
        notes=[
            "공공 영양DB를 사용하지 않고 사진 기반 추정치를 제공합니다.",
            "estimated_nutrients.range 및 assumptions는 추정 근거/불확실성을 나타냅니다.",
        ],
    )

    return validated
