import os
import io
import json
import re
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
import httpx

from google import genai

load_dotenv()

# ======================
# ENV
# ======================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Put it in .env")

# 식품안전나라 OpenAPI 키 (없어도 서버는 돌아가되 db 섹션이 비어있음)
FOODSAFETY_API_KEY = os.getenv("FOODSAFETY_API_KEY", "").strip()

# 너가 성공했던 모델명으로 유지/수정
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip()

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="tandangi-ai", version="0.2.0")

# ======================
# Models (Response)
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


class VisionBlock(BaseModel):
    ingredients: List[str] = Field(default_factory=list)
    estimated_nutrients: VisionNutrients
    assumptions: List[str] = Field(default_factory=list)


class DbNutrients(BaseModel):
    serving_g: Optional[float] = None
    kcal: Optional[float] = None
    carb_g: Optional[float] = None
    protein_g: Optional[float] = None
    fat_g: Optional[float] = None
    sugar_g: Optional[float] = None
    sodium_mg: Optional[float] = None
    cholesterol_mg: Optional[float] = None
    sat_fat_g: Optional[float] = None
    trans_fat_g: Optional[float] = None


class DbBlock(BaseModel):
    matched: bool = False
    query: Optional[str] = None
    matched_name: Optional[str] = None
    nutrients: Optional[DbNutrients] = None
    raw: Optional[Dict[str, Any]] = None


class FinalBlock(BaseModel):
    source: str  # "db_primary" | "vision_estimate"
    nutrients: Dict[str, Any]
    notes: List[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    items: List[str] = Field(default_factory=list)
    vision: VisionBlock
    db: DbBlock
    final: FinalBlock


# ======================
# Prompts
# ======================

JSON_PROMPT = """
너는 음식/영양 분석 도우미야.
사용자가 보낸 "음식 사진"을 보고, 아래 JSON 스키마를 정확히 따라 결과를 만들어.

반드시 JSON만 출력해. 다른 문장/설명/코드블록 절대 금지.

스키마:
{
  "items": [string],  // 음식 후보명(가장 가능성 높은 순서)
  "vision": {
    "ingredients": [string],  // 사진 기반으로 추정한 주요 재료(최대 12개)
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
- items는 "가장 가능성 높은 순서"로 정렬하고, 최대 8개만.
- 음식명이 애매하면 더 일반적인 이름을 우선(예: '곱창전골' 우선, '소곱창전골'은 후보).
- estimated_nutrients는 "1인분(사진 속 1인 기준)"을 가정해서 추정해.
- range는 [최소, 최대]로, 최소<최대가 되게. value는 그 사이의 대표값.
- 너무 확신이 없으면 range를 넓혀.
- ingredients는 양념 포함 가능(고추장/고춧가루/간장 등), 최대 12개.
- assumptions 예: "국물 포함", "1인분 기준", "기름 사용량 보통", "밥 제외/포함" 등.
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


# ======================
# FoodSafety API (I0750)
# ======================

def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    # sometimes "0", "0.0", etc.
    try:
        return float(s)
    except Exception:
        return None


def parse_foodsafety_row(row: Dict[str, Any]) -> DbNutrients:
    # I0750 fields (commonly used)
    # SERVING_WT: 1회 제공량(g)
    # NUTR_CONT1~9: kcal, carb, protein, fat, sugars, sodium, cholesterol, sat_fat, trans_fat
    return DbNutrients(
        serving_g=_to_float(row.get("SERVING_WT")),
        kcal=_to_float(row.get("NUTR_CONT1")),
        carb_g=_to_float(row.get("NUTR_CONT2")),
        protein_g=_to_float(row.get("NUTR_CONT3")),
        fat_g=_to_float(row.get("NUTR_CONT4")),
        sugar_g=_to_float(row.get("NUTR_CONT5")),
        sodium_mg=_to_float(row.get("NUTR_CONT6")),
        cholesterol_mg=_to_float(row.get("NUTR_CONT7")),
        sat_fat_g=_to_float(row.get("NUTR_CONT8")),
        trans_fat_g=_to_float(row.get("NUTR_CONT9")),
    )


async def foodsafety_search_first(query: str) -> Optional[Dict[str, Any]]:
    """
    Returns first row dict if exists, else None.
    """
    if not FOODSAFETY_API_KEY:
        return None

    # FoodSafety Korea OpenAPI: /api/{key}/I0750/json/1/5/DESC_KOR={query}
    # NOTE: some environments require http not https
    base_url = f"http://openapi.foodsafetykorea.go.kr/api/{FOODSAFETY_API_KEY}/I0750/json/1/5"
    params = {"DESC_KOR": query}

    timeout = httpx.Timeout(10.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as hc:
        r = await hc.get(base_url, params=params)
        r.raise_for_status()
        data = r.json()

    # structure varies but usually { "I0750": { "row": [ ... ] } }
    i0750 = data.get("I0750") if isinstance(data, dict) else None
    if not isinstance(i0750, dict):
        return None

    rows = i0750.get("row")
    if not isinstance(rows, list) or not rows:
        return None

    first = rows[0]
    if not isinstance(first, dict):
        return None

    return first


# ======================
# Endpoints
# ======================

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME, "foodsafety": bool(FOODSAFETY_API_KEY)}


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
    "ingredients":[string],
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

    # 3) Validate schema
    try:
        validated = AnalyzeResponse.model_validate(
            {
                # db/final will be filled below
                "items": data.get("items", []),
                "vision": data.get("vision"),
                "db": {"matched": False},
                "final": {"source": "vision_estimate", "nutrients": {}, "notes": []},
            }
        )
        # We overwrite validated.items and validated.vision with stricter values from original parsing:
        validated.items = data.get("items", []) or []
        validated.vision = VisionBlock.model_validate(data.get("vision"))
        validated.vision.ingredients = [x.strip() for x in validated.vision.ingredients if isinstance(x, str) and x.strip()]
validated.vision.ingredients = validated.vision.ingredients[:12]

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Invalid JSON schema from Gemini: {e}")

    # Normalize items (remove empties, trim, de-dup while keeping order)
    cleaned_items: List[str] = []
    seen = set()
    for x in validated.items:
        if not isinstance(x, str):
            continue
        name = x.strip()
        if not name:
            continue
        key = re.sub(r"\s+", "", name)
        if key in seen:
            continue
        seen.add(key)
        cleaned_items.append(name)
    validated.items = cleaned_items[:8]

    # 4) FoodSafety lookup (try each item in order)
    db_block = DbBlock(matched=False, query=None, matched_name=None, nutrients=None, raw=None)

    if FOODSAFETY_API_KEY and validated.items:
        for food_name in validated.items:
            try:
                row = await foodsafety_search_first(food_name)
            except Exception:
                row = None

            if row:
                matched_name = row.get("DESC_KOR") or food_name
                nutrients = parse_foodsafety_row(row)
                db_block = DbBlock(
                    matched=True,
                    query=food_name,
                    matched_name=str(matched_name),
                    nutrients=nutrients,
                    raw=row,
                )
                break

    validated.db = db_block

    # 5) Final (MVP rule)
    if validated.db.matched and validated.db.nutrients:
        # Use DB as final
        final = FinalBlock(
            source="db_primary",
            nutrients=validated.db.nutrients.model_dump(),
            notes=[
                "공공 영양DB(대표값) 기반",
                "사진 기반 재료/추정치는 참고용(vision 섹션)으로 제공",
            ],
        )
    else:
        # Use vision estimate as final
        final = FinalBlock(
            source="vision_estimate",
            nutrients=validated.vision.estimated_nutrients.model_dump(),
            notes=[
                "공공 영양DB 매칭 실패 또는 미설정으로 사진 기반 추정치 제공",
                "추정치에는 가정/범위가 포함됨(vision.assumptions, vision.estimated_nutrients.range)",
            ],
        )

    validated.final = final
    return validated
