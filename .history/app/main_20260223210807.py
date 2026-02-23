import os
import io
import json
import re
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
import base64
from fastapi import Request
from google import genai

from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests

load_dotenv()

# ======================
# ENV
# ======================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Put it in .env")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip()
client = genai.Client(api_key=GEMINI_API_KEY)

# ✅ 멀티 앱(멀티 Firebase 프로젝트) allowlist
# 예: ALLOWED_FIREBASE_PROJECT_IDS="app1-dev,app1-prod,app2-prod"
ALLOWED_FIREBASE_PROJECT_IDS = [
    x.strip()
    for x in (os.getenv("ALLOWED_FIREBASE_PROJECT_IDS", "")).split(",")
    if x.strip()
]

app = FastAPI(title="tandangi-ai", version="0.4.0")

# ======================
# AUTH (멀티 Firebase 프로젝트 토큰 검증)
# ======================

_google_request_adapter = google_requests.Request()

def _extract_bearer_token(req: Request) -> str:
    auth = req.headers.get("authorization") or req.headers.get("Authorization")
    if not auth:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    m = re.match(r"^\s*Bearer\s+(.+)\s*$", auth)
    if not m:
        raise HTTPException(status_code=401, detail="Invalid Authorization header (expected Bearer token)")
    return m.group(1)

def _b64url_decode(data: str) -> bytes:
    # JWT는 base64url padding 생략될 수 있음
    pad = '=' * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)

def _peek_project_id_from_jwt(idt: str) -> str:
    """
    ⚠️ 서명 검증 없이 payload만 읽어서 project_id(aud/iss) 추출.
    이 값은 '판별' 목적으로만 쓰고, 권한 부여는 반드시 verify 후에만.
    """
    parts = idt.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="Invalid JWT format")

    try:
        payload = json.loads(_b64url_decode(parts[1]).decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid JWT payload")

    aud = payload.get("aud")
    iss = payload.get("iss")

    # Firebase ID token의 일반적인 형태:
    # aud: <project_id>
    # iss: https://securetoken.google.com/<project_id>
    project_id = None
    if isinstance(aud, str) and aud.strip():
        project_id = aud.strip()

    if not project_id and isinstance(iss, str):
        m = re.match(r"^https://securetoken\.google\.com/(.+)$", iss.strip())
        if m:
            project_id = m.group(1)

    if not project_id:
        raise HTTPException(status_code=401, detail="Cannot determine Firebase project from token")

    return project_id

def verify_firebase_user(req: Request) -> Dict[str, Any]:
    """
    멀티 앱 공용 인증 Dependency.
    - Bearer token에서 project_id 추출
    - allowlist 검사
    - 해당 project_id를 audience로 Firebase 토큰 검증
    - 검증된 decoded claims 반환
    """
    idt = _extract_bearer_token(req)
    project_id = _peek_project_id_from_jwt(idt)

    if ALLOWED_FIREBASE_PROJECT_IDS:
        if project_id not in ALLOWED_FIREBASE_PROJECT_IDS:
            raise HTTPException(status_code=403, detail=f"Project not allowed: {project_id}")

    try:
        # ✅ 정식 검증 (서명/만료/issuer 등 포함)
        decoded = google_id_token.verify_firebase_token(
            idt,
            _google_request_adapter,
            audience=project_id,
        )
        if not decoded:
            raise HTTPException(status_code=401, detail="Token verification failed")

        # 참고: decoded에 uid는 'user_id'로 들어오는 경우가 많음
        uid = decoded.get("user_id") or decoded.get("sub")
        if not uid:
            raise HTTPException(status_code=401, detail="Token missing user id")

        # 추후 로깅/분기용으로 project_id도 포함해서 반환
        decoded["__project_id"] = project_id
        decoded["__uid"] = uid
        return decoded

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


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
    source: str  # always "vision_estimate"
    nutrients: Dict[str, Any]


class AnalyzeResponse(BaseModel):
    main: List[str] = Field(default_factory=list, description="메인 메뉴(최대 3)")
    sides: List[str] = Field(default_factory=list, description="반찬/사이드(최대 6)")
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
  "main": [string],   // 메인 메뉴(최대 3)
  "sides": [string],  // 반찬/사이드(최대 6)
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

분류 규칙(중요):
- main에는 한 끼의 '주요 요리'를 넣어: 전골/탕/찌개/국/볶음/구이/덮밥/면/밥 등.
- sides에는 '반찬/사이드'를 넣어: 무침/나물/장아찌/절임/김치/샐러드/소스/찍어먹는 반찬류.
- 사진에 반찬 그릇이 명확히 보이면 sides로 분류해.
- 애매하면 main 1개만 넣고, sides는 최소화해.

일반 규칙:
- visible에는 사진에서 실제로 확인 가능한 재료/구성만 넣어. 보이지 않으면 넣지 마.
- assumed에는 보이지 않더라도 관행상 들어갈 가능성이 높은 재료/양념만 넣어(너무 많이 넣지 마).
- estimated_nutrients는 "사진 속 1인분(1인 기준)"을 가정해서 추정해.
- range는 [최소, 최대]로 최소<최대. value는 대표값.
- 확신이 낮으면 range를 넓혀.
- assumptions 예: "국물 포함", "1인분 기준", "기름 사용량 보통", "밥 제외" 등.

visible/assumed 품질 규칙(매우 중요):
- visible에는 "사진에서 형태가 명확히 보이는 것"만 넣어. 애매하면 절대 visible에 넣지 말고 assumed로 보낼 것.
- 보이는 재료를 놓치지 말고, 반대로 보이지 않는 재료를 visible로 추측해서 넣지 말 것.
- 형태가 안 보이면(예: 양념/육수/가루) assumed로만 포함.

""".strip()


# ======================
# Helpers
# ======================
def try_parse_json(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        pass

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

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


def _is_side_like(name: str) -> bool:
    # 매우 단순한 MVP 휴리스틱(LLM 실수 안전망)
    side_keywords = ["무침", "나물", "장아찌", "절임", "김치", "샐러드", "피클", "젓갈", "볶음김치"]
    return any(k in name for k in side_keywords)


def _is_main_like(name: str) -> bool:
    main_keywords = ["전골", "탕", "찌개", "국", "볶음", "구이", "덮밥", "비빔밥", "면", "라면", "파스타", "스테이크"]
    return any(k in name for k in main_keywords)


def _post_classify(main: List[str], sides: List[str]) -> (List[str], List[str]):
    # 1) 기본 정리/중복 제거
    main = _clean_str_list(main, limit=3)
    sides = _clean_str_list(sides, limit=6)

    # 2) main에 반찬 키워드가 섞였으면 sides로 이동
    new_main: List[str] = []
    for m in main:
        if _is_side_like(m):
            if m not in sides:
                sides.append(m)
        else:
            new_main.append(m)
    main = new_main[:3]
    sides = _clean_str_list(sides, limit=6)

    # 3) main이 비었는데 sides만 있으면, 첫 항목을 main으로 승격
    if not main and sides:
        main = [sides[0]]
        sides = sides[1:]

    # 4) main이 여러 개인데 전부 side-like면 첫 번째만 main 유지
    if main and all(_is_side_like(x) for x in main):
        main = [main[0]]

    # 5) sides에 main-like가 들어갔으면(실수) main으로 올리되 main 3개 제한
    promoted: List[str] = []
    remaining_sides: List[str] = []
    for s in sides:
        if _is_main_like(s):
            promoted.append(s)
        else:
            remaining_sides.append(s)
    if promoted:
        merged = main + promoted
        main = _clean_str_list(merged, limit=3)
        sides = _clean_str_list(remaining_sides, limit=6)

    return main, sides


# ======================
# Endpoints
# ======================
@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    # ✅ 여기서 인증 강제 (공통 dependency)
    user_claims: Dict[str, Any] = Depends(verify_firebase_user),

    image: UploadFile = File(...),
    prompt: str = Form(JSON_PROMPT),
):
    # 필요하면 로깅/분기에도 사용 가능
    uid = user_claims["__uid"]
    project_id = user_claims["__project_id"]

    content = await image.read()

    # Validate image
    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 1) Gemini call (vision)
    try:
        resp = client.models.generate_content(model=MODEL_NAME, contents=[prompt, pil_img])
        text = getattr(resp, "text", None) or str(resp)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")

    data = try_parse_json(text)

    # 2) Repair if non-JSON
    if data is None:
        repair_prompt = f"""...""".strip()  # 네 repair_prompt 원본 그대로
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
                "main": data.get("main", []),
                "sides": data.get("sides", []),
                "vision": data.get("vision"),
                "final": {"source": "vision_estimate", "nutrients": {}},
            }
        )
        validated.main = data.get("main", []) or []
        validated.sides = data.get("sides", []) or []
        validated.vision = VisionBlock.model_validate(data.get("vision"))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Invalid JSON schema from Gemini: {e}")

    # 4) Server-side cleanup/caps
    validated.main, validated.sides = _post_classify(validated.main, validated.sides)

    ing = validated.vision.ingredients
    ing.visible = _clean_str_list(ing.visible, limit=10)
    ing.assumed = _clean_str_list(ing.assumed, limit=8)
    validated.vision.assumptions = _clean_str_list(validated.vision.assumptions, limit=5)

    # 5) Final (no notes)
    validated.final = FinalBlock(
        source="vision_estimate",
        nutrients=validated.vision.estimated_nutrients.model_dump(),
    )

    return validated