import os
import re
import io
import cv2
import json
import time
import base64
import torch
import numpy as np
import requests
import supervision as sv
import traceback

from io import BytesIO
from typing import Any, Optional, List, Tuple, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse

from PIL import Image as PILImage, UnidentifiedImageError

# ===== Vertex AI (Gemini) =====
from google.auth.transport.requests import Request as GARequest
from google.oauth2 import service_account
from google.auth import default as google_auth_default

# ===== Grounded-SAM2 / Grounding-DINO =====
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ===================== Config =====================
PROJECT_ID = os.getenv("PROJECT_ID", "alpine-fin-473918-n3")
LOCATION = os.getenv("LOCATION", "us-central1")
ENDPOINT_ID = os.getenv("ENDPOINT_ID", "8659391841038237696")
ENDPOINT_URL = (
    f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
    f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}:generateContent"
)
SA_KEY_PATH = os.getenv("SA_KEY_PATH", "")

GROUNDING_MODEL = os.getenv("GROUNDING_MODEL", "IDEA-Research/grounding-dino-base")
SAM2_CHECKPOINT = os.getenv("SAM2_CHECKPOINT", "./checkpoints/sam2.1_hiera_large.pt")
SAM2_MODEL_CONFIG = os.getenv("SAM2_MODEL_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FIXED_PROMPT = (
    "Identify all visible food items in the image. "
    "Return a JSON array of objects: [{'name': <string>, 'edible': <0-10>}]. "
    "'edible' score: 0-4 = spoiled/inedible, 5-7 = okay, 8-10 = fresh. "
    "Use lowercase singular names (snake_case). "
    "Output JSON only, no text."
)

app = FastAPI(title="Food Detector: Gemini + Grounded-SAM2 (batch3 + gemini-labels + refill)")

# Optional custom color palette (if your supervision_utils defines one)
CUSTOM_COLOR_MAP = None
try:
    from supervision_utils import CUSTOM_COLOR_MAP as _CUSTOM_COLOR_MAP
    CUSTOM_COLOR_MAP = _CUSTOM_COLOR_MAP
except Exception:
    pass


# ===================== Vertex token =====================
def get_access_token() -> str:
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    if SA_KEY_PATH and os.path.exists(SA_KEY_PATH):
        creds = service_account.Credentials.from_service_account_file(SA_KEY_PATH, scopes=scopes)
    else:
        creds, _ = google_auth_default(scopes=scopes)
    if not creds.valid:
        creds.refresh(GARequest())
    return creds.token


# ===================== Utilities =====================
def extract_json_array(text: str) -> Optional[List[Any]]:
    s = (text or "").replace("\ufeff", "").replace("\u200b", "")
    s = s.replace("```json", "```").replace("```JSON", "```").replace("`", "")
    lines = []
    for ln in s.splitlines():
        t = ln.strip()
        if t.startswith("payload=") and t.endswith("[]"):
            continue
        lines.append(ln)
    s = "\n".join(lines).strip()
    m = re.search(r"```(.*?)```", s, flags=re.S)
    if m:
        s = m.group(1).strip()
    t = s.strip()
    if t.startswith("[") and t.endswith("]"):
        try:
            return json.loads(t)
        except Exception:
            pass
    start = t.find("[")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        else:
            if c == '"':
                in_str = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    cand = t[start:i + 1]
                    try:
                        return json.loads(cand)
                    except Exception:
                        return None
    return None

def image_to_b64(img_bgr: np.ndarray, quality: int = 92) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode image")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def split_batch3(names: List[str]) -> List[List[str]]:
    names = [n.strip() for n in names if n and n.strip()]
    if len(names) <= 3:
        return [names]
    return [names[i:i + 3] for i in range(0, len(names), 3)]

# Lightweight canonicalization just for matching DINO tokens back to Gemini names
_CANON_PAT = re.compile(r"[^a-z0-9]+")
def canonical_soft(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("##", "")
    s = _CANON_PAT.sub(" ", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def map_labels_to_gemini(batch_labels: List[str], batch_names: List[str]) -> List[str]:
    """Map DINO token labels back to the original Gemini names within the same batch."""
    canon_names = [canonical_soft(x) for x in batch_names]
    out = []
    for lab in batch_labels:
        c = canonical_soft(lab)
        hit = None
        # exact
        for i, cn in enumerate(canon_names):
            if c == cn:
                hit = batch_names[i]; break
        # contains
        if hit is None:
            for i, cn in enumerate(canon_names):
                if (c in cn) or (cn in c):
                    hit = batch_names[i]; break
        out.append(hit if hit is not None else lab)
    return out


# ===================== IoU / IoA / NMS =====================
def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    return inter / (area_a + area_b - inter + 1e-6)

def ioa_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    return inter / (area_a + 1e-6)

def overlap_strong(a: np.ndarray, b: np.ndarray, thr_iou: float, thr_ioa: float) -> bool:
    if iou_xyxy(a, b) >= thr_iou:
        return True
    return max(ioa_xyxy(a, b), ioa_xyxy(b, a)) >= thr_ioa

def nms_global(xyxy: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    if xyxy.size == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=int)


# ===================== Grounding-DINO inference =====================
def run_grounding_dino_simple(pil_img: PILImage.Image, names: List[str], box_th: float, text_th: float):
    names = [n.strip() for n in names if n and n.strip()]
    if not names:
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=float), []
    text = ". ".join(names) + "."
    inputs = _processor(images=pil_img, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = _grounding_model(**inputs)
    W, H = pil_img.size
    det = _processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=float(box_th), text_threshold=float(text_th),
        target_sizes=[(H, W)]
    )[0]
    boxes = det.get("boxes")
    scores = det.get("scores")
    labels = det.get("labels", [])
    if boxes is None or scores is None:
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=float), []
    return boxes.cpu().numpy().astype(float), scores.cpu().numpy().astype(float), labels


# ===================== Gemini call =====================
def post_gemini(image_b64_jpeg: str) -> List[dict]:
    token = get_access_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"}
    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"inlineData": {"mimeType": "image/jpeg", "data": image_b64_jpeg}},
                {"text": FIXED_PROMPT},
            ],
        }],
        "generationConfig": {
            "temperature": 0.0, "maxOutputTokens": 1024, "responseMimeType": "application/json",
        },
    }
    last_text = None
    for attempt in range(3):
        resp = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(payload), timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return extract_json_array(text) or []
            except Exception:
                raise HTTPException(status_code=500, detail="Model response structure error")
        else:
            last_text = resp.text
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (2 ** attempt))
                continue
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    raise HTTPException(status_code=500, detail=last_text or "Vertex internal error")


# ===================== Model init =====================
print(f"[Init] device = {DEVICE}")
torch.set_grad_enabled(False)
amp_ctx = None
if DEVICE == "cuda":
    try:
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        amp_ctx.__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        amp_ctx = None

_sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
_sam2_predictor = SAM2ImagePredictor(_sam2_model)
_processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
_grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(DEVICE).eval()

_palette = sv.ColorPalette.from_hex(CUSTOM_COLOR_MAP) if CUSTOM_COLOR_MAP else sv.ColorPalette.default()
_box_annotator = sv.BoxAnnotator(color=_palette)
_label_annotator = sv.LabelAnnotator(color=_palette)
_mask_annotator = sv.MaskAnnotator(color=_palette)


# ===================== Overlap rule (uniqueness by Gemini name) =====================
def apply_overlap_rule(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: List[str],
    overlap_iou: float = 0.45,
    overlap_ioa: float = 0.75,
    placeholder_mask: Optional[List[bool]] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], List[int]]:
    """
    Build overlap clusters using IoU/IoA, then:
      - If a cluster contains any globally-unique (non-placeholder) label ? keep those, drop others.
      - Else keep the highest score (prefer non-placeholder), drop the rest.
    Uniqueness is judged by exact string equality of Gemini names.
    """
    N = boxes.shape[0]
    if N <= 1:
        return boxes, scores, labels, list(range(N)), []

    if placeholder_mask is None or len(placeholder_mask) != N:
        placeholder_mask = [False] * N

    label_counts: Dict[str, int] = {}
    for i, L in enumerate(labels):
        if not placeholder_mask[i]:
            label_counts[L] = label_counts.get(L, 0) + 1

    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if overlap_strong(boxes[i], boxes[j], thr_iou=overlap_iou, thr_ioa=overlap_ioa):
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * N
    kept, removed = set(), set()

    for s in range(N):
        if visited[s]:
            continue
        q = [s]; visited[s] = True
        comp = []
        while q:
            u = q.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)

        if len(comp) == 1:
            kept.add(comp[0]); continue

        unique_idxs = [idx for idx in comp
                       if (not placeholder_mask[idx]) and label_counts.get(labels[idx], 0) == 1]
        if len(unique_idxs) > 0:
            for idx in unique_idxs: kept.add(idx)
            for idx in comp:
                if idx not in unique_idxs: removed.add(idx)
            continue

        non_placeholder = [idx for idx in comp if not placeholder_mask[idx]]
        if len(non_placeholder) > 0:
            best = max(non_placeholder, key=lambda k: float(scores[k]))
        else:
            best = max(comp, key=lambda k: float(scores[k]))
        kept.add(best)
        for idx in comp:
            if idx != best: removed.add(idx)

    keep_list = sorted(list(kept))
    rem_list = sorted(list(removed))
    return boxes[keep_list], scores[keep_list], [labels[i] for i in keep_list], keep_list, rem_list


# ===================== Mask-based grayscale =====================
def make_masked_pil_by_sam_masks(pil_img: PILImage.Image, masks_to_mask: np.ndarray, alpha: float = 1.0) -> PILImage.Image:
    arr = np.array(pil_img).astype(np.uint8)
    h, w = arr.shape[:2]
    grey = np.full_like(arr, 128, dtype=np.uint8)
    m = np.zeros((h, w), dtype=bool)
    for mm in masks_to_mask:
        m |= mm.astype(bool)
    arr[m] = (alpha * grey[m] + (1 - alpha) * arr[m]).astype(np.uint8)
    return PILImage.fromarray(arr, mode="RGB")


# ===================== Post refill (uniqueness-first) =====================
def refill_after_masks(
    pil_img: PILImage.Image,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: List[str],
    masks: np.ndarray,
    *,
    cluster_iou: float = 0.45,
    cluster_ioa: float = 0.75,
    far_ioa: float = 0.6,
    single_box_th: float = 0.28,
    single_text_th: float = 0.22,
    max_iters: int = 1
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    After masks are predicted, re-check high-overlap clusters:
      A) If all members are globally unique: mask the winner ? try single-class search for losers,
         then mask all losers ? try single-class search for the winner.
      B) If there exists unique members but not all: keep unique members; among non-unique,
         if winner is unique, drop non-unique losers; else if any loser is unique, drop the winner.
      C) If none are unique: mask the winner and try to re-find each loser elsewhere; if found with low IoA
         against the cluster, replace the loser box with the new one (when score improves).
    Finally drop all marked (score==0.0) and apply overlap rule again.
    """
    N = boxes.shape[0]
    if N <= 1:
        return boxes, scores, labels

    placeholder_mask_global = [float(sc) <= 0.02 for sc in scores]
    name_counts: Dict[str, int] = {}
    for i, L in enumerate(labels):
        if not placeholder_mask_global[i]:
            name_counts[L] = name_counts.get(L, 0) + 1

    def is_unique(idx: int) -> bool:
        if placeholder_mask_global[idx]:
            return False
        return name_counts.get(labels[idx], 0) == 1

    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if overlap_strong(boxes[i], boxes[j], thr_iou=cluster_iou, thr_ioa=cluster_ioa):
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * N
    comps = []
    for s in range(N):
        if visited[s]:
            continue
        q = [s]; visited[s] = True
        comp = []
        while q:
            u = q.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)
        comps.append(comp)

    cur_boxes = boxes.copy()
    cur_scores = scores.copy()
    cur_labels = labels[:]
    cur_masks = masks.copy()

    comps.sort(key=len, reverse=True)

    for _ in range(max_iters):
        changed = False

        for comp in comps:
            if len(comp) <= 1:
                continue

            uniq_flags = [is_unique(i) for i in comp]
            all_unique = all(uniq_flags)
            any_unique = any(uniq_flags)

            best_idx = max(comp, key=lambda k: float(cur_scores[k]))
            losers = [k for k in comp if k != best_idx]

            if all_unique:
                # Winner masks ? try losers
                winner_mask = cur_masks[best_idx][None, ...]
                masked_pil = make_masked_pil_by_sam_masks(pil_img, winner_mask, alpha=1.0)
                for loser in losers:
                    target_label = cur_labels[loser]
                    sb, ss, _ = run_grounding_dino_simple(masked_pil, [target_label],
                                                          box_th=single_box_th, text_th=single_text_th)
                    if sb is None or len(sb) == 0:
                        continue
                    ok_ids = []
                    for t in range(len(sb)):
                        bb = sb[t]
                        max_ioa = 0.0
                        for k in comp:
                            max_ioa = max(max_ioa, ioa_xyxy(bb, cur_boxes[k]), ioa_xyxy(cur_boxes[k], bb))
                        if max_ioa < far_ioa:
                            ok_ids.append(t)
                    if not ok_ids:
                        continue
                    best_loc = max(ok_ids, key=lambda t: float(ss[t]))
                    if float(ss[best_loc]) > float(cur_scores[loser]):
                        cur_boxes[loser] = sb[best_loc]
                        cur_scores[loser] = float(ss[best_loc])
                        changed = True

                # Losers mask ? try winner
                if len(losers) > 0:
                    losers_masks = cur_masks[losers]
                    masked_pil2 = make_masked_pil_by_sam_masks(pil_img, losers_masks, alpha=1.0)
                    target_label = cur_labels[best_idx]
                    sb, ss, _ = run_grounding_dino_simple(masked_pil2, [target_label],
                                                          box_th=single_box_th, text_th=single_text_th)
                    if sb is not None and len(sb) > 0:
                        ok_ids = []
                        for t in range(len(sb)):
                            bb = sb[t]
                            max_ioa = 0.0
                            for k in comp:
                                if k == best_idx:
                                    continue
                                max_ioa = max(max_ioa, ioa_xyxy(bb, cur_boxes[k]), ioa_xyxy(cur_boxes[k], bb))
                            if max_ioa < far_ioa:
                                ok_ids.append(t)
                        if ok_ids:
                            best_loc = max(ok_ids, key=lambda t: float(ss[t]))
                            if float(ss[best_loc]) > float(cur_scores[best_idx]):
                                cur_boxes[best_idx] = sb[best_loc]
                                cur_scores[best_idx] = float(ss[best_loc])
                                changed = True
                continue

            if any_unique:
                if is_unique(best_idx):
                    # Winner unique ? drop all non-unique losers
                    for loser in losers:
                        if not is_unique(loser):
                            cur_scores[loser] = 0.0
                else:
                    # Some losers are unique while winner is not ? drop winner
                    unique_losers = [l for l in losers if is_unique(l)]
                    if len(unique_losers) > 0:
                        cur_scores[best_idx] = 0.0
                continue

            # None unique: winner masks ? try losers elsewhere
            winner_mask = cur_masks[best_idx][None, ...]
            masked_pil = make_masked_pil_by_sam_masks(pil_img, winner_mask, alpha=1.0)
            for loser in losers:
                target_label = cur_labels[loser]
                sb, ss, _ = run_grounding_dino_simple(masked_pil, [target_label],
                                                      box_th=single_box_th, text_th=single_text_th)
                if sb is None or len(sb) == 0:
                    continue
                ok_ids = []
                for t in range(len(sb)):
                    bb = sb[t]
                    max_ioa = 0.0
                    for k in comp:
                        max_ioa = max(max_ioa, ioa_xyxy(bb, cur_boxes[k]), ioa_xyxy(cur_boxes[k], bb))
                    if max_ioa < far_ioa:
                        ok_ids.append(t)
                if not ok_ids:
                    continue
                best_loc = max(ok_ids, key=lambda t: float(ss[t]))
                if float(ss[best_loc]) > float(cur_scores[loser]):
                    cur_boxes[loser] = sb[best_loc]
                    cur_scores[loser] = float(ss[best_loc])
                    changed = True

        if not changed:
            break

    # Remove all boxes marked for deletion (score==0.0)
    keep_mask = np.array([sc > 0.0 for sc in cur_scores], dtype=bool)
    cur_boxes = cur_boxes[keep_mask]
    cur_scores = cur_scores[keep_mask]
    cur_labels = [lbl for k, lbl in enumerate(cur_labels) if keep_mask[k]]

    # Re-apply overlap rule after refill/delete
    placeholder_mask_after = [float(sc) <= 0.02 for sc in cur_scores]
    new_boxes, new_scores, new_labels, _, _ = apply_overlap_rule(
        cur_boxes, cur_scores, cur_labels,
        overlap_iou=cluster_iou, overlap_ioa=cluster_ioa,
        placeholder_mask=placeholder_mask_after
    )
    return new_boxes, new_scores, new_labels


# ===================== Routes =====================
@app.get("/")
def root() -> Any:
    return {
        "service": "food-detector-batch3-nms-overlaprule",
        "routes": ["/healthz", "/infer", "/docs"],
        "models": {
            "vertex_endpoint": ENDPOINT_ID,
            "grounding_model": GROUNDING_MODEL,
            "sam2": SAM2_MODEL_CONFIG
        }
    }

@app.get("/healthz")
def healthz() -> Any:
    return {"ok": True, "device": DEVICE}

@app.post("/infer")
async def infer(
    image: UploadFile = File(..., description="Image file (jpg/png/heic/etc.)"),
    draw_mask: bool = False,
    box_threshold: float = 0.4,
    text_threshold: float = 0.3,
    nms_iou: float = Query(0.5, ge=0.0, le=1.0),
    overlap_iou: float = Query(0.45, ge=0.0, le=1.0),
    overlap_ioa: float = Query(0.75, ge=0.0, le=1.0)
):
    """
    Pipeline:
      1) Gemini ? items
      2) DINO (batch size = 3, labels mapped back to Gemini names)
      3) Global NMS
      4) Per-item backfill (single-class scan; else placeholder center box with low score)
      5) First overlap prune (uniqueness-first, using Gemini names)
      6) SAM2 masks
      7) Post-refill with grayscale masking + second prune
    """
    try:
        raw = await image.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty upload")

        try:
            pil = PILImage.open(BytesIO(raw)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image file (cannot be opened)")

        # Convert to JPEG for Gemini
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=92)
        buf.seek(0)
        jpeg_bytes = buf.read()
        jpeg_b64_for_gemini = base64.b64encode(jpeg_bytes).decode("utf-8")

        # 1) Gemini
        gemini_ok = True
        try:
            items = post_gemini(jpeg_b64_for_gemini)
        except HTTPException as ge:
            print("[Gemini ERROR]", ge.status_code, str(ge.detail)[:500])
            gemini_ok = False
            items = [
                {"name": "food", "edible": 7},
                {"name": "fruit", "edible": 7},
                {"name": "vegetable", "edible": 7},
                {"name": "drink", "edible": 7}
            ]

        # 2) DINO batches of 3 and map DINO labels to Gemini names
        names = [str(x.get("name") or "").strip() for x in items if (x.get("name") or "").strip()]
        batches = split_batch3(names)
        all_batches = [b[:] for b in batches]
        used_prompts: List[str] = []

        all_boxes_list: List[np.ndarray] = []
        all_scores_list: List[np.ndarray] = []
        all_labels_raw: List[str] = []

        for batch in batches:
            prompt = ". ".join(batch) + "."
            used_prompts.append(prompt)
            boxes, scores, labels = run_grounding_dino_simple(pil, batch, box_threshold, text_threshold)
            if len(boxes) == 0:
                continue
            all_boxes_list.append(boxes)
            all_scores_list.append(scores)
            if labels and len(labels) == boxes.shape[0]:
                mapped = map_labels_to_gemini([str(lbl) for lbl in labels], batch)
                all_labels_raw.extend(mapped)
            else:
                all_labels_raw.extend([batch[min(i, len(batch)-1)] for i in range(len(scores))])

        # 3) Global NMS
        if len(all_boxes_list) > 0:
            all_boxes = np.concatenate(all_boxes_list, axis=0).astype(float)
            all_scores = np.concatenate(all_scores_list, axis=0).astype(float)
            keep_idx = nms_global(all_boxes, all_scores, iou_thr=float(nms_iou))
            all_boxes = all_boxes[keep_idx]
            all_scores = all_scores[keep_idx]
            kept_labels = [all_labels_raw[i] for i in keep_idx]
        else:
            all_boxes = np.zeros((0, 4), dtype=float)
            all_scores = np.zeros((0,), dtype=float)
            kept_labels = []

        # 4) Per-item backfill (ensure each Gemini item has a box)
        H, W = pil.size[1], pil.size[0]
        per_item_status: List[Dict[str, Any]] = []
        name_has_box: Dict[str, bool] = {n: False for n in names}
        for lbl in kept_labels:
            if lbl in name_has_box:
                name_has_box[lbl] = True

        add_boxes, add_scores, add_labels = [], [], []
        for n in names:
            if name_has_box[n]:
                per_item_status.append({"name": n, "found_by": "global"})
                continue
            sb, ss, _ = run_grounding_dino_simple(pil, [n], box_th=0.25, text_th=0.20)
            if len(sb) > 0:
                best = int(np.argmax(ss))
                add_boxes.append(sb[best:best + 1])
                add_scores.append(ss[best:best + 1])
                add_labels.append([n])
                per_item_status.append({"name": n, "found_by": "single_pass"})
            else:
                # Placeholder center box (very low score)
                cx0 = int(W * 0.2); cy0 = int(H * 0.2)
                cx1 = int(W * 0.8); cy1 = int(H * 0.8)
                add_boxes.append(np.array([[cx0, cy0, cx1, cy1]], dtype=float))
                add_scores.append(np.array([0.01], dtype=float))
                add_labels.append([n])
                per_item_status.append({"name": n, "found_by": "placeholder"})

        if len(add_boxes) > 0:
            add_boxes = np.concatenate(add_boxes, axis=0)
            add_scores = np.concatenate(add_scores, axis=0)
            add_labels = sum(add_labels, [])
            all_boxes = np.concatenate([all_boxes, add_boxes], axis=0) if all_boxes.size else add_boxes
            all_scores = np.concatenate([all_scores, add_scores], axis=0) if all_scores.size else add_scores
            kept_labels = kept_labels + add_labels

        # 5) First overlap prune (uniqueness-first)
        placeholder_mask = [float(sc) <= 0.02 for sc in all_scores]
        boxes_after_overlap, scores_after_overlap, labels_after_overlap, kept_idx2, removed_idx2 = \
            apply_overlap_rule(
                all_boxes, all_scores, kept_labels,
                overlap_iou=float(overlap_iou),
                overlap_ioa=float(overlap_ioa),
                placeholder_mask=placeholder_mask
            )

        # Early return if nothing to visualize
        if boxes_after_overlap.shape[0] == 0:
            img_bgr0 = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            return JSONResponse(content={
                "gemini_ok": gemini_ok,
                "items": items,
                "num_detections": 0,
                "boxes_xyxy": [],
                "labels": [],
                "text_prompt": " | ".join(used_prompts),
                "all_batches": all_batches,
                "per_item_status": per_item_status,
                "nms_iou": float(nms_iou),
                "overlap_iou": float(overlap_iou),
                "overlap_ioa": float(overlap_ioa),
                "kept_indices_after_overlap": kept_idx2,
                "removed_indices_after_overlap": removed_idx2,
                "bbox_image_b64": image_to_b64(img_bgr0)
            })

        # 6) SAM2 masks
        _sam2_predictor.set_image(np.array(pil))
        masks, sam_scores, logits = _sam2_predictor.predict(
            point_coords=None, point_labels=None, box=boxes_after_overlap, multimask_output=False
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # 7) Post-refill and second prune
        boxes_refilled, scores_refilled, labels_refilled = refill_after_masks(
            pil_img=pil,
            boxes=boxes_after_overlap,
            scores=scores_after_overlap,
            labels=labels_after_overlap,
            masks=masks,
            cluster_iou=float(overlap_iou),
            cluster_ioa=float(overlap_ioa),
            far_ioa=0.6,
            single_box_th=0.28, single_text_th=0.22,
            max_iters=1
        )

        # If updated, recompute masks for consistent visualization
        if (boxes_refilled.shape[0] != boxes_after_overlap.shape[0]) or \
           (not np.allclose(boxes_refilled, boxes_after_overlap)):
            _sam2_predictor.set_image(np.array(pil))
            masks, sam_scores, logits = _sam2_predictor.predict(
                point_coords=None, point_labels=None, box=boxes_refilled, multimask_output=False
            )
            if masks.ndim == 4:
                masks = masks.squeeze(1)

        boxes_final = boxes_refilled
        scores_final = scores_refilled
        labels_final = labels_refilled  # strictly Gemini names

        # Display text without spaces (for compact on-image labels); JSON keeps raw names
        labels_display = [lab.replace(" ", "") for lab in labels_final]
        vis_labels = [f"{lbl} {float(sc):.2f}" for lbl, sc in zip(labels_display, scores_final)]

        # Visualization
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        detections = sv.Detections(
            xyxy=boxes_final,
            mask=masks.astype(bool),
            class_id=np.arange(len(boxes_final))
        )
        boxed = _box_annotator.annotate(scene=img_bgr.copy(), detections=detections)
        boxed = _label_annotator.annotate(scene=boxed, detections=detections, labels=vis_labels)
        if draw_mask:
            boxed = _mask_annotator.annotate(scene=boxed, detections=detections)
        bbox_image_b64 = image_to_b64(boxed)

        return JSONResponse(content={
            "gemini_ok": gemini_ok,
            "items": items,
            "num_detections": int(boxes_final.shape[0]),
            "boxes_xyxy": boxes_final.tolist(),
            "labels": vis_labels,            # compact display (no spaces)
            "labels_raw": labels_final,      # exact Gemini names
            "text_prompt": " | ".join(used_prompts),
            "all_batches": all_batches,
            "per_item_status": per_item_status,
            "nms_iou": float(nms_iou),
            "overlap_iou": float(overlap_iou),
            "overlap_ioa": float(overlap_ioa),
            "kept_indices_after_overlap": kept_idx2,
            "removed_indices_after_overlap": removed_idx2,
            "bbox_image_b64": bbox_image_b64
        })

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print("[SERVER ERROR]", tb)
        raise HTTPException(status_code=500, detail=str(e) or (tb[-500:] if tb else "internal error"))
