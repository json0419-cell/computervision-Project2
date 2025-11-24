import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

"""
Default parameters
"""
GROUNDING_MODEL = "IDEA-Research/grounding-dino-base"
TEXT_PROMPT = "sour_cucumber. tomatoes. cucumbers."
print(f"Text Prompt = {TEXT_PROMPT}")
IMG_PATH = "test1.png"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
OUTPUT_DIR = Path("outputs/grounded_sam2_hf_demo")
DUMP_JSON_RESULTS = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {DEVICE}")

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ??????? autocast????? __enter__??????????
use_bf16 = (DEVICE == "cuda")  # CPU ??????? bf16 autocast
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# build SAM2 image predictor
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino from huggingface
processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(DEVICE)

# ---------- ???????????? RGB ???? PIL ----------
def to_rgb_pil(img_path_or_obj):
    if isinstance(img_path_or_obj, Image.Image):
        return img_path_or_obj.convert("RGB")
    if isinstance(img_path_or_obj, str):
        return Image.open(img_path_or_obj).convert("RGB")
    if isinstance(img_path_or_obj, np.ndarray):
        arr = img_path_or_obj
        if arr.ndim == 2:  # HxW ??
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        # ???? cv2.imread(BGR)????? RGB
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0, 1) * 255
            arr = arr.astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    raise TypeError("Unsupported image type for processor(images=...).")

# setup the input image and text prompt
text = TEXT_PROMPT
image_pil = to_rgb_pil(IMG_PATH)   # ? ??? RGB ??? PIL
# SAM2 ?? numpy RGB?HWC?uint8?
sam2_predictor.set_image(np.array(image_pil))  # PIL -> RGB ndarray(H,W,3)

# GroundingDINO ??
with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=use_bf16):
    inputs = processor(images=image_pil, text=text, return_tensors="pt").to(DEVICE)
    outputs = grounding_model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image_pil.size[::-1]]  # (H, W)
)

# ?????????????????
if len(results) == 0 or results[0]["boxes"].numel() == 0:
    print("No detections found.")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"),
                cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR))
    if DUMP_JSON_RESULTS:
        with open(os.path.join(OUTPUT_DIR, "grounded_sam2_hf_model_demo_results.json"), "w") as f:
            json.dump({
                "image_path": IMG_PATH,
                "annotations": [],
                "box_format": "xyxy",
                "img_width": image_pil.width,
                "img_height": image_pil.height,
            }, f, indent=4)
    raise SystemExit(0)

# get the box prompt for SAM 2
input_boxes = results[0]["boxes"].detach().cpu().numpy()   # (N, 4), xyxy
confidences = results[0]["scores"].detach().cpu().numpy().tolist()
class_names = results[0]["labels"]
class_ids = np.arange(len(class_names), dtype=int)

# SAM2 ????
masks, scores, logits = sam2_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

# convert the shape to (n, H, W)
if masks.ndim == 4:
    masks = masks.squeeze(1)
# ????????
masks = masks.astype(bool)

labels = [f"{c} {s:.2f}" for c, s in zip(class_names, confidences)]

# visualize?OpenCV ??? BGR???????????
img_bgr = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
if img_bgr is None:
    # ??????? PIL ? BGR
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

detections = sv.Detections(
    xyxy=input_boxes,
    mask=masks,
    class_id=class_ids
)

box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
annotated_frame = box_annotator.annotate(scene=img_bgr.copy(), detections=detections)

label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
annotated_frame_with_mask = mask_annotator.annotate(scene=annotated_frame, detections=detections)
cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame_with_mask)

# dump JSON results
def single_mask_to_rle(mask_bool):
    # ??? (H, W) ? bool/uint8
    rle = mask_util.encode(np.array(mask_bool[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

if DUMP_JSON_RESULTS:
    mask_rles = [single_mask_to_rle(m) for m in masks]
    input_boxes_list = input_boxes.tolist()
    scores_list = np.asarray(scores).tolist() if isinstance(scores, (np.ndarray, list)) else list(scores)

    results_json = {
        "image_path": IMG_PATH,
        "annotations": [
            {
                "class_name": c,
                "bbox": b,
                "segmentation": r,
                "score": s,
            }
            for c, b, r, s in zip(class_names, input_boxes_list, mask_rles, scores_list)
        ],
        "box_format": "xyxy",
        "img_width": image_pil.width,
        "img_height": image_pil.height,
    }

    with open(os.path.join(OUTPUT_DIR, "grounded_sam2_hf_model_demo_results.json"), "w") as f:
        json.dump(results_json, f, indent=4)
