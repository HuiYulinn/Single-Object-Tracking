
"""
使用之前需要该下列的配置，每次输入一个文件，输出一个文件的相关结果，重复50次得到最终结果
"""

# -------------------------- CONSTANTS (EDIT IF NEEDED) --------------------------
MODEL_DIR   = "/home/ubuntu/pycharm/models/ckip"             # 调用权重的路径
FRAMES_DIR  = "/home/ubuntu/pycharm/videodata/test/001/color" # 路径
BBOX_FORMAT = "xywh"                                         # 使用'xywh'
BBOX        = [810.0, 372.0, 250.0, 121.0]                   # 第一帧位置
FPS         = 25                                             # 输出视频的帧率，便于查看
OUT_DIR     = "/home/ubuntu/pycharm/output"                  # 输出路径
FORCE_CPU   = False
SAVE_FRAMES = False
KEEP_TMP    = False
# --------------------------------------------------------------------------------

import os, sys, glob, shutil, json
from typing import List

# ---- HOTFIX: patch datasets.utils for modelscope import compatibility ----
try:
    import types
    import datasets.utils as _du
    if not hasattr(_du, "_datasets_server"):
        _du._datasets_server = types.SimpleNamespace()  # 占位对象，不会被实际用到
        print("[hotfix] patched datasets.utils._datasets_server")
except Exception as e:
    print("[hotfix] datasets patch failed:", e)
# ------------------------------------------------------------------------


def ensure_deps():
    try:
        import modelscope  # noqa: F401
    except Exception as e:
        print("ERROR: 'modelscope' is not installed. Try: pip install modelscope opencv-python")
        sys.exit(1)
    try:
        import torch  # noqa: F401
    except Exception as e:
        print("ERROR: 'torch' is not installed. Install PyTorch per your CUDA/OS from https://pytorch.org/")
        sys.exit(1)
    try:
        import cv2  # noqa: F401
    except Exception as e:
        print("ERROR: 'opencv-python' is not installed. Try: pip install opencv-python")
        sys.exit(1)

def maybe_fix_configuration_json(model_dir: str):
    """Ensure a configuration.json exists; if not, copy any configuration*.json to that name."""
    cfg = os.path.join(model_dir, "configuration.json")
    if os.path.isfile(cfg):
        return cfg
    cand = None
    for p in os.listdir(model_dir):
        if p.lower().startswith("configuration") and p.lower().endswith(".json"):
            cand = os.path.join(model_dir, p)
            break
    if cand:
        shutil.copy2(cand, cfg)
        print(f"[Info] configuration.json not found; copied '{cand}' -> '{cfg}'")
        return cfg
    raise FileNotFoundError(f"No configuration.json found in {model_dir} (and no alternative configuration*.json to copy).")

def list_frames(frames_dir: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    names = [os.path.join(frames_dir, n) for n in os.listdir(frames_dir)]
    paths = [p for p in names if os.path.splitext(p)[1].lower() in exts]
    if not paths:
        raise RuntimeError(f"No frames found in {frames_dir}")
    # numeric-aware sort (2 < 10)
    def num_key(p):
        b = os.path.basename(p)
        digits = "".join(ch for ch in b if ch.isdigit())
        try:
            return int(digits)
        except Exception:
            return b
    paths.sort(key=num_key)
    return paths

def frames_to_video(frame_paths: List[str], out_video_path: str, fps: int) -> None:
    import cv2
    first = cv2.imread(frame_paths[0])
    if first is None:
        raise RuntimeError(f"Cannot read first frame: {frame_paths[0]}")
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_video_path, fourcc, float(fps), (w, h))
    if not vw.isOpened():
        raise RuntimeError("Failed to open VideoWriter; try changing codec or extension to .avi")
    for p in frame_paths:
        img = cv2.imread(p)
        if img is None:
            raise RuntimeError(f"Cannot read frame: {p}")
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        vw.write(img)
    vw.release()

def annotate_video_with_boxes(video_path: str, boxes, out_path: str):
    # Prefer ModelScope helper if available
    try:
        from modelscope.utils.cv.image_utils import show_video_tracking_result
        show_video_tracking_result(video_path, boxes, out_path)
        return
    except Exception:
        # Fallback to OpenCV
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for i in range(len(boxes)):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            x1,y1,x2,y2 = boxes[i]
            x1 = max(0, min(w-1, int(round(x1))))
            y1 = max(0, min(h-1, int(round(y1))))
            x2 = max(0, min(w-1, int(round(x2))))
            y2 = max(0, min(h-1, int(round(y2))))
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            writer.write(frame)
        cap.release()
        writer.release()

def save_csvs(frame_paths, boxes, out_dir):
    import csv
    csv_xyxy_path = os.path.join(out_dir, "boxes_xyxy.csv")
    with open(csv_xyxy_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["index", "frame", "x1", "y1", "x2", "y2"])
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = [float(v) for v in b]
            frame_name = frame_paths[i] if i < len(frame_paths) else str(i)
            writer.writerow([i, frame_name, x1, y1, x2, y2])

    csv_xywh_path = os.path.join(out_dir, "boxes_xywh.csv")
    with open(csv_xywh_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["index", "frame", "x", "y", "w", "h"])
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = [float(v) for v in b]
            frame_name = frame_paths[i] if i < len(frame_paths) else str(i)
            writer.writerow([i, frame_name, x1, y1, x2 - x1, y2 - y1])
    print(f"Saved CSV to: {csv_xyxy_path} and {csv_xywh_path}")

def main():
    ensure_deps()
    import torch
    from modelscope.outputs import OutputKeys
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    os.makedirs(OUT_DIR, exist_ok=True)
    # Ensure configuration.json exists (copy if needed)
    cfg_path = maybe_fix_configuration_json(MODEL_DIR)
    # Sanity for weight file
    bin_path = os.path.join(MODEL_DIR, "weight.bin")
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(f"Missing weights: {bin_path}")

    # List frames
    frame_paths = list_frames(FRAMES_DIR)
    print(f"Found {len(frame_paths)} frames under: {FRAMES_DIR}")

    # BBox parse/convert
    if BBOX_FORMAT not in ("xyxy", "xywh"):
        raise ValueError("BBOX_FORMAT must be 'xyxy' or 'xywh'")
    if len(BBOX) != 4:
        raise ValueError("BBOX must be a list of 4 numbers")
    if BBOX_FORMAT == "xywh":
        x, y, w, h = BBOX
        if w <= 0 or h <= 0:
            raise ValueError("Invalid BBOX (xywh): width and height must be > 0")
        x1, y1, x2, y2 = x, y, x + w, y + h
    else:
        x1, y1, x2, y2 = BBOX
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid BBOX (xyxy): must satisfy x2 > x1 and y2 > y1")
    bbox = [float(x1), float(y1), float(x2), float(y2)]
    print(f"Using bbox (xyxy): {bbox}")

    # Pack frames into a temp video
    temp_video = os.path.join(OUT_DIR, "tmp_frames.mp4")
    print(f"Packing frames into: {temp_video} (fps={FPS})")
    frames_to_video(frame_paths, temp_video, FPS)

    # Load & run pipeline
    device = "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from: {MODEL_DIR}")
    tracker = pipeline(Tasks.video_single_object_tracking, model=MODEL_DIR, device=device)
    print("Running tracking...")
    result = tracker((temp_video, bbox))
    boxes = result[OutputKeys.BOXES]

    # Save JSONs
    boxes_path = os.path.join(OUT_DIR, "boxes.json")
    with open(boxes_path, "w", encoding="utf-8") as f:
        json.dump({"boxes": [list(map(float, b)) for b in boxes]}, f, ensure_ascii=False, indent=2)
    print(f"Saved boxes to: {boxes_path}")

    mapping_path = os.path.join(OUT_DIR, "mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({"index_to_frame": {i: frame_paths[i] for i in range(len(frame_paths))}}, f, ensure_ascii=False, indent=2)
    print(f"Saved frame mapping to: {mapping_path}")

    # Save CSVs
    save_csvs(frame_paths, boxes, OUT_DIR)

    # Annotated video
    out_video_path = os.path.join(OUT_DIR, "tracking_result.avi")
    try:
        annotate_video_with_boxes(temp_video, boxes, out_video_path)
        print(f"Saved annotated video to: {out_video_path}")
    except Exception as e:
        print(f"Failed to save annotated video: {e}")

    # Cleanup temp video
    if not KEEP_TMP:
        try:
            os.remove(temp_video)
        except Exception:
            pass

    # Save per-frame images if desired
    if SAVE_FRAMES:
        import cv2
        ann_dir = os.path.join(OUT_DIR, "annotated_frames")
        os.makedirs(ann_dir, exist_ok=True)
        cap = cv2.VideoCapture(temp_video)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or i >= len(boxes):
                break
            x1, y1, x2, y2 = boxes[i]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            base = os.path.basename(frame_paths[i])
            cv2.imwrite(os.path.join(ann_dir, base), frame)
            i += 1
        cap.release()
        print(f"Saved annotated frames to: {ann_dir}")

    print("Done.")

if __name__ == "__main__":
    main()
