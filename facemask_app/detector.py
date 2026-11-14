"""Loads the trained YOLO11n face-mask model and runs detection."""
import cv2
from ultralytics import YOLO

NAMES = {0: "With Mask", 1: "Mask Worn Incorrectly", 2: "Without Mask"}
COLORS = {0: (0, 200, 0), 1: (255, 200, 0), 2: (255, 0, 0)}  # RGB


def load_model(path):
    return YOLO(path)


def detect(model, img_rgb, conf=0.25):
    """Returns (annotated RGB image, list of detections, counts per class)."""
    res = model.predict(source=cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), conf=conf, verbose=False)[0]
    out = img_rgb.copy()
    thick = max(2, int(min(out.shape[:2]) / 250))
    dets, counts = [], {n: 0 for n in NAMES.values()}
    for box, cls, cf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
        x1, y1, x2, y2 = map(int, box.tolist())
        c = int(cls)
        label = f"{NAMES[c]} {float(cf):.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), COLORS[c], thick)
        cv2.putText(out, label, (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 * thick / 2 + 0.3, COLORS[c], max(1, thick - 1))
        dets.append({"Class": NAMES[c], "Confidence": round(float(cf), 3),
                     "Box (x1,y1,x2,y2)": f"{x1},{y1},{x2},{y2}"})
        counts[NAMES[c]] += 1
    return out, dets, counts
