from ultralytics import YOLO
import cv2, numpy as np, os, glob

yolo = YOLO("yolov8n.pt")  # downloads on first use

def analyze_frames(frames_dir, fps=1.0):
    frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    tracks = []
    for i, fp in enumerate(frames):
        img = cv2.imread(fp)
        res = yolo.predict(source=img, conf=0.25, verbose=False)[0]
        bboxes = []
        for k in range(len(res.boxes)):
            b = res.boxes.xyxy[k].cpu().numpy()
            c = int(res.boxes.cls[k].cpu().numpy())
            conf = float(res.boxes.conf[k].cpu().numpy())
            cls_name = res.names[c]
            if cls_name == "elephant":
                x1,y1,x2,y2 = b
                bboxes.append({"bbox": (x1,y1,x2,y2), "conf": conf})
        if not bboxes:
            tracks.append(None)
            continue
        largest = max(bboxes, key=lambda bb: (bb["bbox"][2]-bb["bbox"][0])*(bb["bbox"][3]-bb["bbox"][1]))
        x1,y1,x2,y2 = largest["bbox"]
        w,h = (x2-x1), (y2-y1)
        cx, cy = (x1+x2)/2, (y1+y2)/2
        tracks.append({"frame_idx": i, "bbox": largest["bbox"], "w": w, "h": h,
                       "cx": cx, "cy": cy, "conf": largest["conf"]})

    cues = []
    last = None
    for i, t in enumerate(tracks):
        if t is None:
            last = t
            continue
        if last:
            dt = 1.0 / max(fps, 1e-6)
            speed = np.hypot(t["cx"]-last["cx"], t["cy"]-last["cy"]) / dt
            area_now = t["w"]*t["h"]; area_prev = last["w"]*last["h"]
            area_growth = (area_now - area_prev) / max(area_prev, 1e-6)
            wh_ratio_now = t["w"] / max(t["h"], 1e-6)
            wh_ratio_prev = last["w"] / max(last["h"], 1e-6)
            ear_spread = wh_ratio_now > (1.15 * wh_ratio_prev)
            approach = area_growth > 0.4
            still = speed < 10
            cues.append({"time": i/fps, "ear_spread": bool(ear_spread),
                         "approach": bool(approach), "still": bool(still),
                         "det_conf": float(t["conf"])})
        last = t
    return cues
