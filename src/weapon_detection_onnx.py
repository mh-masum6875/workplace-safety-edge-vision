"""
pip install Ultralytics
"""

import cv2
import numpy as np
import os
from ultralytics import YOLO

MODEL_PATH = r"model.onnx"
video_path = r"Weapon_result_CV_Engineer_model.mp4"
output_path = r"Weapon_detection_onnx.mp4"

WEAPON_CLASSES = ["weapon", "gun", "knife"]
WEAPON_CONF = 0.03
IOU_NMS = 0.45
ALERT_FRAMES = 5
TRACK_IOU_THRESH = 0.3

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(video_path)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def bbox_iou_single(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)


def bbox_iou(box1, boxes):
    xi1 = np.maximum(box1[0], boxes[:, 0])
    yi1 = np.maximum(box1[1], boxes[:, 1])
    xi2 = np.minimum(box1[2], boxes[:, 2])
    yi2 = np.minimum(box1[3], boxes[:, 3])
    inter = np.maximum(0, xi2 - xi1) * np.maximum(0, yi2 - yi1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter
    return inter / np.maximum(union, 1e-6)


def nms_boxes(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return np.array([], dtype=int)
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = bbox_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return np.array(keep)


next_track_id = 0
active_tracks = {}

print("Starting Weapon Detection (ONNX)...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=WEAPON_CONF, iou=IOU_NMS, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    confidences = results.boxes.conf.cpu().numpy()
    names = results.names if results.names else {}

    w_boxes, w_confs, w_classes = [], [], []
    for i in range(len(boxes)):
        class_name = names.get(class_ids[i], f"cls_{class_ids[i]}")
        if class_name in WEAPON_CLASSES and confidences[i] >= WEAPON_CONF:
            w_boxes.append(boxes[i])
            w_confs.append(confidences[i])
            w_classes.append(class_name)

    if len(w_boxes) > 0:
        w_boxes_arr = np.array(w_boxes)
        w_confs_arr = np.array(w_confs)
        keep = nms_boxes(w_boxes_arr, w_confs_arr, iou_threshold=IOU_NMS)
        w_boxes = [w_boxes[k] for k in keep]
        w_confs = [w_confs[k] for k in keep]
        w_classes = [w_classes[k] for k in keep]

    matched_tracks = set()
    det_to_track = {}

    for di in range(len(w_boxes)):
        best_iou = 0
        best_tid = -1
        for tid, trk in active_tracks.items():
            iou = bbox_iou_single(w_boxes[di], trk["box"])
            if iou > best_iou and tid not in matched_tracks:
                best_iou = iou
                best_tid = tid
        if best_iou >= TRACK_IOU_THRESH:
            det_to_track[di] = best_tid
            matched_tracks.add(best_tid)
        else:
            det_to_track[di] = next_track_id
            next_track_id += 1

    new_tracks = {}
    for di in range(len(w_boxes)):
        tid = det_to_track[di]
        prev_count = active_tracks[tid]["count"] if tid in active_tracks else 0
        new_tracks[tid] = {
            "box": w_boxes[di],
            "class": w_classes[di],
            "count": prev_count + 1,
        }
    active_tracks = new_tracks

    weapon_alert = False
    for di in range(len(w_boxes)):
        box = w_boxes[di]
        cls_name = w_classes[di]
        tid = det_to_track[di]

        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame, cls_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
        )

        if active_tracks[tid]["count"] >= ALERT_FRAMES:
            weapon_alert = True

    if weapon_alert:
        cv2.putText(
            frame,
            "WEAPON ALERT",
            (width // 2 - 150, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (width - 10, 80), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    cv2.putText(
        frame,
        f"Weapons: {len(w_boxes)}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    out.write(frame)
    cv2.imshow("Weapon Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processing complete! Output saved to: {output_path}")
