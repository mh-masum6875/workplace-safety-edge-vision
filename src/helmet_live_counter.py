from ultralytics import YOLOE
import cv2
import numpy as np
import os

model = YOLOE("yoloe-11l-seg.pt")
classes = ["Person", "helmet"]
model.set_classes(classes, model.get_text_pe(classes))

video_path = r"./data"
cap = cv2.VideoCapture(video_path)

output_path = r"./data"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

CONFIRM_HELMET_FRAMES = 30
GRACE_FRAMES = 10

person_status = {}
person_helmet_frames = {}
next_person_id = 1
frame_index = 0


def is_helmet_on_person(person_box, helmet_box):
    px1, py1, px2, py2 = person_box
    hx1, hy1, hx2, hy2 = helmet_box

    xi1 = max(px1, hx1)
    yi1 = max(py1, hy1)
    xi2 = min(px2, hx2)
    yi2 = min(py2, hy2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    if inter_area > 0:
        return True

    helmet_cx = (hx1 + hx2) / 2
    helmet_cy = (hy1 + hy2) / 2

    if px1 <= helmet_cx <= px2 and py1 <= helmet_cy <= py2:
        return True

    return False


def match_person_to_id(person_box, existing_persons, threshold=100):
    cx = (person_box[0] + person_box[2]) / 2
    cy = (person_box[1] + person_box[3]) / 2

    best_id = None
    min_dist = threshold

    for pid, st in existing_persons.items():
        dist = np.sqrt((cx - st["cx"])**2 + (cy - st["cy"])**2)
        if dist < min_dist:
            min_dist = dist
            best_id = pid

    return best_id


print("Starting Helmet Detection with Live Count...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.25, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)

    person_boxes = []
    helmet_boxes = []

    for i in range(len(boxes)):
        class_name = classes[class_ids[i]]
        if class_name == "Person":
            person_boxes.append(boxes[i])
        elif class_name == "helmet":
            helmet_boxes.append(boxes[i])

    current_frame_ids = set()

    for person in person_boxes:
        x1, y1, x2, y2 = person
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        person_id = match_person_to_id(person, person_status)

        if person_id is None:
            person_id = next_person_id
            next_person_id += 1
            person_status[person_id] = {
                "cx": cx, "cy": cy,
                "confirmed": False,
                "last_frame": frame_index,
                "present_frames": 1,
            }
            person_helmet_frames[person_id] = 0
        else:
            person_status[person_id]["cx"] = cx
            person_status[person_id]["cy"] = cy
            person_status[person_id]["last_frame"] = frame_index
            person_status[person_id]["present_frames"] += 1

        current_frame_ids.add(person_id)

        helmet_detected = False
        for helmet in helmet_boxes:
            if is_helmet_on_person(person, helmet):
                helmet_detected = True
                hx1, hy1, hx2, hy2 = helmet.astype(int)
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 255), 2)
                cv2.putText(frame, "Helmet", (hx1, hy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                break

        if helmet_detected:
            person_helmet_frames[person_id] += 1

        st = person_status[person_id]
        h_frames = person_helmet_frames[person_id]

        if not st["confirmed"] and h_frames >= CONFIRM_HELMET_FRAMES:
            st["confirmed"] = True

        if st["confirmed"]:
            color = (0, 255, 0)
            status_text = f"ID:{person_id} - SAFE"
            alert_text = None
        elif h_frames > 0 or st["present_frames"] < GRACE_FRAMES:
            color = (0, 165, 255)
            status_text = f"ID:{person_id} - CHECKING"
            alert_text = None
        else:
            color = (0, 0, 255)
            status_text = f"ID:{person_id} - NO HELMET"
            alert_text = "NO HELMET!"

        x1, y1, x2, y2 = person.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, status_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if alert_text:
            cv2.rectangle(frame, (x1, y2 + 5), (x2, y2 + 30), (0, 0, 255), -1)
            cv2.putText(frame, alert_text, (x1 + 5, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    ids_to_remove = []
    for pid in person_status:
        if pid not in current_frame_ids and (frame_index - person_status[pid]["last_frame"]) > 30:
            ids_to_remove.append(pid)

    for pid in ids_to_remove:
        del person_status[pid]
        if pid in person_helmet_frames:
            del person_helmet_frames[pid]

    total_persons = len(current_frame_ids)
    with_helmet = sum(1 for pid in current_frame_ids if person_status[pid]["confirmed"])
    without_helmet = sum(
        1 for pid in current_frame_ids
        if not person_status[pid]["confirmed"]
        and person_helmet_frames[pid] == 0
        and person_status[pid]["present_frames"] >= GRACE_FRAMES
    )

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 130), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    cv2.putText(frame, f"Total Person: {total_persons}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"With Helmet: {with_helmet}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Without Helmet: {without_helmet}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow("Helmet Detection - Live Count", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete! Output saved to: {output_path}")