from ultralytics import YOLOE
import cv2
import numpy as np
import os

model = YOLOE("yoloe-11l-seg.pt")
classes = ["Person", "helmet", "Safety Vest"]
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

person_ppe_status = {}
person_ppe_frames = {}
next_person_id = 1
frame_index = 0


def is_ppe_on_person(person_box, ppe_box):
    px1, py1, px2, py2 = person_box
    gx1, gy1, gx2, gy2 = ppe_box

    xi1 = max(px1, gx1)
    yi1 = max(py1, gy1)
    xi2 = min(px2, gx2)
    yi2 = min(py2, gy2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    if inter_area > 0:
        return True

    ppe_center_x = (gx1 + gx2) / 2
    ppe_center_y = (gy1 + gy2) / 2

    if px1 <= ppe_center_x <= px2 and py1 <= ppe_center_y <= py2:
        return True

    return False


def match_person_to_id(person_box, existing_persons, threshold=100):
    cx = (person_box[0] + person_box[2]) / 2
    cy = (person_box[1] + person_box[3]) / 2

    best_id = None
    min_dist = threshold

    for pid, (px, py, _, _) in existing_persons.items():
        dist = np.sqrt((cx - px)**2 + (cy - py)**2)
        if dist < min_dist:
            min_dist = dist
            best_id = pid

    return best_id


print("Starting PPE Safety Detection...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.05, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)

    person_boxes = []
    helmet_boxes = []
    vest_boxes = []

    for i in range(len(boxes)):
        class_name = classes[class_ids[i]]
        if class_name == "Person":
            person_boxes.append(boxes[i])
        elif class_name == "helmet":
            helmet_boxes.append(boxes[i])
        elif class_name == "Safety Vest":
            vest_boxes.append(boxes[i])

    current_frame_ids = set()

    for person in person_boxes:
        x1, y1, x2, y2 = person
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        person_id = match_person_to_id(person, person_ppe_status)

        if person_id is None:
            person_id = next_person_id
            next_person_id += 1
            person_ppe_status[person_id] = (
                cx, cy,
                {"helmet": False, "vest": False},
                frame_index
            )
            person_ppe_frames[person_id] = {"helmet": 0, "vest": 0}
        else:
            old_status = person_ppe_status[person_id][2]
            person_ppe_status[person_id] = (cx, cy, old_status, frame_index)

        current_frame_ids.add(person_id)

        current_detections = {"helmet": False, "vest": False}

        for helmet in helmet_boxes:
            if is_ppe_on_person(person, helmet):
                current_detections["helmet"] = True
                hx1, hy1, hx2, hy2 = helmet.astype(int)
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 255), 2)
                cv2.putText(frame, "Helmet", (hx1, hy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                break

        for vest in vest_boxes:
            if is_ppe_on_person(person, vest):
                current_detections["vest"] = True
                vx1, vy1, vx2, vy2 = vest.astype(int)
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (255, 0, 255), 2)
                cv2.putText(frame, "Vest", (vx1, vy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                break

        current_frames = person_ppe_frames[person_id]
        for ppe_type in ["helmet", "vest"]:
            if current_detections[ppe_type]:
                current_frames[ppe_type] += 1

        old_status = person_ppe_status[person_id][2]
        final_status = {}
        for ppe_type in ["helmet", "vest"]:
            if old_status[ppe_type]:
                final_status[ppe_type] = True
            elif current_frames[ppe_type] >= 10:
                final_status[ppe_type] = True
            else:
                final_status[ppe_type] = False

        person_ppe_status[person_id] = (cx, cy, final_status, frame_index)
        person_ppe_frames[person_id] = current_frames

        missing_ppe = []
        if not final_status["helmet"]:
            missing_ppe.append("Helmet")
        if not final_status["vest"]:
            missing_ppe.append("Vest")

        if len(missing_ppe) == 0:
            color = (0, 255, 0)
            status_text = f"ID:{person_id} - SAFE"
        else:
            color = (0, 0, 255)
            status_text = f"ID:{person_id} - UNSAFE"

        x1, y1, x2, y2 = person.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        cv2.putText(frame, status_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if missing_ppe:
            missing_text = f"Missing: {', '.join(missing_ppe)}"
            cv2.putText(frame, missing_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            alert_box_height = 25
            cv2.rectangle(frame, (x1, y2 + 5), (x2, y2 + alert_box_height), (0, 0, 255), -1)
            cv2.putText(frame, "UNSAFE!", (x1 + 5, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    ids_to_remove = []
    for pid in person_ppe_status:
        last_frame = person_ppe_status[pid][3]
        if pid not in current_frame_ids and (frame_index - last_frame) > 30:
            ids_to_remove.append(pid)

    for pid in ids_to_remove:
        del person_ppe_status[pid]
        if pid in person_ppe_frames:
            del person_ppe_frames[pid]

    total_persons = len(current_frame_ids)
    safe_count = sum(1 for pid in current_frame_ids if all(person_ppe_status[pid][2].values()))
    unsafe_count = total_persons - safe_count

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (width - 10, 90), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, f"Total People: {total_persons}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Safe: {safe_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Unsafe: {unsafe_count}", (250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow("PPE Safety Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete! Output saved to: {output_path}")