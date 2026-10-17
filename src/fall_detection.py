import cv2
from ultralytics import YOLO
from pathlib import Path

source_video = r"./data"
output_video = str(Path(__file__).with_name("falling_detection.mp4"))
model_path = "yolov8x.pt"
fall_threshold = 0


def draw_boxes(img, results, fall_threshold=0):
    overlay = img.copy()
    alpha = 0.8
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for box in results.boxes.data:
        x1, y1, x2, y2, confidence, class_id = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Skip detection if not the target class (person)
        if int(class_id) != 0:
            continue

        w = x2 - x1
        h = y2 - y1

        # Only process bounding boxes where width + height > 600
        if w + h <= 600:
            continue

        is_fall = h - w < fall_threshold
        color = (0, 0, 255) if is_fall else colors[int(class_id) % len(colors)]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=2)

        # Label for fall detection
        label = (
            f"Fall Detected" if is_fall else f"Class {int(class_id)} {confidence:.2f}"
        )
        cv2.putText(
            overlay,
            label,
            (x1, y1 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img


def process_video(
    source_video, output_video, model_path="yolov8n.pt", fall_threshold=0
):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source video: {source_video}")

    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    fps = int(fps) if fps and fps > 0 else 30
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video for write: {output_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, classes=[0], conf=0.2, verbose=False)
        annotated_frame = draw_boxes(frame, results[0], fall_threshold)
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"Video processing complete. Annotated video saved to {output_path}")


process_video(source_video, output_video, model_path, fall_threshold)
