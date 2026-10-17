import argparse
import time
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from sort import Sort


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def draw_boxes(img, bbox, identities=None, categories=None, confidences=None, fall_threshold=0, colors=None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        tl = opt.thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0

        w = x2 - x1
        h = y2 - y1

        if w + h <= 600:
            continue

        is_fall = h - w < fall_threshold
        color = (0, 0, 255) if is_fall else colors[cat]

        if not opt.nobbox:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

        if not opt.nolabel:
            if is_fall:
                label = f"{id}:Fall Detected" if identities is not None else "Fall Detected"
            else:
                label = f"{id}:Person" if identities is not None else f"Person {confidences[i]:.2f}" if confidences is not None else "Person"
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return img


def process_video(source_video, output_video, model_path, fall_threshold=0, conf_thres=0.25, img_size=640):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    
    print(f"Using provider: {session.get_providers()}")
    
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source video: {source_video}")

    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fps = int(fps) if fps and fps > 0 else 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video for write: {output_path}")

    sort_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    
    start_time = time.time()
    frame_count = 0
    startTime = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        img_preprocessed, ratio, (dw, dh) = letterbox(frame, new_shape=(img_size, img_size), auto=False)
        img_preprocessed = img_preprocessed[:, :, ::-1].transpose(2, 0, 1)
        img_preprocessed = np.ascontiguousarray(img_preprocessed).astype(np.float32) / 255.0
        img_preprocessed = np.expand_dims(img_preprocessed, axis=0)

        outputs = session.run(output_names, {input_name: img_preprocessed})
        
        boxes = []
        output = outputs[0]
        
        if len(output.shape) == 3:
            output = output[0]
        
        if len(output.shape) == 2 and output.shape[1] == 7:
            for detection in output:
                if len(detection) == 7:
                    batch_id, x1, y1, x2, y2, class_id, conf = detection
                    if conf > conf_thres and int(class_id) == 0:
                        x1 = (x1 - dw) / ratio[0]
                        y1 = (y1 - dh) / ratio[1]
                        x2 = (x2 - dw) / ratio[0]
                        y2 = (y2 - dh) / ratio[1]
                        boxes.append([x1, y1, x2, y2, conf, class_id])
        elif len(output.shape) == 2:
            num_det = output.shape[0]
            for i in range(num_det):
                detection = output[i]
                if len(detection) >= 6:
                    x_center, y_center, width, height, conf, *class_scores = detection
                    if conf > conf_thres:
                        class_id = int(np.argmax(class_scores)) if len(class_scores) > 0 else 0
                        class_conf = class_scores[class_id] if len(class_scores) > 0 else conf
                        
                        if class_id == 0 and class_conf > conf_thres:
                            x1 = (x_center - width / 2 - dw) / ratio[0]
                            y1 = (y_center - height / 2 - dh) / ratio[1]
                            x2 = (x_center + width / 2 - dw) / ratio[0]
                            y2 = (y_center + height / 2 - dh) / ratio[1]
                            boxes.append([x1, y1, x2, y2, conf * class_conf, class_id])

        if opt.track and len(boxes) > 0:
            dets_to_sort = np.array(boxes)
            tracked_dets = sort_tracker.update(dets_to_sort, opt.unique_track_color)
            tracks = sort_tracker.getTrackers()

            if len(tracked_dets) > 0:
                bbox_xyxy = tracked_dets[:, :4]
                identities = tracked_dets[:, 8]
                categories = tracked_dets[:, 4]
                confidences = None

                if opt.show_track:
                    for t, track in enumerate(tracks):
                        track_color = colors[int(track.detclass)] if not opt.unique_track_color else sort_tracker.color_list[t]
                        [cv2.line(frame, (int(track.centroidarr[i][0]), int(track.centroidarr[i][1])), 
                                  (int(track.centroidarr[i+1][0]), int(track.centroidarr[i+1][1])),
                                  track_color, thickness=opt.thickness) 
                         for i, _ in enumerate(track.centroidarr) if i < len(track.centroidarr)-1]

                frame = draw_boxes(frame, bbox_xyxy, identities, categories, confidences, fall_threshold, colors)
        else:
            if len(boxes) > 0:
                bbox_xyxy = np.array(boxes)[:, :4]
                identities = None
                categories = np.array(boxes)[:, 5]
                confidences = np.array(boxes)[:, 4]
                frame = draw_boxes(frame, bbox_xyxy, identities, categories, confidences, fall_threshold, colors)

        if opt.show_fps:
            currentTime = time.time()
            fps_display = 1/(currentTime - startTime) if startTime > 0 else 0
            startTime = currentTime
            cv2.putText(frame, f"FPS: {int(fps_display)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        out.write(frame)

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames, detected {len(boxes)} objects...")

    cap.release()
    out.release()
    
    total_time = time.time() - start_time
    print(f"Video processing complete. Processed {frame_count} frames in {total_time:.2f}s")
    print(f"Average FPS: {frame_count/total_time:.2f}")
    print(f"Output saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='source video path')
    parser.add_argument('--weights', type=str, default='yolov7x.onnx', help='onnx model path')
    parser.add_argument('--output', type=str, default='output_fall_track.mp4', help='output video path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--fall-threshold', type=int, default=0, help='fall detection threshold')
    parser.add_argument('--track', action='store_true', help='enable tracking')
    parser.add_argument('--show-track', action='store_true', help='show tracking path')
    parser.add_argument('--show-fps', action='store_true', help='show fps on video')
    parser.add_argument('--thickness', type=int, default=2, help='bounding box thickness')
    parser.add_argument('--nobbox', action='store_true', help='do not show bounding box')
    parser.add_argument('--nolabel', action='store_true', help='do not show label')
    parser.add_argument('--unique-track-color', action='store_true', help='unique color per track')
    
    opt = parser.parse_args()
    
    process_video(
        source_video=opt.source,
        output_video=opt.output,
        model_path=opt.weights,
        fall_threshold=opt.fall_threshold,
        conf_thres=opt.conf_thres,
        img_size=opt.img_size
    )
