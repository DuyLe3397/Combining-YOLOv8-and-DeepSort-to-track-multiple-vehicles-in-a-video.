import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

CLASS_COLORS = {
    0: (255, 0, 0),
    1: (0, 0, 255),
    2: (0, 255, 255),
    3: (255, 0, 255),
}


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = boxA_area + boxB_area - inter
    if union == 0:
        return 0
    return inter / union


def main():

    model = YOLO("best_85epochs.pt")
    print("Model Device:", model.device)

    tracker = DeepSort(
        max_age=40,
        n_init=3,
        nn_budget=200,
        max_cosine_distance=0.4,
        max_iou_distance=0.7,
        nms_max_overlap=0.6
    )

    pred_file = open("predictions.txt", "w")

    video_path = "VNTraffic_Video/VNTraffic_Original-video.mp4"
    cap = cv2.VideoCapture(video_path)

    out = cv2.VideoWriter(
        "output_tracking.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (int(cap.get(3)), int(cap.get(4)))
    )

    frame_idx = 0
    print("Bắt đầu tracking...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.15)[0]

        detections_for_tracker = []
        yolo_boxes = []

        # Lấy BBox từ YOLO
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            w = x2 - x1
            h = y2 - y1

            yolo_boxes.append([x1, y1, w, h, cls, conf])
            detections_for_tracker.append([[x1, y1, w, h], conf, cls])

        # DeepSORT update
        tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

        # Match YOLO box với track
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            track_box = [l, t, r, b]

            best_iou = 0
            best_det = None

            for det in yolo_boxes:
                x1, y1, w, h, cls, conf = det
                det_box = [x1, y1, x1 + w, y1 + h]

                score = iou(track_box, det_box)

                if score > best_iou:
                    best_iou = score
                    best_det = det

            if best_det is None:
                continue

            x1, y1, w, h, cls, conf = best_det

            # Ghi file prediction.txt
            pred_file.write(
                f"{frame_idx},{track_id},{x1},{y1},{w},{h},{conf},-1,-1,-1\n"
            )

            # Vẽ bbox
            x2 = x1 + w
            y2 = y1 + h
            color = CLASS_COLORS.get(cls, (0, 255, 0))

            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), color, 2)

            # ========= CHỈ HIỂN THỊ ID =========
            cv2.putText(frame, f"{track_id}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2)

        out.write(frame)
        cv2.imshow("Tracking", frame)

        frame_idx += 1

        if cv2.waitKey(1) == ord("q"):
            break

    pred_file.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
