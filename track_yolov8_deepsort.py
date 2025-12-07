import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# ------------------------
#  MÀU CHO TỪNG CLASS
# ------------------------
CLASS_COLORS = {
    0: (255, 0, 0),      # Xanh dương (Motorbike)
    1: (0, 0, 255),      # Đỏ (Car)
    2: (0, 255, 255),    # Vàng (Bus)
    3: (255, 0, 255),    # Tím (Truck)
}


def main():

    model = YOLO("best_85epochs.pt")
    print("Model device:", model.device)

    tracker = DeepSort(
        max_age=30,
        nn_budget=100,
        n_init=3,
        max_cosine_distance=0.4,
    )

    video_path = "VNTraffic_Video/VNTraffic_Original-video.mp4"
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output_tracking.mp4", fourcc, 30,
                          (int(cap.get(3)), int(cap.get(4))))

    print("Bắt đầu tracking...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections_for_tracker = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            w = x2 - x1
            h = y2 - y1

            detections_for_tracker.append([
                [x1, y1, w, h],  # cho DeepSORT
                conf,
                cls
            ])

        tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

        # ----------------------------
        # Vẽ bbox + màu theo class
        # ----------------------------
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            cls = track.get_det_class()    # lấy class từ detection

            # Lấy màu theo class
            color = CLASS_COLORS.get(cls, (0, 255, 0))   # fallback màu xanh lá

            # Vẽ bbox
            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)),
                          color, 2)

            # Vẽ ID + class
            cv2.putText(frame,
                        f"ID {track_id} | C{cls}",
                        (int(l), int(t) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)

        out.write(frame)
        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
