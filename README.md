# 📘 YOLOv8 + DeepSORT Vehicle Tracking
Dự án này thực hiện phát hiện và theo dõi đa đối tượng (Multi-Object Tracking – MOT) trong video giao thông Việt Nam.

Mô hình YOLOv8 được huấn luyện để nhận diện 4 loại phương tiện:
- Class ID: 0, nhãn: Motobike
- Class ID: 1, nhãn: Car
- Class ID: 2, nhãn: Bus
- Class ID: 3, nhãn: Truck. Sau khi phát hiện, các bounding box được đưa vào DeepSORT để gán ID và theo dõi đối tượng xuyên suốt video.

## 🧠 Mô tả kỹ thuật
### 1. Phát hiện (Detection) – YOLOv8
YOLOv8 dự đoán cho mỗi đối tượng:
- Bounding box (x1, y1, x2, y2)
- confidence
- class_id = {0,1,2,3}. Kết quả của YOLO được chuyển sang chuẩn xywh để đưa vào DeepSORT.
### 2. Theo dõi (Tracking) – DeepSORT
DeepSORT tạo track ID bằng:
- Kalman Filter (dự đoán vị trí tiếp theo)
- Appearance Embedding (nhận dạng ngoại hình). Track ID ổn định giúp bạn theo dõi phương tiện xuyên suốt video.
### 3. Vẽ kết quả
Mỗi class có 1 màu cố định:
-   Xe máy	(255, 0, 0) – xanh dương
-   Xe con	(0, 0, 255) – đỏ
-   Xe buýt	(0, 255, 255) – vàng
-   Xe tải	(255, 0, 255) – tím. Hiển thị trên video, dạng: `ID 12 | C0` . Trong đó ID 12 là Track ID cho DeepSORT sinh ra, C0 là class xe máy
## ▶️ Cách chạy tracking YOLOv8 + DeepSORT
### 1. Cài đặt thư viện
``` 
pip install ultralytics
pip install deep-sort-realtime
pip install opencv-python
```
### 2. Chạy file tracking
Lệnh: ` python track_yolov8_deepsort.py` 
Kết quả: cửa sổ hiển thị video tracking real-time, file output_tracking.mp4 được tạo tự động và lưu các predictions thành file .txt để sử dụng cho đánh giá model

### 2. Chạy file evaluate
Lệnh: ` python evaluate.py` 
Kết quả: đánh giá predictions.txt do model tạo ra và VNTraffic_GroundTruth.txt do giảng viên cung cấp
## 📜 Giải thích code chính (track_yolov8_deepsort.py)
### 1. Load YOLO
`model = YOLO("best_85epochs.pt")`
### 2. Khởi tạo DeepSORT
```
 tracker = DeepSort(
        max_age=40,
        n_init=3,
        nn_budget=200,
        max_cosine_distance=0.4,
        max_iou_distance=0.7,
        nms_max_overlap=0.6
    )
```
### 3. Chạy qua từng frame video
`results = model(frame, conf=0.15)[0]`
### 4. Chuyển YOLO → DeepSORT
`detections_for_tracker.append([[x1, y1, w, h], conf, cls])`
### 5. DeepSORT cập nhật track
`tracks = tracker.update_tracks(detections_for_tracker, frame=frame)`
### 6. Vẽ bounding box theo class
```
color = CLASS_COLORS[cls]
cv2.rectangle(...)
```
🎥 Kết quả, video đầu ra có: 
-   Bounding box bao quanh mỗi phương tiện
-   Màu theo classID theo DeepSORT
-   Tracking ổn định xuyên suốt video