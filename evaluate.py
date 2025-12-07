import pandas as pd
import numpy as np
import motmetrics as mm


# ============================================================
# PATCH iou_matrix để tương thích NumPy 2.0 (thay np.asfarray)
# ============================================================

def iou_matrix_fixed(objs, hyps, max_iou=0.5):
    """Tính IoU giữa GT và Prediction (dạng [x, y, w, h])"""
    # Convert sang float (thay np.asfarray → np.asarray)
    objs = np.asarray(objs, dtype=float)
    hyps = np.asarray(hyps, dtype=float)

    if len(objs) == 0 or len(hyps) == 0:
        return np.empty((len(objs), len(hyps)))

    # Chuyển xywh → xyxy
    objs_x1y1 = objs[:, :2]
    objs_x2y2 = objs[:, :2] + objs[:, 2:4]

    hyps_x1y1 = hyps[:, :2]
    hyps_x2y2 = hyps[:, :2] + hyps[:, 2:4]

    # Tính IoU
    tl = np.maximum(objs_x1y1[:, None, :], hyps_x1y1[None, :, :])
    br = np.minimum(objs_x2y2[:, None, :], hyps_x2y2[None, :, :])

    wh = np.maximum(0., br - tl)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_obj = objs[:, 2] * objs[:, 3]
    area_hyp = hyps[:, 2] * hyps[:, 3]

    union = area_obj[:, None] + area_hyp[None, :] - inter
    iou = inter / union

    # Chuyển sang dạng distance (1 - IoU)
    dists = 1 - iou

    # Bất kỳ IoU < max_iou coi như không match → distance = np.nan
    dists[iou < max_iou] = np.nan

    return dists


# Gắn patch vào motmetrics
mm.distances.iou_matrix = iou_matrix_fixed


# ============================================================
#               BẮT ĐẦU EVALUATE
# ============================================================

gt_path = "VNTraffic_Video/VNTraffic_GroundTruth.txt"
pred_path = "predictions.txt"

gt = pd.read_csv(
    gt_path,
    header=None,
    names=["frame", "id", "x", "y", "w", "h", "conf", "a1", "a2", "a3"]
)

pred = pd.read_csv(
    pred_path,
    header=None,
    names=["frame", "id", "x", "y", "w", "h", "conf", "a1", "a2", "a3"]
)

acc = mm.MOTAccumulator(auto_id=True)

# danh sách frame gộp GT + Pred
frames = sorted(set(gt["frame"]).union(set(pred["frame"])))

for frame in frames:

    gt_f = gt[gt["frame"] == frame]
    pred_f = pred[pred["frame"] == frame]

    gt_ids = gt_f["id"].tolist()
    pred_ids = pred_f["id"].tolist()

    gt_boxes = gt_f[["x", "y", "w", "h"]].values
    pred_boxes = pred_f[["x", "y", "w", "h"]].values

    # dùng phiên bản đã patch
    d = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)

    acc.update(gt_ids, pred_ids, d)

mh = mm.metrics.create()
summary = mh.compute(
    acc,
    metrics=["num_frames", "mota", "motp", "idf1", "precision", "recall"],
    name="summary"
)

print(summary)
