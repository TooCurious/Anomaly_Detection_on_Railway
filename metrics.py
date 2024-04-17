import torch
from collections import Counter
import numpy as np
from numpy import ndarray
import pandas as pd
from sklearn.metrics import auc
from skimage import measure
from statistics import mean


def compute_anomaly_detection_metrics(masks: ndarray, amaps: ndarray, num_th: int = 50, desc='Running inference') -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
       Compute the area under the curve of ROC (TPR) and 0 to 0.3 FPR
       Compute the area under the curve of Intersection over Union (IoU) and 0 to 0.3 FPR
    Args:
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df1 = pd.DataFrame([], columns=["pro", "tpr", "iou", "fpr", "threshold"])
    # df2 = pd.DataFrame([], columns=["precision", "recall", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        detections = []
        ground_truths = []

        for binary_amap, mask in zip(binary_amaps, masks):
            i = 0
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)
                ground_truths.append([i, *region.bbox])

            for region in measure.regionprops(measure.label(binary_amap)):
                detections.append([i, *region.bbox])

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        tp_pixels = np.logical_and(masks, binary_amaps).sum()
        tpr = tp_pixels / masks.sum()

        iou = tp_pixels / (masks.sum() + fp_pixels)

        pl_recall, pl_precision = PL(detections, ground_truths)

        df1 = pd.concat([df1, pd.DataFrame({"pro": mean(pros), "tpr": tpr,
                                            "iou": iou, "fpr": fpr, "threshold": th},
                                           index=[0])], ignore_index=True)

        # df2 = pd.concat([df2, pd.DataFrame({"precision": pl_precision, "recall": pl_recall, "threshold": th},
        #                                    index=[0])], ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df1 = df1[df1["fpr"] < 0.3]
    df1["fpr"] = df1["fpr"] / df1["fpr"].max()

    pro_auc = auc(df1["fpr"], df1["pro"])
    roc_auc = auc(df1["fpr"], df1["tpr"])
    iou_auc = auc(df1["fpr"], df1["iou"])

    # df2['f-statistic'] = df2.apply(harmonic_mean, axis=1)
    #
    # max_f_index = df2['f-statistic'].idxmax()
    # precision_value = df2.loc[max_f_index, 'precision']
    # recall_value = df2.loc[max_f_index, "recall"]
    # f_value = df2.loc[max_f_index, 'f-statistic']
    # optim_treshold = df2.loc[max_f_index, 'threshold']

    return pro_auc, roc_auc, iou_auc


def PL(detections: list, ground_truths: list, iou_threshold: float = 0.3):
    # detections(list): [[train_idx,x1,y1,x2,y2], ...]
    # ground_truths(list): [[train_idx,x1,y1,x2,y2], ...]

    amount_bboxes = Counter(gt[0] for gt in ground_truths) #+
    amount_bboxes_index = Counter(gt[0] for gt in ground_truths) #+

    for key, val in amount_bboxes.items():
        amount_bboxes[key] = torch.zeros(val)

    for key, val in amount_bboxes_index.items():
        amount_bboxes_index[key] = torch.zeros(val)

    # Initialize TP,FP
    TP = torch.zeros(len(detections)) #+
    FP = torch.zeros(len(detections)) #+

    # TP+FN is the total number of GT boxes in the current category, which is fixed
    total_true_bboxes = len(ground_truths) #+

    for detection_idx, detection in enumerate(detections):

        ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

        num_gts = len(ground_truth_img)

        best_iou = 0
        best_gt_idx = 0
        for idx, gt in enumerate(ground_truth_img):
            # Calculate the IoU of the current prediction box detection and each real box in its picture
            iou = insert_over_union(torch.tensor(detection[1:]), torch.tensor(gt[1:]))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        if best_iou > iou_threshold:
            if amount_bboxes[detection[0]][best_gt_idx] == 0:
                TP[detection_idx] = 1
                amount_bboxes[detection[0]][best_gt_idx] = best_iou
                amount_bboxes_index[detection[0]][best_gt_idx] = detection_idx
            elif best_iou > amount_bboxes[detection[0]][best_gt_idx]:
                TP[detection_idx] = 1
                amount_bboxes[detection[0]][best_gt_idx] = best_iou
                amount_bboxes_index[detection[0]][best_gt_idx] = detection_idx
                TP[int(amount_bboxes_index[detection[0]][best_gt_idx])] = 0
                FP[int(amount_bboxes_index[detection[0]][best_gt_idx])] = 1
            else:
                FP[detection_idx] = 1
        else:
            FP[detection_idx] = 1

    TP = TP.to(torch.int8)
    FP = FP.to(torch.int8)
    assert torch.sum(TP & FP) == 0

    # apply a formula
    recall = torch.sum(TP) / len(ground_truths)
    precision = torch.sum(TP) / len(detections)

    return recall.numpy().astype(np.float64), precision.numpy().astype(np.float64)


def insert_over_union(boxes_preds, boxes_labels):
    # pred_bboxes(list): [[train_idx,class_pred,x1,y1,x2,y2], ...]

    box1_x1 = boxes_preds[..., 0:1]
    box1_y1 = boxes_preds[..., 1:2]
    box1_x2 = boxes_preds[..., 2:3]
    box1_y2 = boxes_preds[..., 3:4]  # shape:[N,1]

    box2_x1 = boxes_labels[..., 0:1]
    box2_y1 = boxes_labels[..., 1:2]
    box2_x2 = boxes_labels[..., 2:3]
    box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Calculate the area of intersection area
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def harmonic_mean(row):
    precision = row['precision']
    recall = row['recall']
    if precision == 0 or recall == 0:
        return 0  # Предотвращение деления на ноль
    return 2 * (precision * recall) / (precision + recall)
