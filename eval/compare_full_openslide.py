import numpy as np
from utils.utils import list_file_tree, muti_bbox_iou
import os


def read_gt(path):
    label_list = open(path, "r").read().splitlines()
    class_name = ("ascus", "lsil", "hsil", "lisl", "hisl")
    bbox_list = list()
    for labe_line in label_list:
        content = labe_line.rstrip('\n').split(", ")
        label_name = content[0].lower()
        if label_name in class_name:
            x_min = float(content[1])
            y_min = float(content[2])
            x_max = float(content[3]) + x_min
            y_max = float(content[4]) + y_min
            bbox_list.append(np.array([x_min, y_min, x_max, y_max], dtype=np.int))
    return np.stack(bbox_list)


def read_predict(path):
    label_list = open(path, "r").read().splitlines()
    class_name = ("ascus", "lsil", "hsil", "lisl", "hisl", "pos")
    bbox_list = list()
    for labe_line in label_list:
        content = labe_line.rstrip('\n').split(", ")
        label_name = content[0].lower()
        if label_name in class_name and float(content[1]) > 0.9:
            x_min = float(content[2])
            y_min = float(content[3])
            x_max = float(content[4])
            y_max = float(content[5])
            bbox_list.append(np.array([x_min, y_min, x_max, y_max], dtype=np.int))
    return np.stack(bbox_list)


def cal_recall():
    gt_path = "/home/khtao/data/openslide_data/annotations"
    predict_path1 = "/media/khtao/workplace/WorkCenter/2018-7/PyTorch-YOLOv3/output/YOLO_608/3dhistech"
    # predict_path1 = "/media/khtao/workplace/WorkCenter/2018-6/faster-rcnn-pytorch/result/god_model/3dhistech"
    gt_file_list = list_file_tree(gt_path, "txt")
    recall_list = list()
    for gt_file in gt_file_list:
        filename = os.path.split(gt_file)[1]
        predict_file1 = os.path.join(predict_path1, filename)
        # predict_file2 = os.path.join(predict_path2, filename)
        gt_bbox = read_gt(gt_file)
        predict_bbox1 = read_predict(predict_file1)
        # predict_bbox2 = read_predict(predict_file2)
        iou = muti_bbox_iou(gt_bbox, predict_bbox1)
        gt_iou = iou.max(axis=1)
        recall = np.sum(gt_iou > 0.5) / len(gt_iou)
        recall_list.append({filename: recall})
    print(recall_list)


if __name__ == '__main__':
    cal_recall()
