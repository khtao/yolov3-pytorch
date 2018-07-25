from __future__ import division

from tqdm import tqdm

from utils.eval_tool import eval_detection_voc
from utils.utils import *


def test(model, yolo_loss, dataloader, opt):
    model.eval()
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels = list(), list()
    for batch_i, (imgs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        imgs = imgs.cuda()
        targets = targets.numpy()

        with torch.no_grad():
            output = model(imgs)
            all_output = []
            for yolo, out in zip(yolo_loss, output):
                all_output.append(yolo(out))
            output = torch.cat(all_output, 1)
            output = non_max_suppression(output, opt.num_classes, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)

        # Compute average precision for each sample
        for sample_i in range(targets.shape[0]):

            # Get labels for sample where width is not zero (dummies)
            annotations = targets[sample_i, targets[sample_i, :, 3] != 0]
            # Extract detections
            detections = output[sample_i]

            if detections is None:
                # If there are no detections but there are annotations mask as zero AP
                detection_bboxes = np.zeros((0, 4))
                detection_labels = np.zeros(0)
                detection_scores = np.zeros(0)
            else:
                detections = detections.cpu().numpy()
                detection_bboxes = detections[:, 0:4]
                detection_scores = detections[:, 5]
                detection_labels = detections[:, 6]
            # Extract target boxes as (x1, y1, x2, y2)
            target_boxes = np.zeros((len(annotations), 4))
            target_boxes[:, 0] = (annotations[:, 1] - annotations[:, 3] / 2)
            target_boxes[:, 1] = (annotations[:, 2] - annotations[:, 4] / 2)
            target_boxes[:, 2] = (annotations[:, 1] + annotations[:, 3] / 2)
            target_boxes[:, 3] = (annotations[:, 2] + annotations[:, 4] / 2)
            target_boxes *= opt.img_size
            target_labels = annotations[:, 0]
            pred_bboxes.append(detection_bboxes)
            pred_labels.append(detection_labels.astype(np.int))
            pred_scores.append(detection_scores)
            gt_bboxes.append(target_boxes)
            gt_labels.append(target_labels.astype(np.int))
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels,
        use_07_metric=True)
    print("Mean Average Precision:" + str(result))
    return result["map"]
