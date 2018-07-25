from __future__ import division

import argparse
import time

import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augment_data import random_flip
from models import *
from test import test
from utils.parse_config import *
from utils.txt_dataset import TextDataSet
from utils.utils import *
from utils.vis_tool import visdom_bbox, Visualizer

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
parser.add_argument('--batch_size', type=int, default=6, help='size of each image batch')
parser.add_argument('--parallels', type=list, default=[0, 1], help='GPU to use')
parser.add_argument('--model_config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--data_config_path', type=str, default='config/solo.data', help='path to data config file')
parser.add_argument('--weights_path', type=str, default='checkpoints/YOLO_07241503_0.742206.weights',
                    help='path to weights file')
parser.add_argument('--pretrain_weights_path', type=str, default='weights/yolov3.weights',
                    help='path to pretrain weights file')
parser.add_argument('--class_path', type=str, default='data/solo.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--iou_thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between saving model weights')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                    help='directory where model checkpoints are saved')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
parser.add_argument('--env', type=str, default="YOLO_V3", help='vis_tool name')

opt = parser.parse_args()

cuda = torch.cuda.is_available() and opt.use_cuda
opt.batch_size *= len(opt.parallels)
# Start training
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, opt.parallels))
os.makedirs('output', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
classes = load_classes(opt.class_path)
opt.num_classes = len(classes)
# Get data configuration
data_config = parse_data_config(opt.data_config_path)
pos_path = data_config['pos_data']
# abnormal_path = data_config['abnormal_data']
openslide_path = data_config['openslide']
# Get hyper parameters
hyperparams = parse_model_config(opt.model_config_path)[0]
learning_rate = float(hyperparams['learning_rate'])
momentum = float(hyperparams['momentum'])
decay = float(hyperparams['decay'])
opt.img_size = int(hyperparams['height'])
# burn_in = int(hyperparams['burn_in'])
print(opt)
lr_step = [10, 20, 25]
# Initiate model
model = Darknet(opt.model_config_path, img_size=opt.img_size)
# model.load_pretrain(opt.pretrain_weights_path)
model.load_weights(opt.weights_path)
yolo_loss = model.yolo_loss
# model.apply(weights_init_normal)
if cuda:
    model = nn.DataParallel(model)
    model = model.cuda()

model.train()
color_jitter = torchvision.transforms.ColorJitter(brightness=float(hyperparams['brightness']),
                                                  contrast=float(hyperparams['contrast']),
                                                  saturation=float(hyperparams['saturation']),
                                                  hue=float(hyperparams['hue']))
txtdataset = TextDataSet(openslide_path,
                         pos_path,
                         mode="train",
                         image_size=opt.img_size,
                         transform=random_flip, augment=color_jitter)

test_txtdataset = TextDataSet(openslide_path,
                              pos_path,
                              image_size=opt.img_size,
                              mode="test")

test_dataloader = torch.utils.data.DataLoader(test_txtdataset,
                                              batch_size=opt.batch_size,
                                              shuffle=False,
                                              num_workers=opt.n_cpu)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=0, weight_decay=decay)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_step,
                                              gamma=0.1)
vis_tool = Visualizer(env=opt.env)
print("init vis_tool")
# best_mAP = test(model, test_dataloader, opt)
best_mAP = 0
for epoch in range(opt.epochs):
    dataloader = torch.utils.data.DataLoader(txtdataset,
                                             batch_size=opt.batch_size,
                                             shuffle=True,
                                             num_workers=opt.n_cpu)
    lr_scheduler.step()
    print("lr= " + str(lr_scheduler.get_lr()))
    for batch_i, (imgs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        model.train()
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)
        optimizer.zero_grad()
        output = model(imgs)
        loss = []
        for yolo, out in zip(yolo_loss, output):
            loss.append(yolo(out, targets))
        loss = sum(loss)
        loss.backward()
        optimizer.step()
        if (batch_i + 1) % 100 == 0:
            vis_tool.plot("loss", loss.item())
            model.eval()
            ori_img = imgs[0].cpu().numpy() * 255
            annotations = targets[0, targets[0, :, 3] != 0].cpu().numpy()
            gt_bbox = np.zeros((len(annotations), 4), dtype=np.int)
            gt_label = np.zeros(len(annotations), dtype=np.int)
            for anno in range(len(annotations)):
                gt_label[anno] = annotations[anno, 0]
                gt_bbox[anno, 0] = (annotations[anno, 1] - annotations[anno, 3] / 2) * opt.img_size
                gt_bbox[anno, 1] = (annotations[anno, 2] - annotations[anno, 4] / 2) * opt.img_size
                gt_bbox[anno, 2] = (annotations[anno, 1] + annotations[anno, 3] / 2) * opt.img_size
                gt_bbox[anno, 3] = (annotations[anno, 2] + annotations[anno, 4] / 2) * opt.img_size

            gt_img = visdom_bbox(ori_img, gt_bbox, gt_label)
            vis_tool.img("gt_img", gt_img)
            no_output = False
            with torch.no_grad():
                output = model(imgs[0].view((1, 3, opt.img_size, opt.img_size)))
                all_output = []
                for yolo, out in zip(yolo_loss, output):
                    all_output.append(yolo(out))
                output = torch.cat(all_output, 1)
                output = non_max_suppression(output, len(classes), conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)
                if output[0] is None:
                    no_output = True
                else:
                    output = output[0][output[0][:, 3] != 0].cpu().numpy()
            if no_output:
                pred_img = visdom_bbox(ori_img, [], [])
            else:
                pred_bbox = np.zeros((len(output), 4), dtype=np.float32)
                pred_label = np.zeros(len(output), dtype=np.int)
                pred_scores = np.zeros(len(output), dtype=np.float32)
                for anno in range(len(output)):
                    pred_label[anno] = output[anno, 6]
                    pred_scores[anno] = output[anno, 5]
                    pred_bbox[anno, 0] = output[anno, 0]
                    pred_bbox[anno, 1] = output[anno, 1]
                    pred_bbox[anno, 2] = output[anno, 2]
                    pred_bbox[anno, 3] = output[anno, 3]
                pred_img = visdom_bbox(ori_img, pred_bbox, pred_label, pred_scores)

            vis_tool.img("pred_img", pred_img)
        # model.seen += imgs.size(0)
    mAP = test(model, yolo_loss, test_dataloader, opt)
    vis_tool.plot("mAP", mAP)
    if best_mAP < mAP:
        timestr = time.strftime('%m%d%H%M')
        best_path = '%s/YOLO_%s_%3f.weights' % (opt.checkpoint_dir, timestr, mAP)
        model.module.save_weights(best_path)
        best_mAP = mAP
