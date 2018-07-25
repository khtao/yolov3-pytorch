from __future__ import division
import argparse
from pprint import pprint
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import *
from utils.openslide_dataset import Slide_Dataset
from utils.utils import *


def testslide(model, yolo_loss, slide_path, opt):
    slideset = Slide_Dataset(slide_path, outsize=opt.img_size,
                             level=0, step=600, readsize=608)
    slide_dataloader = DataLoader(slideset,
                                  batch_size=opt.batch_size,
                                  num_workers=opt.n_cpu,
                                  shuffle=False,
                                  pin_memory=True)
    if os.path.exists(opt.slide_result_path) is False:
        os.mkdir(opt.slide_result_path)
    split_path = os.path.split(slide_path)
    split_dir = os.path.split(split_path[0])
    save_dir = os.path.join(opt.slide_result_path, split_dir[1])
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    save_file = os.path.splitext(split_path[1])[0] + ".txt"
    fs = open(os.path.join(save_dir, save_file), mode="w")
    classes = load_classes(opt.class_path)  # Extracts class labels from file
    model.eval()
    with torch.no_grad():
        for ii, (img, position, scale) in tqdm(enumerate(slide_dataloader), total=len(slide_dataloader)):
            img = img.cuda()
            scale = scale.numpy()
            position = position.numpy()
            output = model(img)
            all_output = []
            for yolo, out in zip(yolo_loss, output):
                all_output.append(yolo(out))
            detections = torch.cat(all_output, 1)
            detections = non_max_suppression(detections, len(classes), opt.conf_thres, opt.nms_thres)
            for (pos, ratio, detction) in zip(position, scale, detections):
                if detction is not None:
                    detction = detction.cpu().numpy()
                    for x1, y1, x2, y2, conf, score, label in detction:
                        if score < 0.8 or label == 0:
                            continue
                        label_name = classes[int(label)]
                        x1 = (x1 / ratio) + pos[0, 0]
                        y1 = (y1 / ratio) + pos[0, 1]
                        x2 = (x2 / ratio) + pos[0, 0]
                        y2 = (y2 / ratio) + pos[0, 1]
                        fs.writelines(label_name + ", " + str(score) + ", " +
                                      str(int(x1)) + ", " +
                                      str(int(y1)) + ", " +
                                      str(int(x2)) + ", " +
                                      str(int(y2)) + "\n")
    fs.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='data/solo_samples', help='path to dataset')
    parser.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
    parser.add_argument('--weights_path', type=str, default='weights/YOLO_07251728_0.743716.weights',
                        help='path to weights file')
    parser.add_argument('--parallels', type=list, default=[0, 1], help='GPU to use')
    parser.add_argument('--class_path', type=str, default='data/solo.names', help='path to class label file')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
    parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
    parser.add_argument('--n_cpu', type=int, default=2, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img_size', type=int, default=608, help='size of each image dimension')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
    parser.add_argument('--slide_result_path', type=str, default="./output/YOLO_608_color",
                        help='path to result save')
    opt = parser.parse_args()
    pprint(opt)
    opt.batch_size *= len(opt.parallels)
    # Start training
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, opt.parallels))
    cuda = torch.cuda.is_available() and opt.use_cuda
    os.makedirs('output', exist_ok=True)
    # Set up model
    model = Darknet(opt.config_path, img_size=opt.img_size)
    model.load_weights(opt.weights_path)
    yolo_loss = model.yolo_loss
    if cuda:
        model = nn.DataParallel(model)
        model.cuda()
    file_list = list_file_tree("/home/khtao/data/openslide_data/original_data/20180112 1848000yi",
                               "mrxs")
    for file in file_list:
        testslide(model, yolo_loss, file.rstrip('\n'), opt)
