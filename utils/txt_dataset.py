from torch.utils.data import dataset
import os
import openslide
import numpy as np


def get_label(content):
    data = content.split(";")
    region_txt = data[0].split(",")
    region_x = int(float(region_txt[0]))
    region_y = int(float(region_txt[1]))
    image_w = int(float(region_txt[2])) - region_x
    image_h = int(float(region_txt[3])) - region_y
    label_txt = data[1:]
    bbox_list = list()
    cls_list = list()
    for txt in label_txt:
        split_txt = txt.split(",")
        if len(split_txt) > 1:
            x_min = float(split_txt[0])
            y_min = float(split_txt[1])
            x_max = float(split_txt[2])
            y_max = float(split_txt[3])
            cls = int(split_txt[4])
            bbox_list.append(np.array([x_min, y_min, x_max, y_max], dtype=np.float32))
            cls_list.append(cls)
        else:
            cls = int(split_txt[0])
            cls_list.append(np.int32(cls))

    image_region = np.array([region_x, region_y, image_w, image_h])

    return image_region, bbox_list, cls_list


def make_label(image_region, bbox_list, cls_list):
    region_txt = str(image_region[0]) + "," + str(image_region[1]) + "," + \
                 str(image_region[0] + image_region[2]) + "," + str(image_region[1] + image_region[3])
    label_txt = str()
    for i in range(len(bbox_list)):
        bbox = bbox_list[i]
        label = cls_list[i]
        label_txt += ";" + str(int(bbox[0])) + "," + str(int(bbox[1])) + "," + str(
            int(bbox[2])) + "," + str(int(bbox[3])) + "," + str(int(label))
    return region_txt + label_txt


class TextDataSet(dataset.Dataset):
    def __init__(self, slide_path, dataset_path, image_size, is_empty=False, level=0, mode="train", readimage=True,
                 transform=None,
                 augment=None):
        """

        :param slide_path:
        :param dataset_path:
        :param is_empty:
        :param level:
        :param mode:
        :param transform:
        :param augment:
        """
        self.slide_path = slide_path
        self.dataset_path = dataset_path
        self.mode = mode
        if is_empty:
            self.dict_list = list()
        else:
            self.read_dataset()
        self.readimage = readimage
        self.level = level
        self.transform = transform
        self.image_size = image_size
        self.augment = augment
        self.label_names = ('abnormal', 'ascus', 'lsil', 'hsil')
        self.max_objects = 50

    def append(self, slide_name, label):
        unopened = True
        if os.path.isabs(slide_name):
            images_path = slide_name
        else:
            images_path = os.path.join(self.slide_path, slide_name)

        for index in range(len(self.dict_list)):
            if self.dict_list[index]["slide"]._filename == images_path:

                if isinstance(label, list):
                    self.dict_list[index]["label"] += label
                elif isinstance(label, str):
                    self.dict_list[index]["label"].append(label)
                else:
                    TypeError("must be a list or string")

                unopened = False
        if unopened:
            slide_image = openslide.OpenSlide(images_path)

            if isinstance(label, list):
                self.dict_list.append({"slide": slide_image, "label": label})
            elif isinstance(label, str):
                self.dict_list.append({"slide": slide_image, "label": [label]})
            else:
                TypeError("must be a list or string")

    def read_dataset(self):
        fs = open(os.path.join(self.dataset_path, self.mode + ".txt"), "r")
        dict_list = list()
        for line in fs:
            content = line.split(", ")
            #
            if content[0][0] == '#':
                continue
            if len(content) < 2:
                continue
            images_path = os.path.join(self.slide_path, content[0].rstrip('\n'))
            label_path = os.path.join(self.dataset_path, content[1].rstrip('\n'))
            slide_image = openslide.OpenSlide(images_path)
            label_list = open(label_path, "r").read().splitlines()
            dict_list.append({"slide": slide_image, "label": label_list})

        fs.close()
        self.dict_list = dict_list

    def statistic(self):
        allclasses = {}
        for dict_line in self.dict_list:
            for label_line in dict_line["label"]:
                _, _, label_list = get_label(label_line)
                for label_i in label_list:
                    cls = self.label_names[label_i]
                    cls = cls.lower()
                    allclasses[cls] = 1 if cls not in allclasses.keys() else allclasses[cls] + 1
        return allclasses

    def write_dataset(self):

        if os.path.exists(self.dataset_path) is False:
            os.mkdir(self.dataset_path)
        if os.path.exists(os.path.join(self.dataset_path, "label")) is False:
            os.mkdir(os.path.join(self.dataset_path, "label"))
        fs = open(os.path.join(self.dataset_path, self.mode + ".txt"), mode='w')
        for dict_line in self.dict_list:
            slidename = os.path.relpath(dict_line["slide"]._filename, self.slide_path)
            label_name = os.path.join("label", os.path.split(slidename)[-1][:-5] + ".txt")
            fs_label = open(os.path.join(self.dataset_path, label_name), 'w')
            fs.write(slidename + ", " + label_name + "\n")
            for label_line in dict_line["label"]:
                if label_line.rstrip('\n')[-1] == "9":
                    continue
                fs_label.write(label_line.rstrip('\n') + "\n")
            fs_label.close()
        fs.close()

    def export_wrong_txt(self, path=None):
        if path is None:
            path = self.dataset_path

        if os.path.exists(path) is False:
            os.mkdir(path)
        if os.path.exists(os.path.join(path, "label")) is False:
            os.mkdir(os.path.join(path, "label"))
        fs = open(os.path.join(path, self.mode + ".txt"), mode='w')
        for dict_line in self.dict_list:
            slidename = os.path.relpath(dict_line["slide"]._filename, self.slide_path)
            label_name = os.path.join("label", os.path.split(slidename)[-1][:-5] + ".txt")
            fs_label = open(os.path.join(path, label_name), 'w')
            fs.write(slidename + ", " + label_name + "\n")
            for label_line in dict_line["label"]:
                if label_line.rstrip('\n')[-1] != "9":
                    continue
                image_region, bbox, cls = get_label(label_line.rstrip('\n'))
                for i in range(len(bbox)):
                    box, label = bbox[i], cls[i]
                    if label != 9:
                        continue
                    box = box[::-1]
                    box[:2] += image_region[:2]
                    box[2:] += image_region[:2]
                    fs_label.writelines("wrong" + ", " + str(0) + ", " +
                                        str(int(box[2])) + ", " +
                                        str(int(box[3])) + ", " +
                                        str(int(box[0])) + ", " +
                                        str(int(box[1])) + "\n")

            fs_label.close()
        fs.close()

    def __len__(self):
        length = 0
        for dict_line in self.dict_list:
            length += len(dict_line["label"])
        return length

    def __getitem__(self, idx):
        temp = idx
        image, bbox, cls = None, None, None
        for dict_line in self.dict_list:
            if temp - len(dict_line["label"]) < 0:
                image_region, bbox, cls = get_label(dict_line["label"][temp].rstrip('\n'))
                if len(bbox) != 0:
                    bbox = np.stack(bbox)
                    bbox = bbox * self.image_size / image_region[2]
                if self.readimage:
                    image = dict_line["slide"].read_region((image_region[0], image_region[1]), self.level,
                                                           (image_region[2], image_region[3])).convert("RGB")
                    if self.augment:
                        image = self.augment(image)
                    if image_region[2] != self.image_size:
                        image = image.resize((self.image_size, self.image_size))
                    image = np.array(image).astype(np.float32).transpose((2, 0, 1)) / 255.0
                    if self.transform:
                        image, bbox = self.transform(image, bbox)
                    image = image.copy()

                break
            else:
                temp = temp - len(dict_line["label"])

        if len(bbox) == 0:
            return image, np.stack(cls).astype(np.int32)
        else:
            filled_labels = np.zeros((self.max_objects, 5))
            for index in range(len(bbox)):
                center_y = (bbox[index][0] + bbox[index][2]) / 2 / self.image_size
                center_x = (bbox[index][1] + bbox[index][3]) / 2 / self.image_size
                bbox_h = (bbox[index][2] - bbox[index][0]) / self.image_size
                bbox_w = (bbox[index][3] - bbox[index][1]) / self.image_size
                filled_labels[index, :] = [cls[index], center_x, center_y, bbox_w, bbox_h]
            return image, filled_labels

    def __add__(self, other):
        if not isinstance(other, TextDataSet):
            TypeError("must be TextDataSet!")

        for dict in other.dict_list:
            self.append(dict["slide"]._filename, dict["label"])
        return self

    def __delitem__(self, key):
        temp = key
        for index in range(len(self.dict_list)):
            if temp - len(self.dict_list[index]["label"]) < 0:
                del self.dict_list[index]["label"][temp]
                if len(self.dict_list[index]["label"]) == 0:
                    del self.dict_list[index]
                break
            else:
                temp = temp - len(self.dict_list[index]["label"])

    def __setitem__(self, key, val):
        temp = key
        bbox_list, cls_list = val
        for index in range(len(self.dict_list)):
            if temp - len(self.dict_list[index]["label"]) < 0:
                image_region, bbox, cls = get_label(self.dict_list[index]["label"][temp].rstrip('\n'))
                self.dict_list[index]["label"][temp] = make_label(image_region, bbox_list, cls_list)
                break
            else:
                temp = temp - len(self.dict_list[index]["label"])
