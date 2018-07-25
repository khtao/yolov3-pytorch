from torch.utils.data import dataset
import openslide
import numpy as np


class Slide_Dataset(dataset.Dataset):
    def __init__(self, filename, outsize=416, level=0, step=128, readsize=640):
        """

        :param filename: openslide文件名
        :param level: 缩放等级
        :param step: 在缩放后图像上的步长，即在原图上的步长为step*2^level
        :param size: 输出图像的大小
        """
        self.root_dir = filename
        self.readsize = readsize
        self.slide_image = openslide.OpenSlide(filename)
        self.level = level
        self.step = step * (2 ** level)
        self.dimensions = self.slide_image.dimensions
        self.start_piont, self.end_piont = self.get_image_region()
        self.len_x = int((self.end_piont[0] - self.start_piont[0]) / self.step)
        self.len_y = int((self.end_piont[1] - self.start_piont[1]) / self.step)
        self.outsize = outsize

    def get_image_region(self):
        step_x = 100
        step_y = 1000
        precision = 10
        start_piont_x = step_x
        start_piont_y = step_y
        [end_piont_x, end_piont_y] = self.dimensions
        end_piont_x -= step_x
        end_piont_y -= step_y
        while True:
            piont = self.slide_image.read_region((end_piont_x, end_piont_y), self.level, (1, 1)).convert("RGB")
            piont = np.array(piont)
            if piont[0, 0, 0] > 0:
                while True:
                    piont2 = self.slide_image.read_region((int(self.dimensions[0] / 2), end_piont_y), self.level,
                                                          (1, 1)).convert("RGB")
                    piont2 = np.array(piont2)
                    if piont2[0, 0, 0] == 0:
                        break
                    else:
                        end_piont_y += precision
                end_piont_y -= precision
                while True:
                    piont2 = self.slide_image.read_region((end_piont_x, int(self.dimensions[1] / 2)), self.level,
                                                          (1, 1)).convert("RGB")
                    piont2 = np.array(piont2)
                    if piont2[0, 0, 0] == 0:
                        break
                    else:
                        end_piont_x += precision
                end_piont_y -= precision
                break
            else:
                end_piont_x -= step_x
                end_piont_y -= step_y
                end_piont_x = np.maximum(end_piont_x, int(self.dimensions[0] / 2))
                end_piont_y = np.maximum(end_piont_y, int(self.dimensions[1] / 2))

        while True:
            piont = self.slide_image.read_region((start_piont_x, start_piont_y), self.level, (1, 1)).convert("RGB")
            piont = np.array(piont)
            if piont[0, 0, 0] > 0:
                while True:
                    piont2 = self.slide_image.read_region((int(self.dimensions[0] / 2), start_piont_y), self.level,
                                                          (1, 1)).convert(
                        "RGB")
                    piont2 = np.array(piont2)
                    if piont2[0, 0, 0] == 0:
                        break
                    else:
                        start_piont_y -= precision
                start_piont_y += precision
                while True:
                    piont2 = self.slide_image.read_region((start_piont_x, int(self.dimensions[1] / 2)), self.level,
                                                          (1, 1)).convert(
                        "RGB")
                    piont2 = np.array(piont2)
                    if piont2[0, 0, 0] == 0:
                        break
                    else:
                        start_piont_x -= precision
                start_piont_x += precision
                break
            else:
                start_piont_x += step_x
                start_piont_y += step_y
                start_piont_x = np.minimum(start_piont_x, int(self.dimensions[0] / 2))
                start_piont_y = np.minimum(start_piont_y, int(self.dimensions[1] / 2))

        return (start_piont_x, start_piont_y), (end_piont_x, end_piont_y)

    def __len__(self):
        return self.len_x * self.len_y

    def __getitem__(self, idx):
        """

        :param idx:
        :return: 返回相应的level的图像，以及该图像在原始分辨率上的坐标
        """
        position_x = int(idx % self.len_x) * self.step + self.start_piont[0]
        position_y = int(idx / self.len_x) * self.step + self.start_piont[1]
        image = self.slide_image.read_region((position_x, position_y),
                                             self.level,
                                             (self.readsize, self.readsize)).convert("RGB")
        if self.outsize != self.readsize:
            image = image.resize((self.outsize, self.outsize))
        image = np.array(image, dtype=np.float32).transpose((2, 0, 1)) / 255.0
        scale = np.array([self.outsize / self.readsize], dtype=np.float32)
        position = np.array([position_x, position_y], dtype=np.float32).reshape([1, 2])
        return image, position, scale
