from torch.utils.data import dataset
import os
import cv2
import numpy as np
from utils.utils import list_file_tree


class PathologyDataset(dataset.Dataset):
    """Face Landmarks dataset."""
    '''
    ./dataset/
        train/
            neg/
            pos/
        test/
            neg/
            pos/
    '''

    def __init__(self, root_dir, mode="train", transform=None, augment=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            mode (string): test or train
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mode = mode
        self.root_dir = root_dir
        neg_path = os.path.join(root_dir, mode, "neg")
        pos_path = os.path.join(root_dir, mode, "pos")
        self.neg_list = list_file_tree(neg_path, "jpg")
        self.pos_list = list_file_tree(pos_path, "jpg")
        self.neg_num = len(self.neg_list)
        self.pos_num = len(self.pos_list)
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return self.neg_num + self.pos_num

    def __getitem__(self, idx):
        if idx < self.pos_num:
            image_path = self.pos_list[idx]
            label = np.array(1).reshape([1, 1])
        else:
            image_path = self.neg_list[idx - self.pos_num]
            label = np.array(0).reshape([1, 1])

        input_image = cv2.imread(image_path)
        if self.augment:
            input_image = self.augment(input_image)
        if self.transform:
            input_image = self.transform(input_image)

        input_image = np.array(input_image, dtype=np.float32).transpose((2, 0, 1))[::-1, :, :] / 255.0

        return input_image.copy(), label.astype(np.long)
