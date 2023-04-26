import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.original_img_dir = '/home/kwang/DD_project/SCHP/input/'
        self.labelmap_dir = '/home/kwang/DD_project/SCHP/output/labelmap/'
        self.mask_dir = '/home/kwang/DD_project/SCHP/output/mask/'
        self.data_txt_path = '/home/kwang/DD_project/ControlNet-main/train_pairs_prompt.txt'
        # self.original_img_dir = './step_2_dataset/orig_imgs/'
        # self.labelmap_dir = './step_2_dataset/labelmap/'
        # self.mask_dir = './step_2_dataset/mask/'
        # self.keypoint_dir = './step_2_dataset/keypoints/'
        # self.clothes_dir = './step_2_dataset/clothes/'
        # self.data_txt_path = './step_2_dataset/prompt.txt'

        self.data = []
        self.prompt = []
        with open(self.data_txt_path, 'rt') as f:
            for line in f:
                self.data.append(line.split('\t')[0])
                self.prompt.append(line.split('\t')[1][0:-1])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]


        orig_img = cv2.imread(self.original_img_dir + img_name)
        labelmap = cv2.imread(self.labelmap_dir + img_name[:-4] + '.png')

        mask = cv2.imread(self.mask_dir + img_name[:-4] + '.png', 0)
        canny = cv2.Canny(orig_img, 100, 200)
        masked_canny = canny * (mask / 255)


        image_caption = self.prompt[idx]

        # Do not forget that OpenCV read images in BGR order.
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        labelmap = cv2.cvtColor(labelmap, cv2.COLOR_BGR2RGB)
        
        # concat source
        hint = np.concatenate((labelmap,  np.expand_dims(masked_canny, axis=2)), axis=2)

        # Normalize source images to [0, 1].
        hint = hint.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        orig_img = (orig_img.astype(np.float32) / 127.5) - 1.0

        


        return dict(jpg=orig_img, txt=image_caption, hint=hint)

