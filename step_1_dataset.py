import cv2
import numpy as np
from PIL import Image
import json

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.original_img_dir = './step_2_dataset/orig_imgs/'
        self.labelmap_dir = './step_2_dataset/labelmap/'
        self.mask_dir = './step_2_dataset/mask/'
        self.keypoint_dir = './step_2_dataset/keypoints/'
        self.clothes_dir = './step_2_dataset/clothes/'
        self.data_txt_path = './step_2_dataset/prompt.txt'

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


        orig_img = cv2.imread(self.labelmap_dir + img_name[:-4] + '.png')
        # hard copy of original image
        labelmap = orig_img.copy()

        #labelmap = cv2.imread(self.labelmap_dir + img_name[:-4] + '.png')
        cloth_img = cv2.imread(self.clothes_dir + img_name[:-6] + '_1.jpg')

        # masked labelmap
        mask = cv2.imread(self.mask_dir + img_name[:-4] + '.png', 0)
        for i in range(labelmap.shape[2]):
            labelmap[:,:,i] = labelmap[:,:,i] * (mask / 255)

        # keypoint map
        keypoint_map = np.zeros((orig_img.shape[0], orig_img.shape[1], 18))
        with open(self.keypoint_dir + img_name[:-6] + '_2.json', 'r') as f:
            keypoint_list = json.loads(f.readline())['keypoints']
            for keypoint in keypoint_list:
                if int(keypoint[3])== -1:
                    continue
                x = int(keypoint[1]) * 2
                y = int(keypoint[0]) * 2
                channel = int(keypoint[3])
                keypoint_map[max(0, x-5) : min(x+5, orig_img.shape[0])-1, max(0, y-5) : min(y+5, orig_img.shape[1])-1, channel] = 1

        #img prompt
        image_caption = self.prompt[idx]

        # cv2.imshow('labelmap', labelmap)
        # cv2.waitKey(0)
        # cv2.imshow('orig_img', orig_img)
        # cv2.waitKey(0)

        # Do not forget that OpenCV read images in BGR order.
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        labelmap = cv2.cvtColor(labelmap, cv2.COLOR_BGR2RGB)
        
        # concat source
        hint = np.concatenate((labelmap, keypoint_map, cloth_img) , axis=2)

        # Normalize source images to [0, 1].
        hint = hint.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        orig_img = (orig_img.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=orig_img, txt=image_caption, hint=hint)

