import numpy as np
from step_1_dataset import MyDataset
from PIL import Image
import cv2

dataset = MyDataset()
print(len(dataset))

item = dataset[1]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']

print(txt)
print(jpg.shape)
print(hint.shape)

