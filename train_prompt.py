import cv2
import numpy as np
from transformers import pipeline
from PIL import Image

data_txt_path = 'step_2_dataset/data_txt.txt'
original_img_dir = 'step_2_dataset/orig_imgs/'

captioner = pipeline("image-to-text",model="Salesforce/blip-image-captioning-large")

with open(data_txt_path, 'rt') as f, open('step_2_dataset/prompt.txt', 'wt') as f2:
     for line in f:
        img_name = line.split()[0]
        orig_img = cv2.imread(original_img_dir + img_name)
        image_caption = captioner(Image.fromarray(orig_img))[0]['generated_text']
        f2.writelines([img_name,'\t', image_caption,'\n'])