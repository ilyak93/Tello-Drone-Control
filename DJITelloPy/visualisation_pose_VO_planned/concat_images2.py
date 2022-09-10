import numpy as np
from PIL import Image

def get_concat_h(img_1, img_2):
    h1, w1 = img_1.shape[:2]
    h2, w2 = img_2.shape[:2]

    img_3 = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    img_3[:, :] = (255, 255, 255)

    img_3[:h1, :w1, :3] = img_1
    img_3[:h2, w1:w1 + w2, :3] = img_2
    return img_3

def get_concat_v(img_1, img_2):
    h1, w1 = img_1.shape[:2]
    h2, w2 = img_2.shape[:2]

    img_3 = np.zeros((h1 + h2, max(w1, w2), 3), dtype=np.uint8)
    img_3[:, :] = (255, 255, 255)

    img_3[:h1, :w1, :3] = img_1
    img_3[h1:h1 + h2, :w2, :3] = img_2
    return img_3

import cv2
import os

folder1 = 'data/exp1/combh'
folder2 = 'data/exp1/combh2'
image_paths1 = sorted([os.path.join(folder1, f)
               for f in os.listdir(folder1) if f.endswith('.png')])
image_paths1.remove('data/exp1/combh/10.png')
image_paths1.append('data/exp1/combh/10.png')

image_paths2 = sorted([os.path.join(folder2, f)
               for f in os.listdir(folder2) if f.endswith('.png')])


for i in range(len(image_paths1)):
    im1 = cv2.imread(image_paths1[i])
    im1_rgb = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.imread(image_paths2[0])
    im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    comb = get_concat_v(im2_rgb, im1_rgb)
    comb_im = Image.fromarray(comb)
    comb_im.save("data/exp1/combv/"+str(i)+".png")

