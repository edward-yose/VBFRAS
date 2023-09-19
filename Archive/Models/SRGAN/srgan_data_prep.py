"""
Original ataset from: http://press.liacs.nl/mirflickr/mirdownload.html

Read high-res. original images and save lower versions to be used for SRGAN.

128x128 that will be  used as HR images
32x32 that will be used as LR images
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def check_dir(mstr, hr, lr):
    if not os.path.exists(os.path.join(mstr, hr)):
        os.makedirs(hr)
    if not os.path.exists(os.path.join(mstr, lr)):
        os.makedirs(lr)
    print(f'{hr} and {lr} dir has been made')


def main(train_dir, hr_image, lr_image):
    for img in os.listdir(train_dir + "/data"):
        img_array = cv2.imread(train_dir + "/data/" + img)

        img_array = cv2.resize(img_array, (128, 128))
        lr_img_array = cv2.resize(img_array, (32, 32))
        cv2.imwrite(train_dir + "/" + hr_image + "/" + img, img_array)
        cv2.imwrite(train_dir + "/" + lr_image + "/" + img, lr_img_array)


if __name__ == '__main__':
    TRAIN_DIR = "../../DATASET/LFW_unlabelled"
    HR_DIR = "hr_images"
    LR_DIR = "lr_images"
    check_dir(TRAIN_DIR, HR_DIR, LR_DIR)
    main(TRAIN_DIR, HR_DIR, LR_DIR)
