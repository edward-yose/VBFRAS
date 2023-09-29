from keras.models import load_model
from numpy.random import randint
import os
import cv2
import numpy as np
from keras.layers import Conv2D, PReLU, BatchNormalization, Flatten, UpSampling2D, \
    LeakyReLU, Dense, Input, add
from matplotlib import pyplot as plt


def main():
    # Load first n number of images (to train on a subset of all images)
    n = 200000  # pool up to n-th images to train test data
    lr_dir = f"../../../../DATASET/img_align_celeba/lr_images"
    hr_dir = f"../../../../DATASET/img_align_celeba/hr_images"

    lr_list = os.listdir(lr_dir)[n:]

    lr_images = []
    for img in lr_list:
        img_lr = cv2.imread(lr_dir + "/" + img)
        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
        lr_images.append(img_lr)

    hr_list = os.listdir(hr_dir)[n:]

    hr_images = []
    for img in hr_list:
        img_hr = cv2.imread(hr_dir + "/" + img)
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        hr_images.append(img_hr)

    lr_images = np.array(lr_images)
    hr_images = np.array(hr_images)

    lr_images = lr_images / 255.
    hr_images = hr_images / 255.

    hr_shape = (hr_images.shape[1], hr_images.shape[2], hr_images.shape[3])
    lr_shape = (lr_images.shape[1], lr_images.shape[2], lr_images.shape[3])

    lr_ip = Input(shape=lr_shape)
    hr_ip = Input(shape=hr_shape)

    counts = 18

    for i in range(1, counts + 1):
        generator = load_model(f'../srgen_upx4_e_{i}.h5', compile=False)

        [X1, X2] = [lr_images, hr_images]
        # select random example
        ix = randint(0, len(X1), 1)
        src_image, tar_image = X1[ix], X2[ix]

        # generate image from source
        gen_image = generator.predict(src_image)

        # plot all three images

        plt.figure(figsize=(16, 8))
        plt.subplot(231)
        plt.title('LR Image')
        plt.imshow(src_image[0, :, :, :])
        plt.subplot(232)
        plt.title(f'Superresolution epoch: {i}')
        plt.imshow(gen_image[0, :, :, :])
        plt.subplot(233)
        plt.title('Orig. HR image')
        plt.imshow(tar_image[0, :, :, :])

        plt.show()


if __name__ == '__main__':
    main()
