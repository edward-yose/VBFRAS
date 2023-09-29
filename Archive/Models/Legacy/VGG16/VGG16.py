from keras.applications.vgg16 import VGG16
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten


# VGG16 model

def architecture():
    model = Sequential()
    model.add(Conv2D(input_shape=(256, 256, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=10, activation="softmax"))

    print(model.summary())

    img1_shape = (224, 224, 3)
    model_224 = VGG16(include_top=False, weights='imagenet', input_shape=img1_shape)
    print(model_224.summary())


def import_model_vgg16():
    # Import vgg model by not defining an input shape.
    vgg_model = VGG16(include_top=False, weights='imagenet')
    print(vgg_model.summary())


def get_configs():
    vgg_model = VGG16(include_top=False, weights='imagenet')
    vgg_config = vgg_model.get_config()
    print(vgg_config)


if __name__ == '__main__':
    # import_model_vgg16()
    # architecture()
    get_configs()
