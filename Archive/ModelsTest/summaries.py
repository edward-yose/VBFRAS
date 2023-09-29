from keras.applications import VGG16
from deepface import DeepFace
from deepface import models


def model_vgg16():
    model = VGG16()
    model.summary()


def model_facenet():
    model = DeepFace.build_model("Facenet")
    model.summary()


if __name__ == '__main__':
    # model_vgg16()
    model_facenet()
