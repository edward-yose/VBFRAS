import os
from os import listdir

import h5py
from PIL import Image

import numpy
from numpy import asarray
from numpy import expand_dims

from matplotlib import pyplot
from keras.models import load_model

from keras_facenet import FaceNet

import pickle
import cv2


def main(folder, face_cascade):
    database = {}
    print("FLAG")
    facenet = FaceNet()
    print("FLAG2")
    for filename in listdir(folder):
        path = folder + filename
        img1 = cv2.imread(path)

        face = face_cascade.detectMultiScale(img1, 1.1, 4)

        if len(face) > 0:
            x, y, w, h = face[0]
        else:
            x, y, w, h = 1, 1, 2, 2

        img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_array = asarray(img)

        faces = img_array[y:y + h, x:x + h]
        faces = Image.fromarray(faces)
        faces = face.resize((160, 160))
        faces = asarray(faces)

        faces = expand_dims(faces, axis=0)

        signature = facenet.embeddings(faces)

        database[os.path.splitext(filename)[0]] = signature

        my_file = open("data.pkl", "wb")
        pickle.dump(database, my_file)
        my_file.close()

        print(database)


if __name__ == '__main__':
    ATTENDEE = f'../../DATASET/Attendee'
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    main(ATTENDEE, FACE_CASCADE)

# SOURCE : https://youtu.be/_CfhRzAlHQM
