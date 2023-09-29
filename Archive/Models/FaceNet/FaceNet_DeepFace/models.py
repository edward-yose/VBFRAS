import numpy as np
import cv2
import os
from tqdm import tqdm
from deepface import DeepFace, models


class FaceNet:
    def __init__(self):
        self.model = DeepFace.build_model("Facenet")

    @staticmethod
    def preprocess_img(img):
        img = cv2.resize(img, (96, 112))
        img = (img.astype('float32') - 127.5) / 128.0
        img = np.expand_dims(img, axis=0)
        return img

    def extract_embedding(self, cropped_img, **kwargs):
        embedding = DeepFace.represent(cropped_img, enforce_detection=False, align=False)
        embedding = np.array(embedding)

        return embedding

    def extract_embeddings(self, X, **kwargs):
        X_embeddings = []
        for x in tqdm(X):
            embedding = self.extract_embedding(x)
            X_embeddings.append(embedding)
        return X_embeddings
