from architecture_facenet import *

import cv2
import random
import numpy as np

from train_facenet import normalize, l2_normalizer
from scipy.spatial.distance import cosine

import pickle
import time

import mtcnn
from retinaface import RetinaFace

confidence_t = 0.99
recognition_t = 0.5
required_size = (160, 160)

# DECISION TO FRAME SKIPPING WITH 1 of CHANCE
RUMBLE_STATS = True
CHANCE_STATS = 1
SECOND_SKIPS = 1


def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)


def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode


def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict


def deploy(img, detector, encoder, encoding_dict):
    # MTCNN
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)

    # results = RetinaFace.detect_faces(img_rgb)

    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + f'| {distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
    return img


if __name__ == "__main__":
    required_shape = (160, 160)
    face_encoder = InceptionResNetV1()
    path_m = "../../Models/Saved/facenet_keras_weights.h5"
    face_encoder.load_weights(path_m)
    encodings_path = 'encodings.pkl'

    face_detector = mtcnn.MTCNN()

    encoding_dict = load_pickle(encodings_path)

    cap = cv2.VideoCapture(0)

    # Get the video's frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('camera', frame_width, frame_height)

    current_frame = 0
    count_frame = 0
    index = 0
    p_time = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("CAM NOT OPENED")
            break

        # face_detector = RetinaFace.detect_faces(frame)

        # TODO: Main | Check Test Method, since detecting feels slow
        # TODO: Main | Transfer all models to TensorLite for EdgeSystems
        frame = deploy(frame, face_detector, face_encoder, encoding_dict)

        count_frame = count_frame + 1
        c_time = time.time()
        proc_time = (c_time - p_time)
        p_time = c_time

        cv2.putText(frame, f'Processing Time: {float(proc_time):.2f}', (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0),
                    1)

        cv2.imshow('camera', frame)

        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('a'):
            current_frame += 60 * cap.get(cv2.CAP_PROP_FPS)  # Jump 1 minute ahead
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)  # Set the frame index
            count_frame = int(count_frame + 60 * cap.get(cv2.CAP_PROP_FPS))
        if RUMBLE_STATS:
            randoms = random.randint(1, CHANCE_STATS)
            if randoms % CHANCE_STATS == 0:
                current_frame += SECOND_SKIPS * cap.get(cv2.CAP_PROP_FPS)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                count_frame = int(count_frame + SECOND_SKIPS * cap.get(cv2.CAP_PROP_FPS))

        print(f'Frames: {count_frame} | data : {index}')
