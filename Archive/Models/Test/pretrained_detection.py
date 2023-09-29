from retinaface import RetinaFace
import mtcnn
import cv2
from matplotlib import pyplot as plt


def face_mtcnn(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Use MTCNN to detect faces
    face_detector = mtcnn.MTCNN()
    results = face_detector.detect_faces(img_rgb)

    # Process the results
    for result in results:
        # Extract bounding box coordinates
        x, y, width, height = result['box']
        cv2.rectangle(img, (int(x), int(y)), (int(x + width), int(y + height)), (0, 255, 0), 1)

    cv2.imshow('MTCNN Face Detection', img)
    cv2.waitKey(10000)


# def face_retinaface(img):
#     # Load an image (img_rgb) here
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # Use RetinaFace to detect faces
#     rf_detector = RetinaFace.detect_faces(img_rgb)
#
#     # Process the results
#     for face in rf_detector:
#         # Extract bounding box coordinates
#         x, y, x1, y1, x2, y2 = face['facial_area']
#         cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)
#
#         # Draw bounding box on the image if needed
#
#         # Extract facial landmarks and their confidence scores
#         facial_landmarks = face['landmarks']
#         confidence_scores = face['scores']
#
#     cv2.imshow('Lite retinaface detection', img)
#
#     cv2.waitKey(10000)

def face_retinaface(frame):
    # Create a RetinaFace detector object
    rf_detector = RetinaFace.detect_faces(frame)

    print(rf_detector)

    # for face in rf_detector:
    #     # print(face['facial_area'])

    cv2.imshow('RetinaFace Face Detection', rf_detector)
    cv2.waitKey(10000)


def haar_cascade(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

    cv2.imshow('Lite Haar Face Detection', frame)

    cv2.waitKey(10000)


def retinaface_extraction(img):
    pass


if __name__ == '__main__':
    IMG = '../../DATASET/dummy/many_face.jpg'
    img = cv2.imread(IMG)
    # face_mtcnn(img)
    # haar_cascade(img)
    face_retinaface(img)
    # retinaface_extraction(img)