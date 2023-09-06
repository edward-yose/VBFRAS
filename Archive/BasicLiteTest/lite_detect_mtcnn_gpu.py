import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(keep_all=True, device=device)
detector = MTCNN()

##### Real Data Usage
# VIDEO_FOOTAGE = '../../DATASET/private_dataset_01_mod.mp4'
# ATTENDEE_FACE = '../../DATASET/Attendee'
# OUTPUT_FACE = '../DATASET/capturedFace'

##### Test Data Usage
VIDEO_FOOTAGE = '../DATASET/dummy/youtube_content/02.mp4'
ATTENDEE_FACE = '../DATASET/dummy/image2'
OUTPUT_FACE = '../DATASET/capturedFace'

cap = cv2.VideoCapture(VIDEO_FOOTAGE)

# Get the video's frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

count = 0

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    frame_BGR = frame.copy()