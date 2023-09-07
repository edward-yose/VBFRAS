import cv2
from mtcnn import MTCNN

detector = MTCNN()

##### Real Data Usage
# VIDEO_FOOTAGE = '../../DATASET/private_dataset_01_mod.mp4'
# ATTENDEE_FACE = '../../DATASET/Attendee'
# OUTPUT_FACE = '../DATASET/capturedFace'

##### Test Data Usage
VIDEO_FOOTAGE = '../DATASET/dummy/youtube_content/02.mp4'
ATTENDEE_FACE = '../DATASET/dummy/image2'
OUTPUT_FACE = '../DATASET/capturedFace'

# Open a video capture object
video_capture = cv2.VideoCapture(VIDEO_FOOTAGE)  # 0 for the default camera, or provide the path to a video file

while True:
    # Read a frame from the video stream
    ret, frame = video_capture.read()
    if not ret:
        break

    # Detect faces in the frame
    faces = detector.detect_faces(frame)

    # Draw bounding boxes around the detected faces
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('Video', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV window
video_capture.release()
cv2.destroyAllWindows()
