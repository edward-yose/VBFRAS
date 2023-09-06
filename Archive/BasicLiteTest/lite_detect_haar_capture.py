import cv2
import os
import time

p_time = 0

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Path to your pre-recorded video file
video_path = '../../DATASET/private_dataset_01_mod.mp4'

# Boundary face capture
output_captured = '../DATASET/capturedFace'
os.makedirs(output_captured, exist_ok=True)

# Initialize the video capture
cap = cv2.VideoCapture(video_path)

# Get the video's frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a resizable window
cv2.namedWindow('Lite Face Capture', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Lite Face Capture', frame_width, frame_height)

current_frame = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(24, 24))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_image = frame[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(output_captured, f'face_{len(os.listdir(output_captured))}.jpg'), face_image)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    # Display the frame with detected faces
    cv2.imshow('Lite Face Capture', frame)

    # Capture key press
    key = cv2.waitKey(1)

    # Check if 'q' key is pressed to exit
    if key & 0xFF == ord('q'):
        break
    # Check if 'd' key is pressed to skip 10 minutes (600 seconds)
    elif key & 0xFF == ord('d'):
        current_frame += 600 * cap.get(cv2.CAP_PROP_FPS)  # Jump 10 minutes ahead
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)  # Set the frame index
    elif key & 0xFF == ord('s'):
        current_frame += 60 * cap.get(cv2.CAP_PROP_FPS)  # Jump 1 minute ahead
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)  # Set the frame index
    elif key & 0xFF == ord('a'):
        current_frame += 10 * cap.get(cv2.CAP_PROP_FPS)  # Jump 10 seconds ahead
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)  # Set the frame index

# Release the video capture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()