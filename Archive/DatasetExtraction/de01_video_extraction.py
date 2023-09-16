import cv2
import os
import time
import pandas as pd
import random

# Video path location
VIDEO_PATH = '../../DATASET/DATASET_PRIVATE_2_MOD.mp4'
# Boundary face capture
OUTPUT_CAPTURED = '../DATASET/capturedFace'
# Report data extraction
OUTPUT_FACE_REPORT = '../DATASET/face2.csv'
# DECISION TO FRAME SKIPPING WITH 1 of CHANCE
RUMBLE_STATS = True
CHANCE_STATS = 1250
SECOND_SKIPS = 1


def main():
    p_time = 0

    # Load the pre-trained Haar Cascade Classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    os.makedirs(OUTPUT_CAPTURED, exist_ok=True)

    pandas_face_report = []
    if not os.path.exists(OUTPUT_FACE_REPORT):
        with open(OUTPUT_FACE_REPORT, 'w+') as f:
            f.write("Face_ID, X, Y, W, H, Frame \n")
            f.close()

    # Initialize the video capture
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Get the video's frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a resizable window
    cv2.namedWindow('Lite Face Capture', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Lite Face Capture', frame_width, frame_height)

    current_frame = 0

    count_frame = 0
    index = 0

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
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            face_image = frame[y:y + h, x:x + w]
            name_face = f'face_{index}'
            index = index + 1
            cv2.imwrite(os.path.join(OUTPUT_CAPTURED, f'{name_face}.jpg'), face_image)
            pandas_face_report.append((name_face, x, y, w, h, count_frame))

        count_frame = count_frame + 1
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
        elif key & 0xFF == ord('a'):
            current_frame += 60 * cap.get(cv2.CAP_PROP_FPS)  # Jump 1 minute ahead
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)  # Set the frame index
            count_frame = int(count_frame + 60 * cap.get(cv2.CAP_PROP_FPS))

        if RUMBLE_STATS:
            randoms = random.randint(1, CHANCE_STATS)
            if randoms % CHANCE_STATS == 1:
                current_frame += SECOND_SKIPS * cap.get(cv2.CAP_PROP_FPS)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                count_frame = int(count_frame + SECOND_SKIPS * cap.get(cv2.CAP_PROP_FPS))

        print(f'Frames: {count_frame} | data : {index}')

    df = pd.DataFrame(pandas_face_report, columns=['name_face', 'x', 'y', 'w', 'h', 'frame'])
    df.to_csv(OUTPUT_FACE_REPORT, index=False)

    # Release the video capture, file operation, and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
