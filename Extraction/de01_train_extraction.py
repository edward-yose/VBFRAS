import cv2
import os
import time
import pandas as pd
import random
import mtcnn

# Video path location
VIDEO_PATH = '../DATASET/DATASET_PRIVATE_1_MOD.mp4'
CATEGORY = 'V1'
# Boundary face capture
OUTPUT_CAPTURED = '../DATASET/extractedFace'
# Report data extraction
OUTPUT_FACE_REPORT = f'../DATASET/face_{CATEGORY}.csv'

# FRAME SKIPPING IN SECONDS
SECOND_SKIPS = 3

'''
Config:
Video 1 : every 3 seconds (72 Frames) | 105 Minutes
Video 2 : every 1 second (24 Frames)  |  40 Minutes
Video 3 : every 1 second (30 Frames)  |  30 Minutes
'''

# EARLY SKIP FOR EXCLUDE TEST DATASET
MINUTES_SKIP = 30


def haar():
    flag_skip = False
    p_time = 0

    # Load the pre-trained Haar Cascade Classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    os.makedirs(OUTPUT_CAPTURED, exist_ok=True)

    pandas_face_report = []
    if not os.path.exists(OUTPUT_FACE_REPORT):
        with open(OUTPUT_FACE_REPORT, 'w+') as f:
            f.write("Face_ID, X, Y, W, H, frame, source \n")
            f.close()

    # Initialize the video capture
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Get the video's frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get the video length per second
    length = cap.get(cv2.CAP_PROP_FPS)

    # Create a resizable window
    cv2.namedWindow('Lite Face Capture', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Lite Face Capture', frame_width, frame_height)

    current_frame = 0
    skipped_frame = 0
    count_frame = 0
    index = 0

    total_frame = getTotalFrames()

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
            name_face = f'face_{CATEGORY}_{index}'
            index = index + 1
            cv2.imwrite(os.path.join(OUTPUT_CAPTURED, f'{name_face}.jpg'), face_image)
            pandas_face_report.append((name_face, x, y, w, h, count_frame, CATEGORY))

        count_frame = count_frame + 1
        c_time = time.time()
        fps = (c_time - p_time)
        p_time = c_time
        cv2.putText(frame, f'Compute: {fps} s / frame', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        # Display the frame with detected faces
        cv2.imshow('Lite Face Capture', frame)

        # Capture key press
        key = cv2.waitKey(1)

        # Check if 'q' key is pressed to exit
        if key & 0xFF == ord('q'):
            break

        current_frame += SECOND_SKIPS * length - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        count_frame = int(count_frame + SECOND_SKIPS * length - 1)

        # Early Skipping
        if not flag_skip:
            flag_skip = True
            current_frame += MINUTES_SKIP * 60 * length
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            count_frame = int(count_frame + MINUTES_SKIP * 60 * length)

        print(
            f'Frames: {count_frame} of {total_frame} (-{skipped_frame})| Progress: {((count_frame - skipped_frame) / (total_frame - skipped_frame) * 100): .2f}% | data : {index}')

    df = pd.DataFrame(pandas_face_report, columns=['name_face', 'x', 'y', 'w', 'h', 'frame', 'source'])
    df.to_csv(OUTPUT_FACE_REPORT, index=False)

    # Release the video capture, file operation, and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()


def mtcnns():
    flag_skip = False
    p_time = 0

    detector = mtcnn.MTCNN()

    os.makedirs(OUTPUT_CAPTURED, exist_ok=True)

    pandas_face_report = []
    if not os.path.exists(OUTPUT_FACE_REPORT):
        with open(OUTPUT_FACE_REPORT, 'w+') as f:
            f.write("Face_ID, X, Y, W, H, frame, source \n")
            f.close()

    # Initialize the video capture
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Get the video's frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get the video length per second
    length = cap.get(cv2.CAP_PROP_FPS)

    # Create a resizable window
    cv2.namedWindow('Lite Face Capture', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Lite Face Capture', frame_width, frame_height)

    current_frame = 0
    skipped_frame = 0
    count_frame = 0
    index = 0

    total_frame = getTotalFrames()

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if the video has ended
        if not ret:
            break

        # Detect faces in the frame
        faces = detector.detect_faces(frame)

        # Draw rectangles around detected faces
        for face in faces:
            x, y, w, h = face['box']
            # giving 2 pixel outer
            cv2.rectangle(frame, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 255, 0), 1)
            face_image = frame[y:y + h, x:x + w]
            name_face = f'face_{CATEGORY}_{index}'
            index = index + 1
            cv2.imwrite(os.path.join(OUTPUT_CAPTURED, f'{name_face}.jpg'), face_image)
            pandas_face_report.append((name_face, x, y, w, h, count_frame, CATEGORY))

        count_frame = count_frame + 1
        c_time = time.time()
        fps = (c_time - p_time)
        p_time = c_time
        cv2.putText(frame, f'Compute: {fps} s / frame', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        # Display the frame with detected faces
        cv2.imshow('Lite Face Capture', frame)

        # Capture key press
        key = cv2.waitKey(1)

        # Check if 'q' key is pressed to exit
        if key & 0xFF == ord('q'):
            break

        current_frame += SECOND_SKIPS * length
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        count_frame = int(count_frame + SECOND_SKIPS * length)

        # Early Skipping
        if not flag_skip:
            flag_skip = True
            current_frame += MINUTES_SKIP * 60 * length
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            count_frame = int(count_frame + MINUTES_SKIP * 60 * length)
            skipped_frame = count_frame

        print(
            f'Frames: {count_frame} of {total_frame} (-{skipped_frame})| Progress: {((count_frame - skipped_frame) / (total_frame - skipped_frame) * 100): .2f}% | data : {index}')

    df = pd.DataFrame(pandas_face_report, columns=['name_face', 'x', 'y', 'w', 'h', 'frame', 'source'])
    df.to_csv(OUTPUT_FACE_REPORT, index=False)

    # Release the video capture, file operation, and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()


def getTotalFrames():
    import cv2

    # Open the video file
    video = cv2.VideoCapture(VIDEO_PATH)

    # Get the total number of frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Print the total number of frames
    print("Total frames in the video:", total_frames)

    # Don't forget to release the video capture when you're done
    video.release()

    return total_frames


if __name__ == '__main__':
    # haar()
    mtcnns()
    # retinafaces()
