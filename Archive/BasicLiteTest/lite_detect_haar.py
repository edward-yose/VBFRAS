import cv2

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Path to your pre-recorded video file
video_path = '../../DATASET/private_dataset_01_mod.mp4'
# video_path = '../DATASET/dummy/youtube_content/01.mp4'

# Initialize the video capture
cap = cv2.VideoCapture(video_path)

# Get the video's frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a resizable window
cv2.namedWindow('Lite Haar Face Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Lite Haar Face Detection', frame_width, frame_height)

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
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with detected faces
    cv2.imshow('Lite Haar Face Detection', frame)

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
