import cv2

def main():

    # Replace 'video_file.mp4' with the path to your video file
    VIDEO_PATH = '../../DATASET/DATASET_PRIVATE_2_MOD.mp4'

    # Open the video file
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Release the video capture object
    cap.release()

    print(f"Total frames in the video: {total_frames}")


if __name__ == '__main__':
    main()
