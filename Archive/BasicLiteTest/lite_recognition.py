import sys
import cv2
import os
import face_recognition
import numpy as np
import math
import time

# VideoFile set Integer for n-th camera live
# videoFile = '../DATASET/dummy/youtube_content/02.mp4'
videoFile = '../../DATASET/private_dataset_01_mod.mp4'
# List database attendee folder
# AttendeeFolder = '../DATASET/dummy/image2'
AttendeeFolder = '../../DATASET/Attendee'

p_time = 0
current_frame = 0


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []

    known_face_encds = []
    known_face_names = []
    process_cur_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        file_location = AttendeeFolder
        for image in os.listdir(file_location):
            face_image = face_recognition.load_image_file(f'{file_location}/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encds.append(face_encoding)
            self.known_face_names.append(image)

        print(self.known_face_names)

    def run_recognition(self):
        global p_time
        global current_frame

        video_path = videoFile
        video_capture = cv2.VideoCapture(video_path)

        if not video_capture.isOpened():
            sys.exit("Video source not found")

        print("Video loaded and start executing")
        while True:
            ret, frame = video_capture.read()

            if self.process_cur_frame:
                current_frame = current_frame+1
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Find all face
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encds, face_encoding)
                    name = 'Unknown'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(self.known_face_encds, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    self.face_names.append(f'{name} ({confidence})')

            self.process_cur_frame = not self.process_cur_frame

            # Display Annotation
            for (t, r, b, l), name in zip(self.face_locations, self.face_names):
                t *= 4
                r *= 4
                b *= 4
                l *= 4

                cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 2)
                cv2.rectangle(frame, (l, b - 30), (r, b), (0, 0, 255), -1)
                cv2.putText(frame, name, (l + 4, b - 4), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            c_time = time.time()
            fps = 1 / (c_time - p_time)
            p_time = c_time
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
            cv2.putText(frame, f'Frame: {int(current_frame)}', (700, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 1)

            cv2.imshow('Lite Face Recognition', frame)

            # Capture key press
            key = cv2.waitKey(1)

            # Check if 'q' key is pressed to exit
            if key & 0xFF == ord('q'):
                break
            # Check if 'd' key is pressed to skip 10 minutes (600 seconds)
            elif key & 0xFF == ord('d'):
                current_frame += 600 * video_capture.get(cv2.CAP_PROP_FPS)  # Jump 10 minutes ahead
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)  # Set the frame index
            elif key & 0xFF == ord('s'):
                current_frame += 60 * video_capture.get(cv2.CAP_PROP_FPS)  # Jump 1 minute ahead
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)  # Set the frame index
            elif key & 0xFF == ord('a'):
                current_frame += 10 * video_capture.get(cv2.CAP_PROP_FPS)  # Jump 10 seconds ahead
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)  # Set the frame index

        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
