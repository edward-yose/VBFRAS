import cv2
import mediapipe as mp
import time


def main():
    # video_path = '../DATASET/dummy/youtube_content/01.mp4'
    video_path = '../../DATASET/private_dataset_01_mod.mp4'

    cap = cv2.VideoCapture(video_path)
    pTime = 0

    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=20)
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        results = faceMesh.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_FACE_OVAL, drawSpec, drawSpec)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        cv2.imshow("Lite Face Landmarking", img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
