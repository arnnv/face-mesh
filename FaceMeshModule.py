import cv2
import mediapipe as mp
import time


class FaceMeshDetector:
    def __init__(self, static_image_mode=False, max_num_faces=2, refine_landmarks=False, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=static_image_mode, max_num_faces=max_num_faces,
                                                 refine_landmarks=refine_landmarks,
                                                 min_detection_confidence=min_detection_confidence,
                                                 min_tracking_confidence=min_tracking_confidence)

    def getFaceMesh(self, img, draw=True):
        img = cv2.resize(img, (1200, 720))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.faceMesh.process(imgRGB)
        faces = []

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS)

                face = []
                for lm in faceLms.landmark:
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    face.append([cx, cy])
                faces.append(face)

        return img, faces


def main():
    cap = cv2.VideoCapture("./videos/3.mp4")
    pTime = 0

    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img, faces = detector.getFaceMesh(img, draw=True)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
