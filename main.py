import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("./videos/2.mp4")
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(static_image_mode=True, min_tracking_confidence=0.1, max_num_faces=2)
drawSpecs = mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
# drawSpecs = mpDraw.DrawingSpec()

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1200, 720))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpecs, drawSpecs)
            for i, lm in enumerate(faceLms.landmark):
                # print(lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(i, cx, cy)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {str(int(fps))}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
