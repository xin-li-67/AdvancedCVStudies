import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:
    success, img = cap.read()
    # # mark the image as not writeable to improve performance
    # img.flags.writeable = False
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLMs in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLMs, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)
    
        for id, lm in enumerate(faceLMs.landmark):
            ih, iw, ic = img.shape
            x, y = int(lm.x * iw), int(lm.y * ih)
            print(id, x, y)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)