import cv2
import time
import mediapipe as mp

class FaceMeshDetector():
    def __init__(self, static_mode=False, max_faces=2, refine_landmarks=True, min_detection_conf=0.5, min_tracking_conf=0.5):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf = min_tracking_conf
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_mode, self.max_faces, self.refine_landmarks, self.min_detection_conf)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def find_facemesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLMs in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLMs, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)
                    face = []
                    for id, lm in enumerate(faceLMs.landmark):
                        h, w, c = img.shape
                        x,y = int(lm.x * w), int(lm.y * h)
                        face.append([x, y])
                        faces.append(face)
                
        return img, faces


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector(max_faces=2)

    while True:
        success, img = cap.read()
        img, faces = detector.find_facemesh(img)
        if len(faces) != 0:
            print(faces[0])
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
        
            cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            cv2.imshow('Image', img)
            cv2.waitKey(1)


if __name__ == '__main__':
    main()