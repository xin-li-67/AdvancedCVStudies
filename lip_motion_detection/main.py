import cv2
import dlib
import time
import json
import imutils
import numpy as np
import matplotlib.pyplot as plt

from imutils import face_utils
from imutils.video import FileVideoStream

SHAPE_PREDICTOR = 'model/shape_predictor_68_face_landmarks.dat'
(LIPFROM, LIPTO) = (49, 68)
HIGH_THRESHOLD = 0.65
LOW_THRESHOLD = 0.4

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

VC = cv2.VideoCapture(0)
FRAME_RATE = VC.get(cv2.CAP_PROP_FPS)
print(FRAME_RATE)

FRAME_WIDTH = int(VC.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(VC.get(cv2.CAP_PROP_FRAME_HEIGHT))

def lip_aspect_ratio(lip):
    A = np.linalg.norm(lip[2] - lip[9])
    B = np.linalg.norm(lip[4] - lip[7])
    C = np.linalg.norm(lip[0] - lip[6])
    lar = (A + B) / (C * 2.0)

    return lar

def frame_processor(frame):
    frame = imutils.resize(frame, width=460)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(frame_gray, 0)

    Lars = []
    for rect in rects:
        shape = predictor(frame_gray, rect)
        shape = face_utils.shape_to_np(shape)

        # locate lip region
        lip = shape[LIPFROM:LIPTO]
        lar = lip_aspect_ratio(lip)
        Lars.append(lar)

        # get the lip shape
        lip_shape = cv2.convexHull(lip)
        cv2.drawContours(frame, [lip_shape], -1, (0, 255, 0), 1)
        cv2.putText(frame, "LAR: {:.2f}".format(lar), (30, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        
        if lar > HIGH_THRESHOLD or lar < LOW_THRESHOLD:
            cv2.putText(frame, "Mouth is Open!", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

    return Lars, frame

def get_mode(nums):
    counts = np.bincount(nums)
    mode = np.argmax(counts)

    return mode

start = time.time()
LARs = []

while (VC.isOpened()):
    # read frames
    rval, frame = VC.read()
    if rval:
        lars, frame = frame_processor(frame)
        LARs += lars
        cv2.imshow("Frame", frame)
        # control imshow lasting time  Explaination: https://juejin.cn/post/6870776834926051342
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
    else: 
        break

plt.figure(figsize=(100,10))
x = np.arange(len(LARs))
plt.plot(x, LARs,'-')
plt.show()

# # cleanup
cv2.destroyAllWindows()
VC.stop()
print(time.time()-start)