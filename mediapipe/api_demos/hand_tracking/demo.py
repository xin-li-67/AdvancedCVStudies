import cv2
import time
from matplotlib.pyplot import draw
import mediapipe as mp

import hand_tracking_module as htm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.HandDetctor()

while True:
    success, img = cap.read()
    img = detector.find_hands(img, draw=True)
    lmList = detector.find_position(img, draw=False)
    if len(lmList) != 0:
        print(lmList[4])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.imshow('Image', img)
    cv2.waitKey(1)