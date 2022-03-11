from json import detect_encoding
import os
import cv2
import time

import hand_tracking_module as htm

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
# cap.set(3, wCam)
# cap.set(4, hCam)

# folderPath = 'finger_images'
# myList = os.listdir(folderPath)
# print(myList)

overlayList = []
# for imPath in myList:
#     image = cv2.imread(f'{folderPath}/{imPath}')
#     overlayList.append(image)
# print(len(overlayList))

pTime = 0
detector = htm.HandDetctor(min_detection_confidence=0.75)
tip_ids = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.find_hands(img)
    lmList = detector.find_position(img, draw=False)
    if len(lmList) != 0:
        fingers = []
        # thumb
        if lmList[tip_ids[0]][1] > lmList[tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # rest four fingers
        for id in range(1, 5):
            if lmList[tip_ids[id]][2] < lmList[tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        total_fingers = fingers.count(1)
        print(f'Found total {total_fingers} fingers in site')

        h, w, c = overlayList[total_fingers - 1].shape
        img[0:h, 0:w] = overlayList[total_fingers - 1]
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(total_fingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow('Image', img)
    cv2.waitKey(1)