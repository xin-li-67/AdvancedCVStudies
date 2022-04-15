import cv2
import time
import mediapipe as mp

# cap = cv2.VideoCapture(0)
# pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
mpDrawStyles = mp.solutions.drawing_styles
mpDrawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2, refine_landmarks=True)

# while True:
#     success, img = cap.read()
#     # # mark the image as not writeable to improve performance
#     # img.flags.writeable = False
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = faceMesh.process(imgRGB)

#     # img.flags.writeable = True
#     # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#     if results.multi_face_landmarks:
#         for faceLMs in results.multi_face_landmarks:
#             mpDraw.draw_landmarks(image=img, landmark_list=faceLMs, connections=mpFaceMesh.FACEMESH_TESSELATION,
#                                   landmark_drawing_spec=None, connection_drawing_spec=mpDrawStyles.get_default_face_mesh_tesselation_style())
#             # mpDraw.draw_landmarks(image=img, landmark_list=faceLMs, connections=mpFaceMesh.FACEMESH_CONTOURS,
#             #                       landmark_drawing_spec=None, connection_drawing_spec=mpDrawStyles.get_default_face_mesh_contours_style())
#             # mpDraw.draw_landmarks(image=img, landmark_list=faceLMs, connections=mpFaceMesh.FACEMESH_IRISES,
#             #                       landmark_drawing_spec=None, connection_drawing_spec=mpDrawStyles.get_default_face_mesh_iris_connections_style())
    
#         for id, lm in enumerate(faceLMs.landmark):
#             ih, iw, ic = img.shape
#             x, y = int(lm.x * iw), int(lm.y * ih)
#             print(id, x, y)
    
#     cTime = time.time()
#     fps = 1 / (cTime - pTime)
#     pTime = cTime

#     cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
#     cv2.imshow("Mediapipe Face Mesh", img)
    
#     if cv2.waitKey(5) & 0xFF == 27:
#         break
    
# cap.release()

images = ['happy.jpg']

for idx, file in enumerate(images):
    image = cv2.imread(file)
    results = faceMesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        continue
    annotated_image = image.copy()

    for faceLMs in results.multi_face_landmarks:
        # print('face_landmarks:', faceLMs)
        mpDraw.draw_landmarks(
            image=annotated_image,
            landmark_list=faceLMs,
            connections=mpFaceMesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mpDrawStyles.get_default_face_mesh_tesselation_style()
        )
        # for id, lm in enumerate(faceLMs.landmark):
        #     print(f'{id}: {lm}')
        print(type(faceLMs.landmark))
    
    cv2.imwrite('Annotated' + str(idx) +'.jpg', annotated_image)