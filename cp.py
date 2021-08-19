import random
import mediapipe as mp
from decord import VideoReader
from decord import cpu
from camera import results
import cv2
import csv
import numpy as np
import os
import sys

ROOT_DIRECTORY_PATH = sys.path[0]

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

# #num_coords = len(results.pose_landmarks.landmark)+len(results.right_hand_landmarks.landmark)+len(results.left_hand_landmarks.landmark)
# num_coords = len(results.pose_landmarks.landmark)
#
# landmarks = ['class']
# for val in range(1, num_coords+1):
#     landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
#
# landmarks
#
# with open('coords.csv', mode='w', newline='') as f:
#     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     csv_writer.writerow(landmarks)

def getFrame(src):
    vr = VideoReader(src, ctx=cpu(0))
    print(len(vr))
    # print(type(len(vr)))
    fend = int(len(vr) / 1.05)
    # print(fend)
    startrand = random.randint(0,fend)
    # endrand = random.randint(startrand,len(vr))

    cap = cv2.VideoCapture(src)

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # for frames in My_list:
                print('start :',startrand)
                cap.set(cv2.CAP_PROP_POS_FRAMES, startrand)
                startrand += 1
                ret, frame = cap.read()


                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make Detections
                results = holistic.process(image)
                # print(results.face_landmarks)

                # Recolor image back to BGR for rendering
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # 1. Draw face landmarks
                # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                #                           mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                #                           mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                #                           )

                # 2. Right hand
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                          )

                # 3. Left Hand
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                          )

                # 4. Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )
                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(
                        np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    # Concate rows
                    row = pose_row

                    # Append class name
                    row.insert(0, class_name)

                    # Export to CSV
                    with open('coords3.csv', mode='a', newline='') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(row)

                except:
                    pass

                cv2.imshow('Taichi', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                if startrand == len(vr):
                    break
                cap.release()

for mp4name in os.listdir('c/'):
    for i in range(1000):
        getFrame(ROOT_DIRECTORY_PATH + '/c/' + mp4name)
        class_name = mp4name.split('.mp4')[-2]
#     cv2.destroyAllWindows()


