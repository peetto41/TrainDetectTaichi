import cv2
import pandas as pd
import mediapipe as mp
import numpy as np
import pickle
import time
import os
from collections import Counter
import matplotlib.pyplot as plt
from tkinter import *

listt = []
TIMER = int(20)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

filename = 'test.mp4'

res = '1080p'

def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

def get_dims(cap, res):
    width, height = STD_DIMENSIONS["1080p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    change_res(cap, width, height)
    return width, height

# VIDEO_TYPE = {
#     'avi': cv2.VideoWriter_fourcc(*'XVID'),
#     'mp4': cv2.VideoWriter_fourcc(*'XVID'),
# }
#
# def get_video_type(filename):
#     filename, ext = os.path.splitext(filename)
#     if ext in VIDEO_TYPE:
#       return  VIDEO_TYPE[ext]
#     return VIDEO_TYPE['mp4']

with open('body_taichi_all2.pkl', 'rb') as f:
    model = pickle.load(f)

    cap = cv2.VideoCapture(0)
    #out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(cap, res))
    get_dims(cap, res)

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make Detections
            results = holistic.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 1. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                      )

            # 2. Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                      )

            # 3. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            # Export coordinates
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array(
                    [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                # Concate rows
                row = pose_row

                # Make Detections
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                print(body_language_class,body_language_prob)
                listt.append(body_language_class)
                # print(list1)
                # Grab ear coords
                coords = tuple(np.multiply(
                    np.array(
                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                         results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                    , [640, 480]).astype(int))

                cv2.rectangle(image,
                              (coords[0], coords[1] + 5),
                              (coords[0] + len(body_language_class) * 20, coords[1] - 30),
                              (245, 117, 16), -1)
                cv2.putText(image, body_language_class, coords,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Get status box
                cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

                # Display Class
                cv2.putText(image, 'NAME'
                            , (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0]
                            , (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display Probability
                cv2.putText(image, 'PROB'
                            , (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2))
                            , (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # out.write(image)
            except:
                pass
            cv2.imshow('taichi', image)
            poseone = listt.count('POSE1')
            posetwo = listt.count('POSE2')
            posethree = listt.count('POSE3')
            posefour = listt.count('POSE4')
            posefive = listt.count('POSE5')
            posesix = listt.count('POSE6')
            poseseven = listt.count('POSE7')
            poseeight = listt.count('POSE8')
            posenine = listt.count('POSE9')
            poseten = listt.count('POSE10')
            one = (poseone / 400) * 100
            two = (posetwo / 675) * 100
            three = (posethree / 200) * 100
            four = (posefour / 635) * 100
            five = (posefive / 140) * 100
            six = (posesix / 960) * 100
            seven = (poseseven / 605) * 100
            eight = (poseeight / 795) * 100
            nine = (posenine / 330) * 100
            ten = (poseten / 645) * 100
            x = [one, two, three, four, five, six, seven, eight, nine, ten]
            # ท่าที่ได้/จำนวนท่าของท่ารำที่ตรวจสอบ * 100
            o = 'จำนวนครั้งท่าที่1', ':', poseone, 'ครั้ง', ',', 'คิดเป็นร้อยละ', ':', one, '%'
            t = 'จำนวนครั้งท่าที่2', ':', posetwo, 'ครั้ง', ',', 'คิดเป็นร้อยละ', ':', two, '%'
            th = 'จำนวนครั้งท่าที่3', ':', posethree, 'ครั้ง', ',', 'คิดเป็นร้อยละ', ':', three, '%'
            fo = 'จำนวนครั้งท่าที่4', ':', posefour, 'ครั้ง', ',', 'คิดเป็นร้อยละ', ':', four, '%'
            fi = 'จำนวนครั้งท่าที่5', ':', posefive, 'ครั้ง', ',', 'คิดเป็นร้อยละ', ':', five, '%'
            si = 'จำนวนครั้งท่าที่6', ':', posesix, 'ครั้ง', ',', 'คิดเป็นร้อยละ', ':', six, '%'
            se = 'จำนวนครั้งท่าที่7', ':', poseseven, 'ครั้ง', ',', 'คิดเป็นร้อยละ', ':', seven, '%'
            e = 'จำนวนครั้งท่าที่8', ':', poseeight, 'ครั้ง', ',', 'คิดเป็นร้อยละ', ':', eight, '%'
            n = 'จำนวนครั้งท่าที่9', ':', posenine, 'ครั้ง', ',', 'คิดเป็นร้อยละ', ':', nine, '%'
            te = 'จำนวนครั้งท่าที่10', ':', poseten, 'ครั้ง', ',', 'คิดเป็นร้อยละ', ':', ten, '%'
            # no = listt.count('POSE1')
            if poseone == 400:
                a = max(x)
                max = 'ผลรวมคะแนนที่มากที่สุด', ':', a, '%'
                ws = Tk(className='โปรแกรมฝึกซ้อมไทเก๊ก')
                label0 = Label(ws, text='', font=(18)).pack()
                label1 = Label(ws, text='ผลรวมคะแนนฝึกซ้อมไทเก๊ก', font=('helvetica', 26)).pack()
                label00 = Label(ws, text='', font=(18)).pack()
                label2 = Label(ws, text=o, font=(18)).pack()
                label3 = Label(ws, text=t, font=(18)).pack()
                label4 = Label(ws, text=th, font=(18)).pack()
                label5 = Label(ws, text=fo, font=(18)).pack()
                label6 = Label(ws, text=fi, font=(18)).pack()
                label7 = Label(ws, text=si, font=(18)).pack()
                label8 = Label(ws, text=se, font=(18)).pack()
                label9 = Label(ws, text=e, font=(18)).pack()
                label10 = Label(ws, text=n, font=(18)).pack()
                label11 = Label(ws, text=te, font=(18)).pack()
                label12 = Label(ws, text=max, font=(18)).pack()
                label13 = Label(ws, text='', font=(18)).pack()
                ws.geometry("500x500")
                ws.eval('tk::PlaceWindow . center')
                exit_button = Button(ws, text="ออกจากหน้าแสดงคะแนน", command=ws.destroy, font=('helvetica', 20),
                                     bg='red')
                exit_button.pack(pady=20)
                ws.mainloop()
                break
            if posetwo == 675:
                a = max(x)
                max = 'ผลรวมคะแนนที่มากที่สุด', ':', a, '%'
                ws = Tk(className='โปรแกรมฝึกซ้อมไทเก๊ก')
                label0 = Label(ws, text='', font=(18)).pack()
                label1 = Label(ws, text='ผลรวมคะแนนฝึกซ้อมไทเก๊ก', font=('helvetica', 26)).pack()
                label00 = Label(ws, text='', font=(18)).pack()
                label2 = Label(ws, text=o, font=(18)).pack()
                label3 = Label(ws, text=t, font=(18)).pack()
                label4 = Label(ws, text=th, font=(18)).pack()
                label5 = Label(ws, text=fo, font=(18)).pack()
                label6 = Label(ws, text=fi, font=(18)).pack()
                label7 = Label(ws, text=si, font=(18)).pack()
                label8 = Label(ws, text=se, font=(18)).pack()
                label9 = Label(ws, text=e, font=(18)).pack()
                label10 = Label(ws, text=n, font=(18)).pack()
                label11 = Label(ws, text=te, font=(18)).pack()
                label12 = Label(ws, text=max, font=(18)).pack()
                label13 = Label(ws, text='', font=(18)).pack()
                ws.geometry("500x500")
                ws.eval('tk::PlaceWindow . center')
                exit_button = Button(ws, text="ออกจากหน้าแสดงคะแนน", command=ws.destroy, font=('helvetica', 20),
                                     bg='red')
                exit_button.pack(pady=20)
                ws.mainloop()
                break
            if posethree == 200:
                a = max(x)
                max = 'ผลรวมคะแนนที่มากที่สุด', ':', a, '%'
                ws = Tk(className='โปรแกรมฝึกซ้อมไทเก๊ก')
                label0 = Label(ws, text='', font=(18)).pack()
                label1 = Label(ws, text='ผลรวมคะแนนฝึกซ้อมไทเก๊ก', font=('helvetica', 26)).pack()
                label00 = Label(ws, text='', font=(18)).pack()
                label2 = Label(ws, text=o, font=(18)).pack()
                label3 = Label(ws, text=t, font=(18)).pack()
                label4 = Label(ws, text=th, font=(18)).pack()
                label5 = Label(ws, text=fo, font=(18)).pack()
                label6 = Label(ws, text=fi, font=(18)).pack()
                label7 = Label(ws, text=si, font=(18)).pack()
                label8 = Label(ws, text=se, font=(18)).pack()
                label9 = Label(ws, text=e, font=(18)).pack()
                label10 = Label(ws, text=n, font=(18)).pack()
                label11 = Label(ws, text=te, font=(18)).pack()
                label12 = Label(ws, text=max, font=(18)).pack()
                label13 = Label(ws, text='', font=(18)).pack()
                ws.geometry("500x500")
                ws.eval('tk::PlaceWindow . center')
                exit_button = Button(ws, text="ออกจากหน้าแสดงคะแนน", command=ws.destroy, font=('helvetica', 20),
                                     bg='red')
                exit_button.pack(pady=20)
                ws.mainloop()
                break
            if posefour == 635:
                a = max(x)
                max = 'ผลรวมคะแนนที่มากที่สุด', ':', a, '%'
                ws = Tk(className='โปรแกรมฝึกซ้อมไทเก๊ก')
                label0 = Label(ws, text='', font=(18)).pack()
                label1 = Label(ws, text='ผลรวมคะแนนฝึกซ้อมไทเก๊ก', font=('helvetica', 26)).pack()
                label00 = Label(ws, text='', font=(18)).pack()
                label2 = Label(ws, text=o, font=(18)).pack()
                label3 = Label(ws, text=t, font=(18)).pack()
                label4 = Label(ws, text=th, font=(18)).pack()
                label5 = Label(ws, text=fo, font=(18)).pack()
                label6 = Label(ws, text=fi, font=(18)).pack()
                label7 = Label(ws, text=si, font=(18)).pack()
                label8 = Label(ws, text=se, font=(18)).pack()
                label9 = Label(ws, text=e, font=(18)).pack()
                label10 = Label(ws, text=n, font=(18)).pack()
                label11 = Label(ws, text=te, font=(18)).pack()
                label12 = Label(ws, text=max, font=(18)).pack()
                label13 = Label(ws, text='', font=(18)).pack()
                ws.geometry("500x500")
                ws.eval('tk::PlaceWindow . center')
                exit_button = Button(ws, text="ออกจากหน้าแสดงคะแนน", command=ws.destroy, font=('helvetica', 20),
                                     bg='red')
                exit_button.pack(pady=20)
                ws.mainloop()
                break
            if posefive == 140:
                a = max(x)
                max = 'ผลรวมคะแนนที่มากที่สุด', ':', a, '%'
                ws = Tk(className='โปรแกรมฝึกซ้อมไทเก๊ก')
                label0 = Label(ws, text='', font=(18)).pack()
                label1 = Label(ws, text='ผลรวมคะแนนฝึกซ้อมไทเก๊ก', font=('helvetica', 26)).pack()
                label00 = Label(ws, text='', font=(18)).pack()
                label2 = Label(ws, text=o, font=(18)).pack()
                label3 = Label(ws, text=t, font=(18)).pack()
                label4 = Label(ws, text=th, font=(18)).pack()
                label5 = Label(ws, text=fo, font=(18)).pack()
                label6 = Label(ws, text=fi, font=(18)).pack()
                label7 = Label(ws, text=si, font=(18)).pack()
                label8 = Label(ws, text=se, font=(18)).pack()
                label9 = Label(ws, text=e, font=(18)).pack()
                label10 = Label(ws, text=n, font=(18)).pack()
                label11 = Label(ws, text=te, font=(18)).pack()
                label12 = Label(ws, text=max, font=(18)).pack()
                label13 = Label(ws, text='', font=(18)).pack()
                ws.geometry("500x500")
                ws.eval('tk::PlaceWindow . center')
                exit_button = Button(ws, text="ออกจากหน้าแสดงคะแนน", command=ws.destroy, font=('helvetica', 20),
                                     bg='red')
                exit_button.pack(pady=20)
                ws.mainloop()
                break
            if posesix == 960:
                a = max(x)
                max = 'ผลรวมคะแนนที่มากที่สุด', ':', a, '%'
                ws = Tk(className='โปรแกรมฝึกซ้อมไทเก๊ก')
                label0 = Label(ws, text='', font=(18)).pack()
                label1 = Label(ws, text='ผลรวมคะแนนฝึกซ้อมไทเก๊ก', font=('helvetica', 26)).pack()
                label00 = Label(ws, text='', font=(18)).pack()
                label2 = Label(ws, text=o, font=(18)).pack()
                label3 = Label(ws, text=t, font=(18)).pack()
                label4 = Label(ws, text=th, font=(18)).pack()
                label5 = Label(ws, text=fo, font=(18)).pack()
                label6 = Label(ws, text=fi, font=(18)).pack()
                label7 = Label(ws, text=si, font=(18)).pack()
                label8 = Label(ws, text=se, font=(18)).pack()
                label9 = Label(ws, text=e, font=(18)).pack()
                label10 = Label(ws, text=n, font=(18)).pack()
                label11 = Label(ws, text=te, font=(18)).pack()
                label12 = Label(ws, text=max, font=(18)).pack()
                label13 = Label(ws, text='', font=(18)).pack()
                ws.geometry("500x500")
                ws.eval('tk::PlaceWindow . center')
                exit_button = Button(ws, text="ออกจากหน้าแสดงคะแนน", command=ws.destroy, font=('helvetica', 20),
                                     bg='red')
                exit_button.pack(pady=20)
                ws.mainloop()
                break
            if poseseven == 605:
                a = max(x)
                max = 'ผลรวมคะแนนที่มากที่สุด', ':', a, '%'
                ws = Tk(className='โปรแกรมฝึกซ้อมไทเก๊ก')
                label0 = Label(ws, text='', font=(18)).pack()
                label1 = Label(ws, text='ผลรวมคะแนนฝึกซ้อมไทเก๊ก', font=('helvetica', 26)).pack()
                label00 = Label(ws, text='', font=(18)).pack()
                label2 = Label(ws, text=o, font=(18)).pack()
                label3 = Label(ws, text=t, font=(18)).pack()
                label4 = Label(ws, text=th, font=(18)).pack()
                label5 = Label(ws, text=fo, font=(18)).pack()
                label6 = Label(ws, text=fi, font=(18)).pack()
                label7 = Label(ws, text=si, font=(18)).pack()
                label8 = Label(ws, text=se, font=(18)).pack()
                label9 = Label(ws, text=e, font=(18)).pack()
                label10 = Label(ws, text=n, font=(18)).pack()
                label11 = Label(ws, text=te, font=(18)).pack()
                label12 = Label(ws, text=max, font=(18)).pack()
                label13 = Label(ws, text='', font=(18)).pack()
                ws.geometry("500x500")
                ws.eval('tk::PlaceWindow . center')
                exit_button = Button(ws, text="ออกจากหน้าแสดงคะแนน", command=ws.destroy, font=('helvetica', 20),
                                     bg='red')
                exit_button.pack(pady=20)
                ws.mainloop()
                break
            if poseeight == 795:
                a = max(x)
                max = 'ผลรวมคะแนนที่มากที่สุด', ':', a, '%'
                ws = Tk(className='โปรแกรมฝึกซ้อมไทเก๊ก')
                label0 = Label(ws, text='', font=(18)).pack()
                label1 = Label(ws, text='ผลรวมคะแนนฝึกซ้อมไทเก๊ก', font=('helvetica', 26)).pack()
                label00 = Label(ws, text='', font=(18)).pack()
                label2 = Label(ws, text=o, font=(18)).pack()
                label3 = Label(ws, text=t, font=(18)).pack()
                label4 = Label(ws, text=th, font=(18)).pack()
                label5 = Label(ws, text=fo, font=(18)).pack()
                label6 = Label(ws, text=fi, font=(18)).pack()
                label7 = Label(ws, text=si, font=(18)).pack()
                label8 = Label(ws, text=se, font=(18)).pack()
                label9 = Label(ws, text=e, font=(18)).pack()
                label10 = Label(ws, text=n, font=(18)).pack()
                label11 = Label(ws, text=te, font=(18)).pack()
                label12 = Label(ws, text=max, font=(18)).pack()
                label13 = Label(ws, text='', font=(18)).pack()
                ws.geometry("500x500")
                ws.eval('tk::PlaceWindow . center')
                exit_button = Button(ws, text="ออกจากหน้าแสดงคะแนน", command=ws.destroy, font=('helvetica', 20),
                                     bg='red')
                exit_button.pack(pady=20)
                ws.mainloop()
                break
            if posenine == 330:
                a = max(x)
                max = 'ผลรวมคะแนนที่มากที่สุด', ':', a, '%'
                ws = Tk(className='โปรแกรมฝึกซ้อมไทเก๊ก')
                label0 = Label(ws, text='', font=(18)).pack()
                label1 = Label(ws, text='ผลรวมคะแนนฝึกซ้อมไทเก๊ก', font=('helvetica', 26)).pack()
                label00 = Label(ws, text='', font=(18)).pack()
                label2 = Label(ws, text=o, font=(18)).pack()
                label3 = Label(ws, text=t, font=(18)).pack()
                label4 = Label(ws, text=th, font=(18)).pack()
                label5 = Label(ws, text=fo, font=(18)).pack()
                label6 = Label(ws, text=fi, font=(18)).pack()
                label7 = Label(ws, text=si, font=(18)).pack()
                label8 = Label(ws, text=se, font=(18)).pack()
                label9 = Label(ws, text=e, font=(18)).pack()
                label10 = Label(ws, text=n, font=(18)).pack()
                label11 = Label(ws, text=te, font=(18)).pack()
                label12 = Label(ws, text=max, font=(18)).pack()
                label13 = Label(ws, text='', font=(18)).pack()
                ws.geometry("500x500")
                ws.eval('tk::PlaceWindow . center')
                exit_button = Button(ws, text="ออกจากหน้าแสดงคะแนน", command=ws.destroy, font=('helvetica', 20),
                                     bg='red')
                exit_button.pack(pady=20)
                ws.mainloop()
                break
            if poseten == 645:
                a = max(x)
                max = 'ผลรวมคะแนนที่มากที่สุด', ':', a, '%'
                ws = Tk(className='โปรแกรมฝึกซ้อมไทเก๊ก')
                label0 = Label(ws, text='', font=(18)).pack()
                label1 = Label(ws, text='ผลรวมคะแนนฝึกซ้อมไทเก๊ก', font=('helvetica', 26)).pack()
                label00 = Label(ws, text='', font=(18)).pack()
                label2 = Label(ws, text=o, font=(18)).pack()
                label3 = Label(ws, text=t, font=(18)).pack()
                label4 = Label(ws, text=th, font=(18)).pack()
                label5 = Label(ws, text=fo, font=(18)).pack()
                label6 = Label(ws, text=fi, font=(18)).pack()
                label7 = Label(ws, text=si, font=(18)).pack()
                label8 = Label(ws, text=se, font=(18)).pack()
                label9 = Label(ws, text=e, font=(18)).pack()
                label10 = Label(ws, text=n, font=(18)).pack()
                label11 = Label(ws, text=te, font=(18)).pack()
                label12 = Label(ws, text=max, font=(18)).pack()
                label13 = Label(ws, text='', font=(18)).pack()
                ws.geometry("500x500")
                ws.eval('tk::PlaceWindow . center')
                exit_button = Button(ws, text="ออกจากหน้าแสดงคะแนน", command=ws.destroy, font=('helvetica', 20),
                                     bg='red')
                exit_button.pack(pady=20)
                ws.mainloop()
                break
            if cv2.waitKey(10) & 0xFF == ord('q'):
                # count all
                #count = Counter(listt)
                # print result
                # print('Result All :', count)
                a = max(x)
                max = 'ผลรวมคะแนนที่มากที่สุด', ':', a, '%'
                ws = Tk(className='โปรแกรมฝึกซ้อมไทเก๊ก')
                # GUI
                label0 = Label(ws, text='', font=(18)).pack()
                label1 = Label(ws, text='ผลรวมคะแนนฝึกซ้อมไทเก๊ก', font=('helvetica', 26)).pack()
                label00 = Label(ws, text='', font=(18)).pack()
                label2 = Label(ws, text=o, font=(18)).pack()
                label3 = Label(ws, text=t, font=(18)).pack()
                label4 = Label(ws, text=th, font=(18)).pack()
                label5 = Label(ws, text=fo, font=(18)).pack()
                label6 = Label(ws, text=fi, font=(18)).pack()
                label7 = Label(ws, text=si, font=(18)).pack()
                label8 = Label(ws, text=se, font=(18)).pack()
                label9 = Label(ws, text=e, font=(18)).pack()
                label10 = Label(ws, text=n, font=(18)).pack()
                label11 = Label(ws, text=te, font=(18)).pack()
                label12 = Label(ws, text=max, font=(18)).pack()
                label13 = Label(ws, text='', font=(18)).pack()
                ws.geometry("500x500")
                ws.eval('tk::PlaceWindow . center')
                exit_button = Button(ws, text="ออกจากหน้าแสดงคะแนน", command=ws.destroy, font=('helvetica', 20), bg='red')
                exit_button.pack(pady=20)
                ws.mainloop()
                break
    cap.release()
    # out.release()
    cv2.destroyAllWindows()