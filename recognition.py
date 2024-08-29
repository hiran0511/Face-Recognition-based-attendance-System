import cv2
import numpy as np;
import xlwrite
#import firebase_admin
#import firebase as Firebase;
#import firebase.firebase_ini as fire;
#import firebase
import time
#import sys
from playsound import playsound

start = time.time()
period = 8
face_cas = cv2.CascadeClassifier('c:/users/rishi/appdata/local/programs/python/python311/lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0);
recognizer = cv2.face.LBPHFaceRecognizer_create();
recognizer.read('C:/Users/rishi/OneDrive/Desktop/Face_Recognition_Attendance_System/trainer.yml');
flag = 0;
id = 0;
filename = 'filename';
dict = {
    'item1': 1
        }
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, img = cap.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    faces = face_cas.detectMultiScale(gray, 1.3, 7);
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2);
        id, conf = recognizer.predict(roi_gray)
        if (conf < 50):
            if (id == 1241 or id==2141):
                id = 'Hiranmayee'
                if ((str(id)) not in dict):
                    filename = xlwrite.output('attendance', 'class1', 1, id, 'yes');
                    dict[str(id)] = str(id);
            elif (id == 2155):
                id = 'Shiva'
                if ((str(id)) not in dict):
                    filename = xlwrite.output('attendance', 'class1', 2, id, 'yes');
                    dict[str(id)] = str(id);

            elif (id == 101):
                id = 'Hiran'
                if ((str(id)) not in dict):
                    filename = xlwrite.output('attendance', 'class1', 3, id, 'yes');
                    dict[str(id)] = str(id);
           # elif (id == 3):
            #    id = 'rishitha'
             #   if ((str(id)) not in dict):
              #      filename = xlwrite.output('attendance', 'class1', 4, id, 'yes');
               #     dict[str(id)] = str(id);
           # elif (id == 2):
            #    id = 'bhagyalaxmi'
             #   if ((str(id)) not in dict):
              #      filename = xlwrite.output('attendance', 'class1', 7, id, 'yes');
               #     dict[str(id)] = str(id);
                    
           # elif (id == 1253):
            #    id = 'Pranav'
             #   if ((str(id)) not in dict):
              #      filename = xlwrite.output('attendance', 'class1', 5, id, 'yes');
               #     dict[str(id)] = str(id);

            elif (id == 1255):
                id = 'Sai_Kumar'
                if ((str(id)) not in dict):
                    filename = xlwrite.output('attendance', 'class1', 6, id, 'yes');
                    dict[str(id)] = str(id);

        else:
            id = 'Unknown, can not recognize'
            flag = flag + 1
            break

        cv2.putText(img, str(id) + " " + str(conf), (x, y - 10), font, 0.55, (120, 255, 120), 1)
    cv2.imshow('frame', img);
    if flag == 10:
        playsound('transactionSound.mp3')
        print("Transaction Blocked")
        break;
    if time.time() > start + period:
        break;
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break;

cap.release();
cv2.destroyAllWindows();
