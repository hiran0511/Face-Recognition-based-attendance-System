import os,cv2;
import numpy as np
from PIL import Image;

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("c:/users/rishi/appdata/local/programs/python/python311/lib/site-packages/cv2/data/haarcascade_frontalface_default.xml");

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    Ids=[]
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces=detector.detectMultiScale(imageNp)
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids

faces,Ids = getImagesAndLabels('C:/Users/rishi/OneDrive/Desktop/Face_Recognition_Attendance_System/Create data')
s = recognizer.train(faces, np.array(Ids))
print("Successfully trained")
recognizer.write('C:/Users/rishi/OneDrive/Desktop/Face_Recognition_Attendance_System/trainer.yml')
