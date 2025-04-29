import cv2
import os
import pyttsx3


engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

haar_file = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
datasets = 'datasets'
person_name = input("Enter your name: ")  

path = os.path.join(datasets, person_name)
if not os.path.exists(path):
    os.makedirs(path)

(width, height) = (130, 100)
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

count = 1
while count <= 100:
    print(f"Capturing image {count}...")
    _, img = webcam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite(f'{path}/{count}.png', face_resize)

    count += 1
    cv2.imshow('Face Capture', img)
    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()