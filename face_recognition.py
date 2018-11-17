import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from Tkinter import *
import os.path

nama = "uknown"
face_dimen = 200
training_count = 20

def retrieve_input(textBox, root):
    inputValue = textBox.get("1.0","end-1c")
    global nama
    nama = inputValue
    root.destroy()

def showTextBox():
    root = Tk()
    root.wm_title("Input your name")
    textBox = Text(root, height=1, width=30)
    textBox.pack()
    buttonOK = Button(root, text="OK", command=lambda: retrieve_input(textBox, root))
    buttonOK.pack()
    mainloop()


face_set = np.empty((0, face_dimen*face_dimen), int)
name_set = np.array([])
name_code = np.array([])
if (os.path.isfile('face_set.npy')):
    face_set = np.load('face_set.npy')

if (os.path.isfile('name_set.npy')):
    name_set = np.load('name_set.npy')

if (os.path.isfile('name_code.npy')):
    name_code = np.load('name_code.npy')

cam = cv2.VideoCapture(0)
facec = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

KNN = KNeighborsClassifier(n_neighbors=10)

isTrain = False
rotation = 0

frame_count = 0
new_face = np.empty((0, face_dimen*face_dimen), int)
new_name = []
last_name_idx = 0

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    if ret:
        row, col, o = frame.shape
        M = cv2.getRotationMatrix2D((col / 2, row / 2), 90 * rotation, 1)
        frame = cv2.warpAffine(frame, M, (col, row))
        frame = cv2.resize(frame, (700, 500))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray, 1.3, 5)

        if(len(name_set) > 0):
            last_name_idx = name_set[len(name_set)-1] + 1

        for (x, y, w, h) in faces:
            face_component = gray[y:y+h, x:x+w]
            face_component = cv2.resize(face_component, (face_dimen, face_dimen))
            if(len(name_set) > 0 and not isTrain):
                KNN.fit(face_set, name_set)
                idx = KNN.predict([face_component.flatten()])
                nama = name_code[int(idx)]
                cv2.putText(frame, nama, (x,y-10), font,1, (0, 0, 255), 2)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 3)
            
            if (isTrain):
                if (frame_count % 10 == 0 and len(new_face) < training_count):
                    new_face = np.concatenate((new_face, [face_component.flatten()]))
                    new_name.append(last_name_idx)
                    print "face added", len(new_face)
                if(len(new_face) == training_count):
                    nface = np.asarray(new_face)
                    nname = np.asarray(new_name)
                    face_set = np.concatenate((face_set, nface))
                    showTextBox()
                    name_set = np.append(name_set, nname)
                    np.save('face_set', face_set)
                    np.save('name_set', name_set)
                    name_code = np.append(name_code, nama)
                    np.save('name_code', name_code)
                    isTrain = False
                    print "Face",nama,"Added"
            frame_count += 1


        key_code = cv2.waitKey(1)
        if (key_code == 114):
            rotation += 1
            rotation = rotation % 4
            print rotation

        if(isTrain):
            percentage = (len(new_face)/float(training_count))*100.0
            cv2.putText(frame, ("Training "+str(percentage)+"%"), (5, 25), font, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Testing", (5, 25), font, 1, (0, 255, 0), 2)

        cv2.imshow("Curva Studios - Face Recognition", frame)
        if(key_code == 116):
            isTrain = True
            new_face = np.empty((0, face_dimen*face_dimen), int)
            new_name = []

        if (key_code == 27):
            break
    else:
        print("Camera not detected")
        break

cam.release()
cv2.destroyAllWindows()
