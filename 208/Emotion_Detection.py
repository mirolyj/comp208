# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import messagebox
from tkinter import *
import cv2
from PIL import Image,ImageTk
from tensorflow import keras
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
camera = cv2.VideoCapture(0)
window_det = tk.Tk()
window_det.geometry('950x850')
window_det.title('emotion detection')
canvas = tk.Canvas(window_det, height=60, width=450)
image_file = tk.PhotoImage(file='emotion_detection.png')
image = canvas.create_image(0, 0, anchor='nw', image=image_file)
canvas.pack(side='top')
model = keras.models.load_model("./model/model_simple.h5")
model.load_weights("./model/model_weight_1-1.h5")
emotion_labels = ['angry', 'disgust:', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_class = len(emotion_labels)
label1 = Label(window_det,width = 640, height = 480,borderwidth=2, relief="solid")
label1.pack()




def video_loop():
    success, img = camera.read()
    if success:
        cv2.waitKey(50)
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        current_image = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=current_image)
        label1.imgtk = imgtk
        label1.config(image=imgtk)
        window_det.after(1, video_loop)
def predict_emotion(face_img):
    img_size = 48
    face_img = face_img / 255.0
    face_img = cv2.resize(face_img, (img_size, img_size))
    #creat a list to store the img
    img_resize = []
    results = []
    img_resize.append(face_img)  
    img_resize.append(face_img[2:45, :])
    img_resize.append(face_img[3:45, :])
    img_resize.append(face_img[0:44, :])
    for i in range(0,len(img_resize)):
        img_resize.append(cv2.flip(img_resize[i],1))
    # rsz_img.append(cv2.flip(rsz_img[1],1))

    i = 0
    for img in img_resize:
        img_resize[i] = cv2.resize(img, (img_size, img_size))
        #resize it to 1,48,48,1 each image
        img_resize[i] = img_resize[i].reshape(1,img_size,img_size,1)        
        i += 1
    i = 0
    for img in img_resize:
        list_of_list = model.predict_proba(img, batch_size=32, verbose=1)  # predict
        result = [prob for lst in list_of_list for prob in lst]
        results.append(result)
    return results
def usr_get_photo():
    # tk.messagebox.showinfo(title='Welcome', message='How are you? ' + usr_name)
    #label1.destroy()
    ref, imgFace = camera.read()
    cv2.imwrite("./face.png", imgFace, [int(cv2.IMWRITE_PNG_COMPRESSION), 9]) #0-9 more higher the number more wague the photo
    # camera.release()
    # cv2.destroyAllWindows()
    bm1 = PhotoImage(file='face.png')
    label1.configure(image=bm1)
    faces, img_gray, img = face_detect("./face.png")
    if len(faces) == 0:
        tk.messagebox.showinfo(title='Reminder', message='The face was not been detected')
        os.remove("./face.png")
    else:
        for (x, y, w, h) in faces:
            spb = img.shape
            #cut the part of face
            face_img_gray = img_gray[y:y + h, x:x + w]
            #predict the result
            results = predict_emotion(face_img_gray)
            result_sum = np.array([0] * num_class)
            for result in results:
                result_sum = result_sum + np.array(result)
            size = 1
            thick = int(spb[0] * size / 400)
            angry, disgust, fear, happy, sad, surprise, neutral = result_sum
            print('angry:', angry, 'disgust:', disgust, ' fear:', fear, ' happy:', happy, ' sad:', sad,
                  ' surprise:', surprise, ' neutral:', neutral)
            emo_result = np.argmax(result_sum)
            emo = emotion_labels[emo_result]
            emo = str(emo)
            print('Emotion : ', emo)
            sug = ""
            if emo=='angry':
                sug = sug + " Do not make any decisions when you are angry."
            elif emo =='disgust':
                sug = sug + " Try to appreciate people and things."
            elif emo == 'fear':
                sug = sug + " Distract yourself."
            elif emo == 'happy':
                sug = sug + " Keep going~"
            elif emo == 'sad':
                sug = sug + " Accept the changes."
            elif emo == 'surprise':
                sug = sug + " Congradualations! You are good at discovering suprises."
            elif emo == 'neutral':
                sug = sug + (' Still waters run deep.')

            thick_1 = int((w + 10) * size / 200)
            www_s = 0.5
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thick)
            cv2.putText(img, emo+sug, (x + 2, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        www_s, (0, 255, 0), thickness=thick_1, lineType=1)
            cv2.imshow("prediction",img)
            #delete the img when everything is finished
            os.remove("./face.png")
            

    
def face_detect(image_path):
    cascPath = './haarcascade_frontalface_alt.xml'

    faceCasccade = cv2.CascadeClassifier(cascPath)

    # load the img and convert it to bgrgray
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # face detection
    faces = faceCasccade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces, img_gray, img

def usr_exit():
    result = tk.messagebox.askquestion(title='exit program', message='Do you really want to exit the program ?')
    if result == 'yes':
        window_det.destroy()

def usr_help():
    tk.messagebox.showinfo(title='Help', message= 'The program will take a picture and analyse the emotion once you press start button')


btn_get = tk.Button(window_det, text='start', command=usr_get_photo)
btn_get.place(x=255, y=630)
btn_exit = tk.Button(window_det, text='exit', command=usr_exit)
btn_exit.place(x=455, y=630)
btn_help = tk.Button(window_det, text='help', command=usr_help)
btn_help.place(x=655, y=630)

video_loop()
window_det.mainloop()
camera.release()
cv2.destroyAllWindows()
