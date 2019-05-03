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
    face_img = face_img * (1. / 255)
    img_resized = cv2.resize(face_img, (img_size, img_size))
    rsz_img = []
    rsh_img = []
    results = []
    # print (len(resized_img[0]),type(resized_img))
    rsz_img.append(img_resized[:, :])  # resized_img[1:46,1:46]
    rsz_img.append(img_resized[2:45, :])
    rsz_img.append(cv2.flip(rsz_img[0], 1))
    # rsz_img.append(cv2.flip(rsz_img[1],1))

    '''rsz_img.append(resized_img[0:45,0:45])
    rsz_img.append(resized_img[2:47,0:45])
    rsz_img.append(resized_img[2:47,2:47])
    rsz_img.append(cv2.flip(rsz_img[2],1))
    rsz_img.append(cv2.flip(rsz_img[3],1))
    rsz_img.append(cv2.flip(rsz_img[4],1))'''
    i = 0
    for rsz_image in rsz_img:
        rsz_img[i] = cv2.resize(rsz_image, (img_size, img_size))
        # =========================
        # cv2.imshow('%d'%i,rsz_img[i])
        i += 1
    # why 4 parameters here, what's it means?
    for rsz_image in rsz_img:
        rsh_img.append(rsz_image.reshape(1, img_size, img_size, 1))
    i = 0
    for rsh_image in rsh_img:
        list_of_list = model.predict_proba(rsh_image, batch_size=32, verbose=1)  # predict
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
    image_result = cv2.imread('./face.png',0)
    #image_show = cv2.imshow('image1', image_result)
    faces, img_gray, img = face_detect("./face.png")
    #spb = img.shape
    #sp = img_gray.shape
    #height = sp[0]
    #width = sp[1]
    #ize = 600
    #image_show1 = cv2.imshow('image2', faces)
    if len(faces) == 0:
        tk.messagebox.showinfo(title='Reminder', message='The face was not been detected')
        os.remove("./face.png")
    else:
        for (x, y, w, h) in faces:
            spb = img.shape
            sp = img_gray.shape
            height = sp[0]
            width = sp[1]
            size = 600
            #cut the part of face
            face_img_gray = img_gray[y:y + h, x:x + w]
            #predict the result
            results = predict_emotion(face_img_gray)
            result_sum = np.array([0] * num_class)
            for result in results:
                result_sum = result_sum + np.array(result)
                print(result)
            t_size = 2
            ww = int(spb[0] * t_size / 400)
            angry, disgust, fear, happy, sad, surprise, neutral = result_sum
            print('angry:', angry, 'disgust:', disgust, ' fear:', fear, ' happy:', happy, ' sad:', sad,
                  ' surprise:', surprise, ' neutral:', neutral)
            label = np.argmax(result_sum)
            emo = emotion_labels[label]
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

            www = int((w + 10) * t_size / 200)
            #www_s = int((w + 20) * t_size / 100) * 2 / 5
            www_s = 0.5
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), ww)
            cv2.putText(img, emo+sug, (x + 2, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        www_s, (0, 255, 0), thickness=www, lineType=1)
            cv2.imshow("prediction",img)
            os.remove("./face.png")

    
def face_detect(image_path):
    cascPath = './haarcascade_frontalface_alt.xml'

    faceCasccade = cv2.CascadeClassifier(cascPath)

    # load the img and convert it to bgrgray
    # img_path=image_path
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # face detection
    faces = faceCasccade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # print('img_gray:',type(img_gray))
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
