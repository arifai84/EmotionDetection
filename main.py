# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:14:06 2023

@author: ahmad.rifai
"""

import cv2 as cv
import datetime
from datetime import datetime as dt2
import pandas as pd
from vosk import Model, KaldiRecognizer
from pycsp.parallel import *
import pyaudio
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deepface import DeepFace
feelings = ['angry','disgust','fear','happy','sad','surprise','neutral','speech neg','speech neu','speech pos', 'speech compound']
df_face_and_sound = pd.DataFrame(feelings, columns=['Labels'])
df_face_and_sound.set_index("Labels", inplace=True)
dt = datetime.datetime.now()
def capture_date(q):
    while True:
        q.put(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
negative = 0
neutral = 0
positive = 0
compound = 0
listxx = []
@process
def listen_speech():
    global df_face_and_sound
    global listxx
    global dt, negative, positive, neutral, compound
    
    model = Model(r"C:/Users/ahmad.rifai/test/open_model_zoo/vosk-model-small-en-us-0.15")
    recognizer = KaldiRecognizer(model, 16000)
    mic = pyaudio.PyAudio()
    stream = mic.open(format=pyaudio.paInt16, channels=1,
                      rate=16000, input=True, frames_per_buffer=8192)
    stream.start_stream()
    print("Start Stream")
    while True:
        print(df_face_and_sound)
        print("Identify DF")
        data = stream.read(8192)
        print("Stream Read")
        print(data)
        if recognizer.AcceptWaveform(data):
            print("recognizer")
            result = recognizer.Result()
            print("Resultq")
            sid_obj = SentimentIntensityAnalyzer()
            sentiment_dict = sid_obj.polarity_scores(result)
            negative = sentiment_dict['neg']
            neutral = sentiment_dict['neu']
            positive = sentiment_dict['pos']
            compound = sentiment_dict['compound']
        else: print("Whateber")
        df_face_and_sound.at['speech neg',dt] = negative
        df_face_and_sound.at['speech neu',dt] = neutral
        df_face_and_sound.at['speech pos',dt] = positive
        df_face_and_sound.at['speech compound',dt] = compound
        if cv.waitKey(1) & 0xFF == ord('w'):
             stream.stop_stream()
             stream.close()
             mic.terminate()
        break
@process
def open_camera():
  global  df_face_and_sound
  global dt
  face_cascade_name = cv.data.haarcascades + 'haarcascade_frontalface_alt.xml'  #getting a haarcascade xml file
  face_cascade = cv.CascadeClassifier()  #processing it for our project
  if not face_cascade.load(cv.samples.findFile(face_cascade_name)):  #adding a fallback event
   print("Error loading xml file")
#video
  cap = cv.VideoCapture(0)
#  faceNotPresentDuration = 0
  while(True):
    # Capture frame-by-frame
    _,frame = cap.read()
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    #changing the video to grayscale to make the face analisis work properly
    face=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    for x,y,w,h in face:
          img=cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)  #making a recentangle to show up and detect the face and setting it position and colour
    try:
          dt = datetime.datetime.now()
          emo = DeepFace.analyze(img,actions=['emotion'])  #same thing is happing here as the previous example, we are using the analyze class from deepface and using ‘frame’ as input
          emotion = emo['emotion']
          df_face_and_sound.at['angry',dt] = emotion['angry']
          df_face_and_sound.at['disgust',dt] = emotion['disgust']
          df_face_and_sound.at['fear',dt] = emotion['fear']
          df_face_and_sound.at['happy',dt] = emotion['happy']
          df_face_and_sound.at['sad',dt] = emotion['sad']
          df_face_and_sound.at['surprise',dt] = emotion['surprise']
          df_face_and_sound.at['neutral',dt] = emotion['neutral']
          print(df_face_and_sound)
          #converting normal format of emotions to a DataFrame for manipulations
    except:
          print("no face")
    cv.imshow('cap',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):

        break
  cap.release()
  cv.destroyAllWindows()
if __name__ == "__main__":
    Parallel(open_camera(),listen_speech())
    print('Finishing')
    df_face_and_sound.to_csv("C:/Users/ahmad.rifai/test/dataframe_dump.csv")
