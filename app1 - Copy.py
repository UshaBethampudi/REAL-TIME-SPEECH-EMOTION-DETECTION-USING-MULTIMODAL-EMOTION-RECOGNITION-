from flask import Flask, render_template, request, Response, redirect, url_for
import numpy as np
import keras.models
import cv2
import time
#from keras.utils import img_to_array
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
#from tensorflow.keras.models import load_model

#from __future__ import division
# import numpy as np
import pandas as pd
from library.AudioEmotionRecognition import *
import time
import re
import os
from collections import Counter
import altair as alt

### Flask imports
import requests
from flask import Flask, render_template, session, request, redirect, flash, Response, url_for

### Audio imports ###
from library.speech_emotion_recognition import *




#model = keras.models.load_model('model.h5')
#model = load_model('model.h5')

port = int(os.environ.get("PORT", 5000))
app = Flask(__name__)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'
app.config['UPLOAD_FOLDER'] = '/Upload'
app.config['TEMPLATES_AUTO_RELOAD'] = True
basedir = os.path.abspath(os.path.dirname(__file__))


from simple_facerec import SimpleFacerec

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("faces/")

# Load Camera
cap = cv2.VideoCapture(0)
pred = "Unknown"
camera = cv2.VideoCapture(0)

NAME = ""
def isVerified(file, name):
    for line in open(file, 'r'):
        if(line[:len(line) - 1] == name):
            return "verified"
    return "unverified"


# Generating txt file with emotion labels
# Return: name of the text file
def emotionDetection(Videofilename):
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    classifier = load_model('model.h5')

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    predictions = []
    cap = cv2.VideoCapture(Videofilename)
    textfile = None
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    while True:
        ret, frame = cap.read()
        if(ret == True):

        # FRAME_COUNT += 1
            time.sleep(1 / fps)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = classifier.predict(roi)[0]
                    label = emotion_labels[prediction.argmax()]
                    label_position = (x, y)
                    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # print(label)
                    predictions.append(label)
                else:
                    cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # cv2.imshow('Emotion Detector', frame)

            if (cv2.waitKey(1) & 0xFF == ord('q')):
                # print(predictions)

                break
        else:
            textfile = open("file.txt", "w")
            for element in predictions:
                textfile.write(element + "\n")
            textfile.close()
            break

    return textfile.name

# Creating Pie chart from o/p of emotiondetection function.
def createPi(textfilename):
    dict_emotions = {'Angry': 0, 'Disgust': 0, 'Fear': 0, 'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Surprise': 0}

    # for emotion in dict_emotions:

    for line in open(textfilename, 'r'):
        dict_emotions[line[:len(line) - 1]] = dict_emotions[line[:len(line) - 1]] + 1

    print(dict_emotions)

    labels = []
    sizes = []

    max = -1
    label_max = "some"

    for x, y in dict_emotions.items():
        if (y > max):
            max = y
            label_max = x
        if (y != 0):
            labels.append(x)
            sizes.append(y)

    # print(label_max)
    # Plot
    plt.pie(sizes, labels=labels)
    plt.axis('equal')
    plt.savefig("static/emotionAnalysis.jpg")
    return label_max


def detect_face():
    global NAME
    while True:
        success, frame = camera.read()
        if (not success):
            break
        else:
            face_locations, face_names = sfr.detect_known_faces(frame)
            for face_loc, name in zip(face_locations, face_names):
                if(name=="Unknown"):
                    return "unverified"
                else:
                    NAME = name
                    return "verified"
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        break


def generate_frames():

    status = "unverified"
    time = 0
    while True:

        ## read the camera frame
        success, frame = camera.read()
        if (not success):
            break
        else:
            face_locations, face_names = sfr.detect_known_faces(frame)
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                # if(name != "Unknown"):
                #     status = "verified"
                if(name=='Unknown'):
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
                else:
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 4)
                # time = time+1

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


## Restful APIs

@app.route('/index1', methods=['GET'])
def index1():
    return render_template('index.html')


# Read the overall dataframe before the user starts to add his own data
df = pd.read_csv('static/js/db/histo.txt', sep=",")



# Audio Index
@app.route('/audio_index', methods=['POST'])
def audio_index():

    # Flash message
    flash("After pressing the button above, you will have 15sec to record your audio")
    
    return render_template('audio.html', display_button=False)

# Audio Recording
# @app.route('/audio_recording', methods=("POST", "GET"))
# def audio_recording():
#
#     # Instanciate new SpeechEmotionRecognition object
#     data = request.form.get('audio_data', "default_name")
#     print("I am running")
#     print(data)
#     SER = speechEmotionRecognition()
#
#     # Voice Recording
#     rec_duration = 16 # in sec
#     rec_sub_dir = os.path.join('tmp','voice_recording.wav')
#     rec_html_dir = os.path.join('static/audios','voice_recording.wav')
#     SER.voice_recording(rec_sub_dir,rec_html_dir, duration=rec_duration)
#
#     # Send Flash message
#     flash("The recording is over! You now have the opportunity to do an analysis of your emotions. If you wish, you can also choose to record yourself again.")
#
#     return render_template('audio.html', display_button=True,audio=rec_html_dir)


# Audio Emotion Analysis
@app.route('/audio_dash', methods=("POST", "GET"))
def audio_dash():
    filenamechanged=""
    if request.method =="POST":
        print(request.files)
        data = request.files['record']
        data.filename = "voice_recording"
        filenamechanged = data.filename+".wav"
        data.save(os.path.join('static/audios',filenamechanged))

        # redirect(url_for("audio_dash"))

        # return redirect(url_for('/audio_dash'))
    print("I am running")
    # Sub dir to speech emotion recognition model
    model_sub_dir = os.path.join('Models', 'MODEL_CNN_LSTM.hdf5')
    model_sub_dir_svm = os.path.join('Models')
    print(model_sub_dir)

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition(model_sub_dir)
    SVMSER = AudioEmotionRecognition(model_sub_dir_svm)

    # Voice Record sub dir

   # rec_sub_dir = os.path.join('static/audios',filenamechanged)
    filepath =  os.path.join('static/audios','voice_recording.wav')
    print(filepath)

    # Predict emotion in voice at each time step
    step = 1 # in sec
    sample_rate = 16000 # in kHz
    emotions, timestamp = SER.predict_emotion_from_file(filepath, chunk_step=step*sample_rate)
    svmemotions,svmtimestamp = SVMSER.predict_emotion_from_file(filepath,chunk_step=step*sample_rate)

    # Export predicted emotions to .txt format
    SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions.txt"), mode='w')
    SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions_other.txt"), mode='a')

    # Get most common emotion during the interview
    major_emotion = max(set(emotions), key=emotions.count)

    # Calculate emotion distribution
    emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]
    # svm_emotion_dist = [int(100 * svmemotions.count(emotion) / len(svmemotions)) for emotion in SVMSER._emotion.values()]

    # Export emotion distribution to .csv format for D3JS
    df = pd.DataFrame(emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df.to_csv(os.path.join('static/js/db','audio_emotions_dist.txt'), sep=',')

    # Get most common emotion of other candidates
    df_other = pd.read_csv(os.path.join("static/js/db", "audio_emotions_other.txt"), sep=",")


    major_emotion_other = df_other.EMOTION.mode()[0]

    # Calculate emotion distribution for other candidates
    emotion_dist_other = [int(100 * len(df_other[df_other.EMOTION==emotion]) / len(df_other)) for emotion in SER._emotion.values()]

    # Export emotion distribution to .csv format for D3JS
    df_other = pd.DataFrame(emotion_dist_other, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df_other.to_csv(os.path.join('static/js/db','audio_emotions_dist_other.txt'), sep=',')
    
    emotion_rec =emotion_dist.sort(reverse=True)

    # Sleep
    time.sleep(0.5)

    emotions_list=[]
    emotion_obj = {'Angry':0, 'Disgust':0, 'Fear':0, 'Happy':0, 'Neutral':0, 'Sad':0, 'Surprise':0}

    for emotion in SER._emotion.values():
        print(len(emotions))
        emotion_obj[emotion] = emotions.count(emotion)
        # print(f'{emotion} {emotions.count(emotion)}')
    print(emotions)
    for emotion_c in emotion_obj.values():
        emotions_list.append(round((emotion_c*100)/len(emotions),2))
    print(emotion_obj)
    print(emotions_list)
    # print(emotion_dist)
    # print(svmemotions)

    return render_template('audio_dash.html', emo=major_emotion, emo_other=major_emotion_other, prob=emotions_list, prob_other=emotion_dist_other,emotion_rec=emotion_rec)








@app.route('/')
def index():
    return render_template('homepage.html')

@app.route('/homepage')
def homepage():
    return render_template('homepage.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/reviews')
def reviews():
    return render_template('reviews.html')

@app.route('/programs')
def programs():
    return render_template('programs.html')

@app.route('/cells')
def cells():
    return render_template('cells.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/care')
def care():
    return render_template('care.html')

@app.route('/stream')
def stream():
    return render_template('stream.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cell9')
def cell9():
    status = isVerified("cell9.txt", NAME)
    if(status == "verified"):
        text_filename = emotionDetection("patientvid.mp4")
        label = createPi(text_filename)
        return render_template('cell9.html', emotion=label)
    else:
        return redirect(url_for('cells'))

@app.route("/getimage")
def get_img():
    return "emotionAnalysis.jpg"


@app.route('/chart')
def chart():
    return render_template('chart.html')


@app.route('/requests',methods=['POST','GET'])
def tasks():
    if request.method == 'POST' or request.method == 'GET':

        if request.form.get('Verify') == 'Verify':

            x=detect_face()
            if (x == "unverified"):
                print("unverified here")
                return render_template('homepage.html')
            elif (x == "verified"):
                print("here")
                with app.app_context():
                    return redirect(url_for('index1'))
    print("here")
    return render_template('stream.html')

if __name__ == "__main__":
    app.run(debug=True)
