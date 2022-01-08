#!/usr/bin/env python
# coding: utf-8

# In[1]:


#process.py 1
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import math
import imutils
import time
import dlib
import cv2
from cv2 import VideoWriter_fourcc, VideoWriter
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.models import load_model


def eye_brow_distance(leye,reye):
    global points
    distq = dist.euclidean(leye,reye)
    points.append(int(distq))
    return distq
    #region of interest..periocular region
def emotion_finder(faces,frame):
    global emotion_classifier
    EMOTIONS = ["angry" ,"awful","fear", "happy", "sad", "surprise","satisfied"]
    x,y,w,h = face_utils.rect_to_bb(faces)
    frame = frame[y:y+h,x:x+w]
    roi = cv2.resize(frame,(64,64))
    roi = roi.astype("float") / 255.0
    # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
    # the ROI for classification via the CNN
    roi = img_to_array(roi)
    roi = np.expand_dims(roi,axis=0)
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    if label in ['fear','sad', 'neutral']:
        label = 'stressed'
    else:
        label = 'not stressed'
    return label
    
def normalize_values(points,disp):
    #max-min normalization
    #eyebrow's 
    normalized_value = abs(disp - np.min(points))/abs(np.max(points) - np.min(points))
    stress_value = np.exp(-(normalized_value))
    return stress_value



detector = dlib.get_frontal_face_detector()# front face feautres'extraction
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotion_classifier = load_model("_mini_XCEPTION.102-0.66.hdf5", compile=False)
print(emotion_classifier, flush = True)
#video input
cap = cv2.VideoCapture('IMG_4291.mp4')
# cap = cv2.VideoCapture(0)
# fps=30 #Frames per second
# size1=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# videoWriter=cv2.VideoWriter('MyVideo123.avi',cv2.VideoWriter_fourcc('I','4','2','0'),fps,size1)
# success,frame =cap.read()
# numFramesRemaining=10*fps -1
# while success and numFramesRemaining>0:
#        videoWriter.write(frame) 
#        success,frame= cap.read() 
#        numFramesRemaining-=1
points = []
stress_list = []
stressval_list = []
stressgraph = []
size=0
while(True):
    _,frame = cap.read()
    if(not _): break
    frame = cv2.flip(frame,1)
    frame = imutils.resize(frame, width=500,height=500)
    
    #resize frame
    (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
    (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]

    #preprocessing the image
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    detections = detector(gray,0)
    for detection in detections:
        #emotion finder
        emotion = emotion_finder(detection,gray)
        cv2.putText(frame, emotion, (10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (256, 0, 256), 0)
        shape = predictor(frame,detection)
        shape = face_utils.shape_to_np(shape)
        
        leyebrow = shape[lBegin:lEnd]
        reyebrow = shape[rBegin:rEnd]
            
        reyebrowhull = cv2.convexHull(reyebrow)
        leyebrowhull = cv2.convexHull(leyebrow)

        cv2.drawContours(frame, [reyebrowhull], -1, (256, 0, 256), 1)
        cv2.drawContours(frame, [leyebrowhull], -1, (256, 0, 256), 1)

        distq = eye_brow_distance(leyebrow[-1],reyebrow[0])
        stress_value = normalize_values(points,distq)
        print(stress_value)
        #if stress_value!=1.0: stress_list.append(stress_list)
        if math.isnan(stress_value):
            continue
        cv2.putText(frame,"stress level:{}".format(str(int(stress_value*100))),(20,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (256, 0, 256), 0)
        stress_list.append(frame)

    height, width, layers = frame.shape
    size = (width,height)
    stressval_list.append(stress_value)
out = cv2.VideoWriter('resvid.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
cap.release()
print("END REACHED")
for i in range(len(stress_list)):
    out.write(stress_list[i])


# In[6]:


#train.py 2
from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os

num_classes = 7
img_rows,img_cols = 48,48
batch_size = 32

train_data_dir = "train"
validation_data_dir = "test"

train_datagen = ImageDataGenerator(
					rescale=1./255,
					rotation_range=30,
					shear_range=0.3,
					zoom_range=0.3,
					width_shift_range=0.4,
					height_shift_range=0.4,
					horizontal_flip=True,
					fill_mode='nearest')
# this is the augmentation configuration we will use for training

validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode='grayscale',
					target_size=(img_rows,img_cols),
					batch_size=batch_size,
					class_mode='categorical',
					shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(img_rows,img_cols),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)


model = Sequential()

# LAYER 1

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# LAYER 2

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# LAYER 3

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


# LAYER 4

model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# LAYER 5

model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# LAYER 6

model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())

from tensorflow.keras.optimizers import RMSprop,SGD,Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint('Emotion_little_vgg.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)
#it uses to save the model

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])

nb_train_samples = 24176
nb_validation_samples = 3006
epochs=25

history=model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples//batch_size)

model.save('trained_model.hdf5')


# In[9]:


plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()


# In[35]:


# test.py(3)
from keras.preprocessing.image import img_to_array
import cv2
from keras.models import load_model
import numpy as np
# loading files
haar_file="haarcascade_frontalface_default.xml"
# haar_file="lbpcascade_frontalcatface.xml"
# an Object Detection Algorithm used to identify faces in an image or a real time video. 
emotion_model='_mini_XCEPTION.102-0.66.hdf5'

cascade=cv2.CascadeClassifier(haar_file)
emotion_classifier=load_model(emotion_model,compile=True)
emotion_names=["angry","awful","fear", "happy", "sad", "surprise","satisfied"]
# frame=cv2.imread('12.JPG')
frame=cv2.imread('22.JPG')
# frame=cv2.imread('sadm.JPG')
# frame=cv2.imread('surprise1.JPG')
# frame=cv2.imread('dis1.JPG')
# frame=cv2.imread('resize-1638372400424976794I.jpg')
gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
faces=cascade.detectMultiScale(gray_frame,1.5,5)
text=[]
for (x,y,w,h) in faces:
    roi=gray_frame[y:y+h,x:x+w]
    roi=cv2.resize(roi,(64,64))
    roi=roi.astype("float")/255.0
    roi=img_to_array(roi)
    roi=np.expand_dims(roi,axis=0)
    
    predicted_emotion=emotion_classifier.predict(roi)[0]
    probab=np.max(predicted_emotion)
    label=emotion_names[predicted_emotion.argmax()]
    percen=predicted_emotion*100
for j in range(7):
        text.append(emotion_names[j]+" : "+str(percen[j]))
for i in range(7):    
#          cv2.putText(frame,text[i],(5,i*30+15),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),0)
        print(text[i])
cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),0)
cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),0)
cv2.imwrite('res1.jpg', frame)


# In[2]:


from playsound import playsound

labels = ["happy", "angry", "awful", "disgust", "sad", "surprised", "satisfied"]
# label = "fear"..dictionary
tips = {"fear":["Drink water","Get a good night's sleep","Eat wholesome meals","Go for a walk","Turn off news feed/social media","Talk to someone"],
        "angry":["Repeat gentle phrases to yourself","Take a walk","Use visualization to calm down","Focus on your breathing","Phone a friend","Watch a stand up comedy"],
        "sad":["Do things you enjoy (or used to)","Get quality exercise","Eat a nutritious diet","Challenge negative thinking"]
       }
website_links = {"fear":["https://www.businessinsider.in/science/health/heres-how-to-take-care-of-yourself-if-youre-feeling-scared-or-sad-right-now/articleshow/55342883.cms","https://mhanational.org/what-can-i-do-when-im-afraid"],
                 "angry":["https://www.thehotline.org/resources/how-to-cool-off-when-youre-angry/","https://www.mayoclinic.org/healthy-lifestyle/adult-health/in-depth/anger-management/art-20045434"],
                 "sad":["https://www.vandrevalafoundation.com/","https://www.healthline.com/health/depression/recognizing-symptoms#fatigue"]
                }
youtube_links = {"fear":["https://www.youtube.com/watch?v=IAODG6KaNBc"],
                 "angry":["https://www.youtube.com/watch?v=P6aPg3YBvBQ"],
                 "sad":["https://www.youtube.com/watch?v=P6aPg3YBvBQ"]
                }
song_links = {"fear":["https://www.youtube.com/watch?v=GyA8ccqwp-4&feature=youtu.be","https://www.bing.com/videos/search?q=alone+part+2&docid=607990227673701963&mid=1B6860319511BF2C5CC21B6860319511BF2C5CC2&view=detail&FORM=VIRE"],
              "angry":["https://www.youtube.com/watch?v=e74wLJ_KRes&feature=youtu.be","https://www.youtube.com/watch?v=JwWz-94a_Hk&feature=youtu.be"],
              "sad":["https://www.youtube.com/watch?v=25ROFXjoaAU&feature=youtu.be","https://www.youtube.com/watch?v=BzE1mX4Px0I"],
              "happy":["https://www.youtube.com/watch?v=vGZhMIXH62M","https://www.youtube.com/watch?v=Pkh8UtuejGw"]
             }
tunes = {"fear":'fear.mp3',
         "angry":'angry.mp3',
         "sad":'sad.mp3'
         }

if (label == "happy"):
    # songs
    print("Here are some song suggestions for your mood:")
    for s in song_links.get('happy'):
        print(s)

elif (label == "angry"):
    # songs
    print("Here are some song suggestions for your mood:")
    for s in song_links.get('angry'):
        print(s)
    # tips
    print("Here are some tips to help you feel better:")
    for i in tips.get('angry'):
        print("-> "+i)
    # resources
    print("Here are some resources that you may find beneficial:")
    for j in website_links.get('angry'):
        print(j)
    for k in youtube_links.get('angry'):
        print(k)
    # tunes
    print("Here's a tune that will help you calm down.")
    playsound(tunes.get('angry'))

elif (label == "fear"):
    # songs
    print("Here are some song suggestions for your mood:")
    for s in song_links.get('fear'):
        print(s)
    # tips
    print("Here are some tips to help you feel better:")
    for i in tips.get('fear'):
        print("-> "+i)
    # resources
    print("Here are some resources that you may find beneficial:")
    for j in website_links.get('fear'):
        print(j)
    for k in youtube_links.get('fear'):
        print(k)
    # tunes
    print("Here's a tune that will make you feel better.")
    playsound(tunes.get('fear'))

elif (label == "sad"):
    # songs
    print("Here are some song suggestions for your mood:")
    for s in song_links.get('sad'):
        print(s)
    # tips
    print("Here are some tips to help you feel better:")
    for i in tips.get('sad'):
        print("-> "+i)
    # resources
    print("Here are some resources that you may find beneficial:")
    for j in website_links.get('sad'):
        print(j)
    for k in youtube_links.get('sad'):
        print(k)
    # tunes
    print("Listen to a tune that will soothe you.")
    playsound(tunes.get('sad'))


# In[ ]:




