
# coding: utf-8

# In[ ]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense
 
#creating the model
def create_model():
    model = Sequential()
    model.add(Convolution2D(16, 5, 5, activation='relu', input_shape=(28,28, 3)))
    model.add(MaxPooling2D(2, 2))
 
    model.add(Convolution2D(32, 5, 5, activation='relu'))
    model.add(MaxPooling2D(2, 2))
 
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
 
    model.add(Dense(11, activation='softmax'))
    return model
 
#loading model
model = create_model()
model.load_weights('model_mnist3.h5')
 
 
import operator
cap = cv2.VideoCapture(1)
st = ""
no_of_times = 0
blanck=np.zeros((512,512,3),np.uint8)
while(True):
    #reading frames from camera
    blanck=np.zeros((300,300,3),np.uint8)
    ret, frame1 = cap.read()
    frame2 = frame1[150:500 , 160:950]
    frame = frame2.copy()
    frame_new= frame2.copy()
    x1 = 160
    y1 = 150
    x2 = 950
    y2 = 500
     
    ret, img = cv2.threshold(frame, 120, 255, cv2.THRESH_BINARY_INV)
    cvt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img,contours, hierarchy = cv2.findContours(cvt,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
     
    thisdict = {}
    mylist = []
    for c in contours:
        (x, y, w, h)= cv2.boundingRect(c)
        if (w>10) or (h>10):
            mylist.append((x,y,w,h))
             
    first_height = 0
    #dictionary to hold squares
    squares={}
 
    #dictionary for holding normal numbers
    numbers={}
    height_set=False
     
     
    for i in range(0, len(mylist)):
        x = mylist[i][0]
        y = mylist[i][1]
        w = mylist[i][2]
        h = mylist[i][3]
        if h/w>3:
            x=x-10
            w=w+20
        if w/h>3:
            y=y-60
            h=h+110
        y=y-27
        x=x-25
        w=w+50
        h=h+54
        #variable for comparing heights to sort squares
        if height_set is False:
            first_height=h
            height_set = True
        #if small number is detected square variable will become true
        square = False
         
        if h < (first_height-30):
            square = True
         
             
        
        cv2.rectangle(frame1,(x+x1,y+y1),(x+w+x1,y+h+y1), (0,0, 255), 2)
        img1 = frame_new[y:y+h, x:x+w]
        ret, gray = cv2.threshold(img1,120,255,cv2.THRESH_BINARY )
         
        try:
            im = cv2.resize(gray, (40,40))
            im1 = cv2.resize(im, (28,28))
            next2 = cv2.resize(im1, (28,28))
            ret, gray1 = cv2.threshold(next2,254,255,cv2.THRESH_BINARY)
 
            ar = np.array(gray1).reshape((28,28,3))
            ar = np.expand_dims(ar, axis=0)
            prediction = model.predict(ar)[0]
 
            #predicrion of class labels
            for i in range(0,12):
                if prediction[i]==1.0:
##                    if i==0:
##                        j= ","
                    if i==1:
                        j= "0"
                    if i==2:
                        j= "1"
                    if i==3:
                        j= "2"
                    if i==4:
                        j= "3"
                    if i==5:
                        j= "4"
                    if i==6:
                        j= "5"
                    if i==7:
                        j= "6"
                    if i==8:
                        j= "7"
                    if i==9:
                        j= "8"
                    if i==10:
                        j= "9"
                      
            #printing prediction
                    cv2.putText(frame1, j, (x+x1,y+y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
                    thisdict[x]= str(j)
                    if square is True:
                        squares[x]=str(j)
                    else:
                        numbers[x]=str(j)
 
        except:
            pass
 
        sort = sorted(thisdict.items(), key=operator.itemgetter(0))
        s = ""
        square_lst=""
        number_lst=""
    for x in range(0,len(sort)):
        s=s+str(sort[x][1])
        cv2.putText(frame1, s, (100,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(blanck, 'Detected: ', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1,cv2.LINE_AA)
 
 
    present1 = False; present2 = False;
     
    try:
        numbers_sort = sorted(numbers.items(), key=operator.itemgetter(0))
 
        square_sort = sorted(squares.items(), key=operator.itemgetter(0))
         
         
         
        for x in range(0,len(numbers)):
            number_lst = number_lst+str(numbers_sort[x][1])
 
        for x in range(0,len(squares)):
            square_lst = square_lst+str(square_sort[x][1])
            present2 = True
 
        "".join(square_lst); "".join(number_lst);
         
    except: pass
 
    print('number: {}   power: {}'.format(number_lst, square_lst ))
     
     
    if len(number_lst)>0 and len(square_lst)>0 and len(square_lst)< 3:
             
            square_lst=int(square_lst); number_lst=int(number_lst);
 
            solution = number_lst**square_lst
 
            cv2.putText(blanck, str(number_lst)+'^'+str(square_lst),(60,100),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0), 1, cv2.LINE_AA)
             
            cv2.putText(blanck, ' = '+str(solution),(90,150),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0), 1, cv2.LINE_AA)
 
     
     
    cv2.imshow('main frame', frame1)
    cv2.imshow('binary',cvt)
    cv2.imshow('calculation',blanck)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break       
cap.release()
cv2.destroyAllWindows()

