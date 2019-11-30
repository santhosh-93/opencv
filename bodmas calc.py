
# coding: utf-8

# In[ ]:


import math
from sympy import *
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

    model.add(Dense(18, activation='softmax'))
    return model

#loading model
model = create_model()
model.load_weights('model_mnist5.h5')

def add_doublestar(list_n):
    for i in range(0, len(list_n)):
        if list_n[i] == '^':
            list_n.insert(i, '*')
            list_n.insert(i+1, '*')
            break
    list_n.remove('^')
    return list_n

def add_star(s, frame1):
    list_n = [0]*len(s)
    for i in range(0, len(list_n)):
        list_n[i] = s[i]
    for i in range(0, len(list_n)+1):
        if list_n[i] == 'x' and list_n[i-1] != '*':
            list_n.insert(i, '*')
    return list_n

def convert(list1): 
    res = ""
    for i in range (0, len(list1)):
        res += str(list1[i])
    return(res) 

def print_fun(xi,expr, frame1):
    ans = []
    tempy = []
    if expr:
        xa = symbols('x')
        if expr:
            if 'x' in expr[1]:
                if '-' in expr[0]:
                    for i in range(len(xi)):
                        expra = sympify(expand(0)-(expand(expr[1])))
                        tempy.append(int(expra.subs(x, xi[i])))
                        temp = (xi[i],int(expra.subs(x, xi[i])))
                        ans.append(temp)
                else:
                    for i in range(len(xi)):
                        expra = sympify(expr[1])
                        tempy.append(int(expra.subs(xa, xi[i])))
                        temp = (xi[i],int(expra.subs(xa, xi[i])))
                        ans.append(temp)
    m = 150
    srt = expr[1]
    cv2.putText(frame1, "Points = ", (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    for i in range(0,len(xi)):
        m += 40
        cv2.putText(frame1, "x=" + str(xi[i]) + " => " +  " y = " + str(srt.replace("**2", "^2").replace("x", str("("+str((xi[i]))+")")).replace("*","x")) + " => y = " + str(tempy[i]) , (70, m), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    return(tempy, ans)

def plot_graph1(xi, tempy, ans, frame1):  

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = np.linspace(-10,10,200)

    
    ax.grid(True, which='both')
    ax.spines['left'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    for i in range(len(xi)):
        plt.text(int(xi[i]), int(tempy[i]), str(ans[i]), fontsize=8)
    ax.scatter(*zip(*ans))
    plt.savefig('/Users/santhosh/admatic/graph1.jpg')
    y_offset = 410
    x_offset = 80
    img = cv2.imread('/Users/santhosh/admatic/graph1.jpg')
    frame1[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
    return frame1

def plot_graph2(xi, tempy, ans, frame1, expr): 
    x = symbols('x')
    y = lambdify(x, expand(expr[1]), 'numpy')
    fig = plt.figure()
    x = np.linspace(-10,10,200)
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True, which='both')
    ax.spines['left'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.plot(x,y(x),'-r',label="y=2x")
    plt.savefig('/Users/santhosh/admatic/graph2.jpg')
    y_offset = 410
    x_offset = 700
    img = cv2.imread('/Users/santhosh/admatic/graph2.jpg')
    frame1[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
    return frame1
def output(exp):
    split_by_plus=exp.split('+')
    split=split_by_plus[-1]
    x  = symbols('x')

    res=solve(exp,x)
    print('res',res)
import operator
cap = cv2.VideoCapture(1)
st = ""
fullexp = " "
expr = []
exp2  = []
slop = []
inte = []
flag = False

no_of_times = 0

while(True):
    #reading frames from camera
    blanck=np.zeros((300,300,3),np.uint8)
    ret, frame1 = cap.read()
    
    frame2 = frame1
    frame = frame2.copy()
    frame_new= frame2.copy()
    
    #finding contours
    ret, img = cv2.threshold(frame, 110, 255, cv2.THRESH_BINARY_INV)
    cvt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images,contours, hierarchy = cv2.findContours(cvt,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    thisdict = {}
    
    flag=0
    noted_y=0
    mylist = []
    for c in contours:
        (x, y, w, h)= cv2.boundingRect(c)
        if (w>45) or (h>45):
            #cv2.rectangle(frame1,(x,y),(x+w,y+h), (255,0, 0), 3)
            mylist.append((x,y,w,h))
            
                    
    for i in range(0, len(mylist)):
        x = mylist[i][0]
        y = mylist[i][1]
        w = mylist[i][2]
        h = mylist[i][3]

        for j in range(0, len(mylist)):
            x1 = mylist[j][0]
            y1 = mylist[j][1]
            w1 = mylist[j][2]
            h1 = mylist[j][3]

            if abs(x1-x)<10 and y1 != y:
                flag = 1
                mylist.remove((x1,y1,w1,h1))
                break
            
        if flag is 1:
            break
    
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
        cv2.rectangle(frame1,(x,y),(x,y), (0,0, 255), 2)
        img1 = frame_new[y:y+h, x:x+w]
        ret, gray = cv2.threshold(img1,120,255,cv2.THRESH_BINARY_INV )
        try:
            im = cv2.resize(gray, (40,40))
            im1 = cv2.resize(im, (28,28))
            next2 = cv2.resize(im1, (28,28))
            ret, gray1 = cv2.threshold(next2,130,255,cv2.THRESH_BINARY_INV)

            ar = np.array(gray1).reshape((28,28,3))
            ar = np.expand_dims(ar, axis=0)
            prediction = model.predict(ar)[0]

            #predicrion of class labels
            for i in range(0,19):
                if prediction[i]==1.0:
                    if i==0:
                        j= "+"
                    if i==1:
                        j= "-"
                    if i==2:
                        j= "0"
                    if i==3:
                        j= "1"
                    if i==4:
                        j= "2"
                    if i==5:
                        j= "3"
                    if i==6:
                        j= "4"
                    if i==7:
                        j= "5"
                    if i==8:
                        j= "6"
                    if i==9:
                        j= "7"
                    if i==10:
                        j= "8"
                    if i==11:
                        j= "9"
                    #if i==12:
                       # j= "="
                   # if i==13:
                     #   j= "^"
                    #if i==14:
                       # j= "/"
                   # if i==15:
                      #  j= "X"
                   # if i==16:
                     #   j= "x"
                   # if i==17:
                        #j= "y" 
                    
                    cv2.putText(frame1, j, (x+70,y+200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
                    thisdict[x]= str(j)
        except:
            d=0

    sort = sorted(thisdict.items(), key=operator.itemgetter(0))
    s = ""
   
    for x in range(0,len(sort)):
        s=s+str(sort[x][1])
    solution=""    
    try:      
        solution=(eval(s))
    except:pass  
        
    cv2.putText(frame1, s, (70,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)  
    cv2.putText(blanck, 'Detected: ', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1,cv2.LINE_AA)
    cv2.putText(blanck, s, (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1,cv2.LINE_AA)
    cv2.putText(blanck, str(solution), (10,160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1,cv2.LINE_AA)
    if (st != s):
        no_of_times = 0
        st = s
        
   
    try:
        list_n1 = add_star(s, frame1) 
        list_n = add_doublestar(list_n1)
        
        fullexp =convert(list_n)
        print(fullexp)
        cv2.putText(blanck, s, (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1,cv2.LINE_AA)
        if fullexp[0]=='*':
            fullexp=fullexp[1:]
            print(fullexp)
        
        verify_lst=[]
        [verify_lst.append(i) for i in fullexp]
        if verify_lst[0] =='x':
            output(fullexp)
        
        no_print =0
        while no_print < no_of_times :
            if no_of_times == 1:
                xi = [-2]
                tempy, ans = print_fun(xi, expr, frame1)   
                frame1 = plot_graph1(xi, tempy, ans, frame1)
            if no_of_times == 2:
                xi = [-2, -1]
                tempy, ans = print_fun(xi, expr, frame1)   
                frame1 = plot_graph1(xi, tempy, ans, frame1)
            if no_of_times == 3:
                xi = [-2, -1, 0]
                tempy, ans = print_fun(xi, expr, frame1)   
                frame1 = plot_graph1(xi, tempy, ans, frame1)
            if no_of_times == 4:
                xi = [-2, -1, 0, 1]
                tempy, ans = print_fun(xi, expr, frame1)   
                frame1 = plot_graph1(xi, tempy, ans, frame1)
            if no_of_times == 5:
                xi = [-2, -1, 0, 1, 2]
                tempy, ans = print_fun(xi, expr, frame1)   
                frame1 = plot_graph1(xi, tempy, ans, frame1)
            if no_of_times == 6:
                xi = [-2, -1, 0, 1, 2]
                tempy, ans = print_fun(xi, expr, frame1)  
                frame1 = plot_graph1(xi, tempy, ans, frame1)
                frame1 = plot_graph2(xi, tempy, ans, frame1, expr)
            no_print+=1
        
    
    except:
        d=0
    cv2.imshow('frame', frame1)
    cv2.imshow('frame1', img)
    cv2.imshow('calculation',blanck)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break        
cap.release()
cv2.destroyAllWindows()

