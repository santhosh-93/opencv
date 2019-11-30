#identifying type of traingle formed by 3 aruco markers
## Installation
''' SH
pip install opencv-python
pip install numpy
pip install aruco

'''
import numpy as np
import cv2
import cv2.aruco as arc
import matplotlib.pyplot as plt

## requirements
<p>prinout  3 aruco markers of 6x6 from: [create_aruco](http://chev.me/arucogen/)<br>
specify the 6x6 and number id and get printouts



## opens the camera(0-for lapcam,1-external connected webcam)
cap = cv2.VideoCapture(1)

##'''python code
while True:
    #finding contours
#Contours can be explained simply as a curve joining all the continuous points (along the boundary),
# having same color or intensity. The contours are a useful tool for shape analysis and object detection and recognition.
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    arc_dict = arc.Dictionary_get(arc.DICT_6X6_250)
    arc_param = arc.DetectorParameters_create()
    # Detect Aruco markers    
    corners,ids,rejectedImgPoints = arc.detectMarkers(gray,arc_dict,parameters=arc_param)
    a=[]
    ad = None
    bo = 0
    if np.all(ids) != None:
        size = ids.size
        dsp = arc.drawDetectedMarkers(frame,corners)
        
        print(ids)

       ### detecting centre of aruco markers
<p>From the centre of aruco markers line are drawn, below code finds the centre co-ordinates</p> 
'''python
    try:
        for i in range(0,ids.size):
            
            if(ids[i]==19):
                cx=int(corners[i][0][0][0])
                cy=int(corners[i][0][0][1])
                c1x=int(corners[i][0][2][0])
                c1y=int(corners[i][0][2][1])
                c2x=int(corners[i][0][1][0])
                c2y=int(corners[i][0][1][1])

            elif(ids[i]==16):
                bx=int(corners[i][0][0][0])
                by=int(corners[i][0][0][1])
                b1x=int(corners[i][0][2][0])
                b1y=int(corners[i][0][2][1])
            elif(ids[i]==15):
                dx=int(corners[i][0][0][0])
                dy=int(corners[i][0][0][1])
                d1x=int(corners[i][0][2][0])
                d1y=int(corners[i][0][2][1])

        #centre of aruco markers
        px=int((cx+c1x)/2)#1
        py=int((cy+c1y)/2)#1
        qx=int((bx+b1x)/2)#4
        qy=int((by+b1y)/2)#4
        rx=int((dx+d1x)/2)#5
        ry=int((dy+d1y)/2)#5
        <p> 2 markers create line.  three marekrs added creates triangle </p>
        ''' python
        if len(ids)==3:
            #Syntax: cv2.circle(image, center_coordinates, radius, color, thickness)
            cv2.circle(frame, (px,py), 3, (255, 0,0), 1)
            cv2.circle(frame, (qx,qy), 3, (255, 0,0), 1)
            cv2.circle(frame, (rx,ry), 3, (255, 0,0), 1)
            #Syntax: cv2.line(image, start_point, end_point, color, thickness)
            #draw in frame
            cv2.line(frame, (int(px), int(py)), (int(qx), int(qy)), (0,255,0), 5)
            cv2.line(frame, (int(px), int(py)), (int(rx), int(ry)), (0,255,0), 5)
            cv2.line(frame, (int(rx), int(ry)), (int(qx), int(qy)), (0,255,0), 5)
           
            #Syntax: cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
            if (px,py)==(qx,qy)==(rx,ry):
                cv2.putText(frame,"Equilateral triangle",cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
            elif(px,py)==(qx,qy)or (qx,qy)==(rx,ry) or(rx,ry)== (px,py):
                cv2.putText(frame,"isosceles triangle",cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
            else:
                cv2.putText(frame,"isosceles triangle",cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)    

    except: pass
##displays the image [link](https://www.dropbox.com/s/85e56spnvr5q2kc/identify%20type%20of%20triangle%20i.jpg?dl=0)
    cv2.imshow('output',frame)
    #cv2.imshow('Display',frame)
       
## Wait on this frame and press q to exit 
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    
cap.release()
cv2.destroyAllWindows()