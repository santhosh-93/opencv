
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import cv2.aruco as arc
import matplotlib.pyplot as plt


cap = cv2.VideoCapture(1)
while True:
    
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    arc_dict = arc.Dictionary_get(arc.DICT_6X6_250)
    arc_param = arc.DetectorParameters_create()
    
    corners,ids,rejectedImgPoints = arc.detectMarkers(gray,arc_dict,parameters=arc_param)
    a=[]
    ad = None
    bo = 0
    if np.all(ids) != None:
        size = ids.size
        dsp = arc.drawDetectedMarkers(frame,corners)
        
        print(ids)
        
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
        
        if len(ids)==3:
            #Syntax: cv2.circle(image, center_coordinates, radius, color, thickness)
            cv2.circle(frame, (px,py), 3, (255, 0,0), 1)
            cv2.circle(frame, (qx,qy), 3, (255, 0,0), 1)
            cv2.circle(frame, (rx,ry), 3, (255, 0,0), 1)
            #Syntax: cv2.line(image, start_point, end_point, color, thickness)
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

    cv2.imshow('output',frame)
    #cv2.imshow('Display',frame)
       
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    
cap.release()
cv2.destroyAllWindows()

