
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import cv2.aruco as arc
import matplotlib.pyplot as plt


cap = cv2.VideoCapture(1)

while True:
    
    ret,frame = cap.read()
    
    im_dst = frame 
    #arc=aruco
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Constant parameters used in Aruco methods
    arc_dict = arc.Dictionary_get(arc.DICT_6X6_250)
    arc_param= arc.DetectorParameters_create()
      # Detect Aruco markers
    corners,ids,rejectedImgPoints = arc.detectMarkers(gray,arc_dict,parameters=arc_param)
    
    if np.all(ids) != None:
        disp = arc.drawDetectedMarkers(frame, corners)
        im_dst = frame
        cv2.imshow('Display',im_dst) 
    else:
        display = frame
        cv2.imshow('Display',display) 
          # Wait on this frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()

