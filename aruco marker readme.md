# ArUco marker detection 
<p> 1 ArUco marker is used here
## Installation
```sh

pip install opencv-python
pip install numpy
pip install aruco

'''
## requirements

<p>prinout  a aruco marker of 6x6 from: [create_aruco](http://chev.me/arucogen/)<br>
specify the 6x6 and number id and get printouts
</p>

## configurations
### detecting ID and corners
<p>detectMarkers fuction detects its corners </p>

## opens the camera(0-for lapcam,1-external connected webcam)
cap = cv2.VideoCapture(1)

'''python
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()
 # Detect Aruco markers [link](https://www.dropbox.com/s/4rhpxhtrvmoh4su/aruco%20detect%20i.jpg?dl=0)   
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict, parameters=parameters)

'''
 if np.all(ids) != None:
        disp = arc.drawDetectedMarkers(frame, corners)
        im_dst = frame
##displayes the image
        cv2.imshow('Display',im_dst) 
    else:
        display = frame
        cv2.imshow('Display',display) 

## Wait on this frame and press q to exit 

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
cap.release()

cv2.destroyAllWindows()
