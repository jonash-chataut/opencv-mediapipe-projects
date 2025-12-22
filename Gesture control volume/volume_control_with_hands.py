import cv2
import mediapipe as mp
import time
import numpy as np
import math
import handtracking_module as htm

# pycaw code used for volume control of our device
from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities,IAudioEndpointVolume

wCam, hCam =640,480

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime=0

detector=htm.handDetector(detectionCon=0.7)


# Get default speakers
device = AudioUtilities.GetSpeakers()

# Cast to volume control
volume = device.EndpointVolume

# set volume
# volume.SetMasterVolumeLevel(-60,None)
# Get volume range
volRange=volume.GetVolumeRange()
minVol=volRange[0]
maxVol=volRange[1]
vol=0
myDeviceVol=volume.GetMasterVolumeLevel()
myDeviceVolPer=np.interp(myDeviceVol,[minVol,maxVol],[0,100])
myDeviceVolBar=np.interp(myDeviceVol,[minVol,maxVol],[400,150])

volBar=myDeviceVolBar
volPer=myDeviceVolPer


while True:
    success,img=cap.read()
    img=detector.findHands(img)
    lmlist=detector.findPosition(img,draw=False)
    if len(lmlist)!=0:
        # print(lmlist[4],lmlist[8])
        x1,y1=lmlist[4][1],lmlist[4][2]
        x2,y2=lmlist[8][1],lmlist[8][2]
        cx,cy=(x1+x2)//2,(y1+y2)//2


        cv2.circle(img,(x1,y1),10,(255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),10,(255,0,255),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        # center of line dot 
        cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)

        length=math.hypot(x2-x1,y2-y1)
        # print(length)

        # Hand range 200 to 50
        # volume range -65 to 0

        vol=np.interp(length,[50,210],[minVol,maxVol])
        volBar=np.interp(length,[50,210],[400,150])
        volPer=np.interp(length,[50,210],[0,100])
        # print(vol)
        volume.SetMasterVolumeLevel(vol,None)


        if length<50:
            cv2.circle(img,(cx,cy),10,(0,255,0),cv2.FILLED)

    cv2.rectangle(img,(50,150),(85,400),(8,255,0),3)
    cv2.rectangle(img,(50,int(volBar)),(85,400),(8,255,0),cv2.FILLED)
    cv2.putText(img,f"{int(volPer)}%",(40,450),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),3)



    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f"FPS: {str(int(fps))}",(10,70),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),3)

    cv2.imshow("Video ",img)
    cv2.waitKey(1)
