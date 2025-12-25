import cv2
import time
import numpy as np
import handtracking_module as htm
import autopy

wScr,hScr=autopy.screen.size()
wCam,hCam=640,480
frameR=100 # frame reduction
smothening=7

pTime=0
plocX,plocY=0,0
clocX,clocY=0,0

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

detector=htm.handDetector(maxHands=1)


while True:
    # find hand Landmarks
    success,img=cap.read()
    img=detector.findHands(img)
    lmlist,bbpx=detector.findPosition(img,draw=False)

    # get the tip of the index and middle fingers
    if len(lmlist)!=0:
        x1,y1=lmlist[8][1:]
        x2,y2=lmlist[12][1:]

        # check which fingers are up
        fingers=detector.fingerUP()

        cv2.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)


        # only index finger: Moving mouse
        if fingers[1]==1 and fingers[2]==0:

            # Convert Coordinates

            x3=np.interp(x1,(frameR,wCam-frameR),(0,wScr))
            y3=np.interp(y1,(frameR,hCam-frameR),(0,hScr))

            # Smoothen Values
            clocX=plocX+(x3-plocX)/smothening
            clocY=plocY+(y3-plocY)/smothening



            # Move mouse
            autopy.mouse.move(wScr-clocX,clocY)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            plocX,plocY=clocX,clocY

        # Both index and middle fingers are up: Clicking mode
        if fingers[1]==1 and fingers[2]==1:
            # Find distance between fingers
            length,img,lineinfo=detector.findDistance(8,12,img)
            # print(length)
            # Click mouse if distance short
            if length<30:
                cv2.circle(img,(lineinfo[4],lineinfo[5]),15,(0,255,0),cv2.FILLED)
                autopy.mouse.click()
                

        

        

    # Frame rate  
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime  
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    # Display
    cv2.imshow("Image",img)
    cv2.waitKey(1)
