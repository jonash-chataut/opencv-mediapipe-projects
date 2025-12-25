import cv2
import numpy as np
import mediapipe as mp
import time
import handtracking_module as htm
import os

folderPath=r"virtual painter using hand\virtual designs"
mylist=os.listdir(folderPath)
# print(mylist)
overLayList=[]

for imPath in mylist:
    image=cv2.imread(f"{folderPath}/{imPath}")
    overLayList.append(image)
# print(len(overLayList))

header=overLayList[0]
drawColor=(255,0,255)
brushThickness=3
eraserThickness=100



cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector=htm.handDetector(detectionCon=0.85)
xp,yp=0,0

imgCanvas=np.zeros((720,1280,3),np.uint8)
while True:
    # Import image
    success,img=cap.read()
    img=cv2.flip(img,1)


    # Find hand landmarks
    img=detector.findHands(img)
    lmlist=detector.findPosition(img,draw=False)

    if len(lmlist)!=0:
        # print(lmlist)

        x1,y1=lmlist[8][1:] #tip of index finger
        x2,y2=lmlist[12][1:] #tip of middle finger
        # print(x1,y1)

        # Check which fingers are up
        fingers=detector.fingerUP()
        # print(fingers)

        # if selection mode i.e two fingers are up
        # 300 to 400 500 to 610 800 to 930 1075 to 1200
        if fingers[1] and fingers[2]:
            xp,yp=0,0
            cv2.rectangle(img,(x1,y1-30),(x2,y2+30),drawColor,cv2.FILLED)
            if(y1<125):
                if 300<x1<400:
                    header=overLayList[0]
                    drawColor=(255,0,255)
                elif 500<x1<610:
                    header=overLayList[1]
                    drawColor=(255,0,0)
                elif 800<x1<930:
                    header=overLayList[2]
                    drawColor=(0,255,0)
                elif 1075<x1<1200:
                    header=overLayList[3]
                    drawColor=(0,0,0)
                
    
            # print("selection mode")

    
        # if draw mode i.e index finger up
        if fingers[1] and fingers[2]==False:
            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
            # print("Drawing mode")
            if xp==0 and yp==0:
                xp,yp=x1,y1
            
            if drawColor==(0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)

            cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
            cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
            xp,yp=x1,y1

    imgGray=cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _,imgInv=cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv=cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,imgInv)
    img=cv2.bitwise_or(img,imgCanvas)
    # setting the header image
    img[0:125,0:1280]=header

    # img=cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image",img)
    # cv2.imshow("Canvas",imgCanvas)
    cv2.waitKey(1)
