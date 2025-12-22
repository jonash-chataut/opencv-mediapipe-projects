import cv2
import mediapipe as mp
import time
import os

import handtracking_module as htm

cap = cv2.VideoCapture(0)
wCam, hCam =640,480
cap.set(3,wCam)
cap.set(4,hCam)
pTime=0

detector=htm.handDetector(detectionCon=0.7)

folderpath="projects/opencv mediapipe projects/finger counter/fingers"
mylist=os.listdir(folderpath)
# print(mylist)
overlayList=[]
for imPath in mylist:
    image =cv2.imread(f"{folderpath}/{imPath}")
    overlayList.append(image)

print(len(overlayList))

tipIds =[4,8,12,16,20]

while True:
    success,img=cap.read()

    img=detector.findHands(img)
    lmlist=detector.findPosition(img,draw=False)
    if len(lmlist)!=0:
        fingers=[]
        # just for right hands thumb
        if lmlist[tipIds[0]][1]>lmlist[tipIds[0]-1][1]:
                fingers.append(1)
        else:
            fingers.append(0)


        for id in range(1,5): #for 4 fingers except thumb
            if lmlist[tipIds[id]][2]<lmlist[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalfingers=fingers.count(1) #count the number of 1 in the list
        print(totalfingers)

        h, w, _ = overlayList[totalfingers-1].shape
        img[0:h, 0:w] = overlayList[totalfingers-1]

        cv2.rectangle(img,(20,255),(170,425),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(totalfingers),(45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),25)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime  
    cv2.putText(img,str(int(fps)),(550,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)