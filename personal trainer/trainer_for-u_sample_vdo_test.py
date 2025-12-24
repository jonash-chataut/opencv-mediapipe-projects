import cv2
import numpy as np
import mediapipe as mp
import time
import Module_pose_detection as mpd


cap =cv2.VideoCapture("personal trainer\pushup sample vidoes\pushup2.mp4")
# cap =cv2.VideoCapture(0)
pTime=0
detector=mpd.poseDetector()
count=0
dir=0


# img=cv2.imread("personal trainer\sample videos\dumble.jpg")
while True:
    success,img=cap.read()
    img=cv2.resize(img,(1280,720))

    # img=cv2.imread("personal trainer\sample videos\dumble.jpg")
    img=detector.findPose(img,False)
    lmlist=detector.findPosition(img,False)
    # print(lmlist)
    if len(lmlist)!=0:
        # left arm
        angle=detector.findAngle(img,11,13,15)
        # right arm
        # angle=detector.findAngle(img,12,14,16)

        # for the sample video pushup2 and left hand
        per=np.interp(angle,(65,155),(100,0))
        bar=np.interp(angle,(65,155),(100,650))

        # for webcam angle in rt hand: 40 to 140
        # per=np.interp(angle,(40,140),(0,100))
        # bar=np.interp(angle,(40,140),(650,100))


        # check for the curls
        
        color=(0,255,0)
        if per == 0:
            # color=(255,0,255)
            if dir==0: #going down this means 
                count+=0.5
                dir=1
        if per==100:
            color=(255,0,255)
            if dir==1: #going up
                count+=0.5
                dir=0
        # print(count)


        # draw bar
        cv2.rectangle(img,(1100,100),(1175,650),(0,255,0),2)
        cv2.rectangle(img,(1100,int(bar)),(1175,650),color,cv2.FILLED)
        cv2.putText(img,f"{int(per)}",(1100,75),cv2.FONT_HERSHEY_PLAIN,4,(255,0,0),4)

        # draw call count
        cv2.rectangle(img,(0,450),(250,720),(0,255,0),cv2.FILLED)
        cv2.putText(img,f"{int(count)}",(50,650),cv2.FONT_HERSHEY_PLAIN,15,(255,0,0),25)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)

