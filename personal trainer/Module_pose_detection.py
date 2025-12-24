import cv2
import mediapipe as mp
import time
import math

mpDraw=mp.solutions.drawing_utils
mpPose=mp.solutions.pose
pose=mpPose.Pose()


class poseDetector():
    def __init__(self,mode=False,upBody=False,smoothness=True,detectionCon=0.5,trackingCon=0.5):
        self.mode=mode
        self.upBody=upBody
        self.smoothness=smoothness
        self.detectionCon=detectionCon
        self.trackingCon=trackingCon
        self.mpDraw=mp.solutions.drawing_utils
        self.mpPose=mp.solutions.pose
        self.pose=self.mpPose.Pose(static_image_mode=self.mode,
            smooth_landmarks=self.smoothness,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackingCon)

    def findPose(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self,img,draw=True):
        self.lmlist=[]
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                self.lmlist.append([id,cx,cy])
                if draw:  
                    cv2.circle(img,(cx,cy),3,(255,0,0),cv2.FILLED)
        return self.lmlist   

    def findAngle(self,img,p1,p2,p3,draw=True):
        # Get landmarks
        x1,y1=self.lmlist[p1][1:]
        _,x2,y2=self.lmlist[p2]
        x3,y3=self.lmlist[p3][1:]

        # calculate the angle
        angle=math.degrees(math.atan2(y1-y2,x1-x2)-math.atan2(y3-y2,x3-x2))

        if angle<0:
            angle+=360

        # draw the points 
        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
            cv2.line(img,(x3,y3),(x2,y2),(255,255,255),3)
            cv2.circle(img,(x1,y1),10,(0,0,255),cv2.FILLED)
            cv2.circle(img,(x1,y1),15,(0,0,255),2)
            cv2.circle(img,(x2,y2),10,(0,0,255),cv2.FILLED)
            cv2.circle(img,(x2,y2),15,(0,0,255),2)
            cv2.circle(img,(x3,y3),10,(0,0,255),cv2.FILLED)
            cv2.circle(img,(x3,y3),15,(0,0,255),2)
            
            # to display angle
            # cv2.putText(img,str(int(angle)),(x2-60,y2+50),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255))
        return angle
            







def main():
    pTime=0
    cap=cv2.VideoCapture("C:\skills\python\learnings\libraries learning\OpenCV\mediapipe_learn\pose_tracking\pose_videos\V1.mp4")
    detector=poseDetector()
    
    while True:
        success,img=cap.read()
        img=detector.findPose(img)
        lmlist=detector.findPosition(img,draw=False)
        if len(lmlist) != 0:
            print(lmlist[14])
            cv2.circle(img,(lmlist[14][1],lmlist[14][2]),15,(0,0,255),cv2.FILLED)
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow("Pose",img)
        cv2.waitKey(1)


if __name__=="__main__":
    main()
