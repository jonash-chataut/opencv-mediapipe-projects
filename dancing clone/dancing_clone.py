import cv2
import mediapipe as mp
from collections import deque

cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# 2 seconds delay (~30 FPS)
pose_buffer = deque(maxlen=60)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    # ---------------- CURRENT SKELETON ----------------
    if results.pose_landmarks:
        landmarks = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
        pose_buffer.append(landmarks)

        mp_draw.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # ---------------- CLONE SKELETON ----------------
    if len(pose_buffer) == pose_buffer.maxlen:
        delayed = pose_buffer[0]

        # Draw bones
        for start, end in mp_pose.POSE_CONNECTIONS:
            x1, y1 = delayed[start]
            x2, y2 = delayed[end]

            # mirror + shift clone
            p1 = (int((1 - x1) * w) + 50, int(y1 * h))
            p2 = (int((1 - x2) * w) + 50, int(y2 * h))

            cv2.line(img, p1, p2, (255, 0,255), 3)

        # Draw joints
        for x, y in delayed:
            cx = int((1 - x) * w) + 50
            cy = int(y * h)
            cv2.circle(img, (cx, cy), 5, (255,0 , 255), -1)

    cv2.imshow("Dance Clone Skeleton", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
