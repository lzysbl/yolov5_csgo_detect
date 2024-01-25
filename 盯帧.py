import cv2
# 读取视频,用来盯帧
video = cv2.VideoCapture("runs/detect/exp10/0.mp4")
frame = video.read()[1]
while(frame is not None):
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
    frame = video.read()[1]
