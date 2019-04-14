import numpy as np
import cv2

video_feed = cv2.VideoCapture("http://username:password@192.168.1.93:8080/video")

while (video_feed.isOpened()):
    feed_opened, frame = video_feed.read()

    if not feed_opened:
        break

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_feed.release()
cv2.destroyAllWindows()