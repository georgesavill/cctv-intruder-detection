# Script to handle input video stream from CCTV camera.

import cv2

video_feed = cv2.VideoCapture("http://username:password@192.168.1.93:8080/video")

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output_video = cv2.VideoWriter( 'output.avi', fourcc, 20.0, (720,720))

while (video_feed.isOpened()):
    feed_opened, frame = video_feed.read()

    if not feed_opened:
        break

    output_video.write(frame)
    # Crop image to region of interest
    # frame = frame[350:900, 538:1088]

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

output_video.release()
video_feed.release()
cv2.destroyAllWindows()