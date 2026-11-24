import cv2
import os

rtsp_url = os.getenv("TAPO_RTSP_URL")

if not rtsp_url:
    print("RTSP URL is missing!")
    exit()

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("CCTV connection failed!")
    exit()

print("Tapo C500 connected successfully!")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Unable to receive video")
        break

    cv2.imshow("Tapo C500 LIVE CCTV", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()