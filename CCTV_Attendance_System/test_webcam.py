
import cv2
import time

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Camera could not open.")
    exit()

camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Camera opened successfully.")
print("Preparing webcam...")

# Allow the camera time to adjust exposure
for i in range(30):
    success, frame = camera.read()
    time.sleep(0.1)

if not success:
    print("Unable to receive camera frames.")
else:
    print("Brightness:", frame.mean())

    while True:
        success, frame = camera.read()

        if not success:
            print("Frame capture failed.")
            break

        cv2.imshow("Live Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()

print("Webcam test completed.")
