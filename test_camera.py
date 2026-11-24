
import cv2
from urllib.parse import quote
import sys

# ==========================================
# TAPO C500 CAMERA SETTINGS
# ==========================================

CAMERA_IP = "192.168.0.184"
CAMERA_PORT = 554

# Enter your Tapo Camera Account credentials
USERNAME = "varshini"
PASSWORD = "30963096"

# Use stream2 (lower resolution)
STREAM_PATH = "/stream2"

# ==========================================
# CREATE RTSP URL
# ==========================================

encoded_username = quote(USERNAME, safe="")
encoded_password = quote(PASSWORD, safe="")

RTSP_URL = (
    f"rtsp://{encoded_username}:{encoded_password}"
    f"@{CAMERA_IP}:{CAMERA_PORT}{STREAM_PATH}"
)

# ==========================================
# CONNECT TO CAMERA
# ==========================================

print("=" * 45)
print("       TAPO C500 CAMERA TEST")
print("=" * 45)

print("Camera IP:", CAMERA_IP)
print("RTSP Port:", CAMERA_PORT)
print("Stream:", STREAM_PATH)
print("\nConnecting to camera...")

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ==========================================
# CHECK CONNECTION
# ==========================================

if not cap.isOpened():
    print("\nERROR: Camera connection failed!")
    print("Check camera account credentials and RTSP settings.")
    sys.exit(1)

print("\nSUCCESS: Camera connected!")
print("Live video started.")
print("Press Q to exit.")

# ==========================================
# DISPLAY LIVE CCTV
# ==========================================

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Cannot receive video frame.")
            break

        cv2.imshow("Tapo C500 - Live CCTV", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Closing camera...")
            break

except KeyboardInterrupt:
    print("\nCamera stopped.")

finally:
    cap.release()
    cv2.destroyAllWindows()

    print("Camera disconnected.")
