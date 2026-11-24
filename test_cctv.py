
import subprocess
import imageio_ffmpeg
from getpass import getpass
from urllib.parse import quote

CAMERA_IP = "192.168.0.184"

username = quote(input("CCTV username: "), safe="")
password = quote(getpass("CCTV password: "), safe="")

rtsp_url = (
    f"rtsp://{username}:{password}"
    f"@{CAMERA_IP}:554/stream1"
)

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

command = [
    ffmpeg,
    "-hide_banner",
    "-loglevel", "error",
    "-rtsp_transport", "tcp",
    "-i", rtsp_url,
    "-t", "5",
    "-f", "null",
    "-"
]

print("\nTesting Tapo CCTV with FFmpeg...")

result = subprocess.run(
    command,
    capture_output=True,
    text=True,
    timeout=20
)

print("\nFFmpeg return code:", result.returncode)
print("\nFFmpeg error details:")

if result.stderr:
    print(result.stderr)
else:
    print("No error message received.")

if result.returncode == 0:
    print("\nSUCCESS: FFmpeg received the CCTV stream.")
