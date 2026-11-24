
import os
import sys

VLC_DIR = r"C:\Program Files\VideoLAN\VLC"

print("Python:", sys.executable)
print("VLC folder exists:", os.path.exists(VLC_DIR))
print("VLC executable exists:", os.path.exists(
    os.path.join(VLC_DIR, "vlc.exe")
))
print("VLC plugins folder exists:", os.path.exists(
    os.path.join(VLC_DIR, "plugins")
))

os.add_dll_directory(VLC_DIR)
os.environ["VLC_PLUGIN_PATH"] = os.path.join(VLC_DIR, "plugins")

import vlc

print("Python VLC version:", vlc.__version__)

instance = vlc.Instance("--intf=dummy", "--no-video-title-show")

if instance:
    print("VLC engine initialized successfully")
    instance.release()
else:
    print("VLC engine initialization failed")
