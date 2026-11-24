
import os
import time
import vlc
from urllib.parse import quote
from getpass import getpass

VLC_DIR = r"C:\Program Files\VideoLAN\VLC"

os.add_dll_directory(VLC_DIR)
os.environ["VLC_PLUGIN_PATH"] = os.path.join(VLC_DIR, "plugins")

username = input("Enter camera username: ").strip()
password = getpass("Enter camera password: ")

username = quote(username, safe="")
password = quote(password, safe="")

url = f"rtsp://{username}:{password}@192.168.0.184:554/stream1"

instance = vlc.Instance(
    "--intf=dummy",
    "--no-video-title-show",
    "--verbose=2",
    "--rtsp-tcp",
    "--plugin-path=C:\\Program Files\\VideoLAN\\VLC\\plugins"
)

player = instance.media_player_new()
media = instance.media_new(url)
media.add_option(":rtsp-tcp")
player.set_media(media)

print("Starting CCTV test...")
player.play()

for second in range(20):
    time.sleep(1)
    state = player.get_state()
    print(f"Second {second + 1}: State.{state}")

    if state == vlc.State.Ended:
        break

print("Final state:", player.get_state())
input("Press ENTER to stop CCTV...")

player.stop()
player.release()
media.release()
instance.release()
