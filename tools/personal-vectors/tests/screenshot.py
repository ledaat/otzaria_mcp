"""CI: start the built .exe on the synthetic Otzaria folder and photograph the screen once its
check has run, so the window (Hebrew right-to-left, the summary) can be looked at."""
import os, subprocess, sys, time
from PIL import ImageGrab

exe, appdata, out = sys.argv[1], sys.argv[2], sys.argv[3]
env = dict(os.environ, APPDATA=appdata, LOCALAPPDATA=os.path.join(appdata, "local"))
p = subprocess.Popen([exe], env=env)
time.sleep(float(os.environ.get("WAIT", "40")))
ImageGrab.grab().save(out)
alive = p.poll() is None
p.kill()
print("window still open after the wait:", alive)
sys.exit(0 if alive else 1)
