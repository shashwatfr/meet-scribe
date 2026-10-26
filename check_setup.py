"""Setup check: are ffmpeg and ffprobe installed, and is every API key in .env?

Run it by hand after installing (uv run python check_setup.py). It prints
"present" or "missing" for each key, never the key itself.
"""
import shutil
import sys

from config import REQUIRED_KEYS, missing_keys

ok = True

for tool in ("ffmpeg", "ffprobe"):
    found = shutil.which(tool) is not None
    print(f"{tool + ':':<16} {'found' if found else 'NOT FOUND'}")
    ok = ok and found

missing = missing_keys()
for name in REQUIRED_KEYS:
    status = "missing" if name in missing else "present"
    print(f"{name + ':':<16} {status}")
ok = ok and not missing

sys.exit(0 if ok else 1)
