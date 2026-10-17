"""Prepare once; training only reads and verifies the bundled corpus."""

import hashlib
from pathlib import Path
import urllib.request

DATA_PATH = Path(__file__).parent / "data" / "input.txt"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def read_verified_data() -> bytes:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Missing corpus. Run: uv run prepare.py")
    data = DATA_PATH.read_bytes()
    if hashlib.sha256(data).hexdigest() != DATA_SHA256:
        raise ValueError("Corpus checksum differs from the fixed workshop dataset.")
    return data


if __name__ == "__main__":
    if not DATA_PATH.exists():
        data = urllib.request.urlopen(DATA_URL, timeout=30).read()
        if hashlib.sha256(data).hexdigest() != DATA_SHA256:
            raise ValueError("Downloaded corpus has an unexpected checksum.")
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        DATA_PATH.write_bytes(data)
    data = read_verified_data()
    print(f"Verified {len(data):,} bytes: {DATA_SHA256}")
