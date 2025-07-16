import cv2
import time
from PIL import Image
import imagehash

class DuplicateFilter:
    """Detect duplicate frames using perceptual hashing with an optional bypass.

    When used with RTSP sources, ``mpdecimate`` in FFmpeg is recommended to drop
    duplicates before they reach Python. This filter provides a fallback for
    other sources.
    """
    def __init__(self, threshold: int = 2, bypass_seconds: int = 2):
        self.threshold = threshold
        self.bypass_seconds = bypass_seconds
        self.prev = None
        self.bypass_until = 0.0

    def is_duplicate(self, frame) -> bool:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((64, 64))
        ph = imagehash.phash(img)
        if self.prev is None:
            self.prev = ph
            return False
        diff = self.prev - ph
        self.prev = ph
        if diff > self.threshold:
            self.bypass_until = time.time() + self.bypass_seconds
            return False
        if time.time() < self.bypass_until:
            return False
        return True
