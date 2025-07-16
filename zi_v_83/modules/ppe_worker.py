import time
import json
import threading
from pathlib import Path
import cv2
from loguru import logger
from ultralytics import YOLO
import redis
import torch


class PPEDetector(threading.Thread):
    """Background worker that reads person_logs and performs PPE detection."""

    def __init__(self, cfg: dict, redis_url: str, snap_dir: Path, update_callback=None):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.redis = redis.Redis.from_url(redis_url)
        self.model = YOLO(cfg.get("ppe_model", "mymodelv5.pt"))
        self.device = cfg.get("device", "cpu")
        if not self.device or self.device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
        logger.info(f"Loading PPE model {cfg.get('ppe_model', 'mymodelv5.pt')} on {self.device}")
        if self.device.startswith("cuda"):
            self.model.model.to(self.device).half()
        else:
            self.model.model.to(self.device)
        self.last_ts = 0
        self.snap_dir = Path(snap_dir)
        self.running = True
        self.update_callback = update_callback

    def _process_entry(self, entry: dict) -> None:
        """Run the PPE model on a single log entry."""
        tasks = entry.get("ppe_tasks", self.cfg.get("track_ppe", []))
        if not tasks:
            return
        path = entry.get("path")
        if not path:
            return
        img_path = Path(path)
        if not img_path.is_absolute():
            img_path = self.snap_dir / img_path.name
        img = cv2.imread(str(img_path))
        if img is None:
            return
        res = self.model.predict(img, device=self.device, verbose=False)[0]
        scores = {}
        for *xyxy, conf, cls in res.boxes.data.tolist():
            label = self.model.names[int(cls)]
            if conf > scores.get(label, 0):
                scores[label] = conf
        for item in tasks:
            conf = scores.get(item, 0)
            status = item if conf >= self.cfg.get("helmet_conf_thresh", 0.5) else f"no_{item}"
            ts = int(time.time())
            log = {
                "ts": ts,
                "cam_id": entry.get("cam_id"),
                "track_id": entry.get("track_id"),
                "status": status,
                "conf": conf,
                "path": str(img_path.name),
            }
            self.redis.zadd("ppe_logs", {json.dumps(log): ts})
            if status.startswith('no_'):
                self.redis.incr(f"{status}_count")
            if self.update_callback:
                self.update_callback()

    def run(self):
        while self.running:
            # process queued requests first
            while True:
                queued = self.redis.lpop("ppe_queue")
                if not queued:
                    break
                try:
                    entry = json.loads(queued)
                except Exception:
                    continue
                self._process_entry(entry)

            # fallback to reading person_logs for backward compatibility
            entries = [
                json.loads(e)
                for e in self.redis.zrangebyscore("person_logs", self.last_ts + 1, "+inf")
            ]
            for entry in entries:
                self.last_ts = max(self.last_ts, entry.get("ts", 0))
                self._process_entry(entry)

            time.sleep(1)
