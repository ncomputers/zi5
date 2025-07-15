from __future__ import annotations
import queue
import threading
import time
import json
from datetime import date, datetime
import cv2
import torch
from loguru import logger
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import redis
from pathlib import Path
from .utils import send_email, lock, SNAP_DIR
from .duplicate_filter import DuplicateFilter
from core.config import ANOMALY_ITEMS, COUNT_GROUPS

class PersonTracker:
    """Tracks entry and exit counts using YOLOv8 and DeepSORT."""

    def __init__(self, cam_id: int, src: str, classes: list[str], cfg: dict,
                 tasks: dict | None = None,
                 src_type: str = "http", line_orientation: str | None = None,
                 reverse: bool = False, resolution: str = "original",
                 update_callback=None):
        self.cfg = cfg
        for k, v in cfg.items():
            setattr(self, k, v)
        self.cam_id = cam_id
        self.src = src
        self.src_type = src_type
        self.classes = classes
        self.tasks = tasks or {"counting": ["in", "out"], "ppe": []}
        self.count_classes = cfg.get("count_classes", [])
        self.ppe_classes = cfg.get("ppe_classes", [])
        self.alert_anomalies = cfg.get("alert_anomalies", [])
        self.line_orientation = line_orientation or cfg.get("line_orientation", "vertical")
        self.reverse = reverse
        self.resolution = resolution
        self.helmet_conf_thresh = cfg.get("helmet_conf_thresh", 0.5)
        self.detect_helmet_color = cfg.get("detect_helmet_color", False)
        self.track_misc = cfg.get("track_misc", True)
        self.show_lines = cfg.get("show_lines", True)
        self.show_ids = cfg.get("show_ids", True)
        self.show_track_lines = cfg.get("show_track_lines", False)
        self.duplicate_filter_enabled = cfg.get("duplicate_filter_enabled", False)
        self.duplicate_filter_threshold = cfg.get("duplicate_filter_threshold", 0.1)
        self.duplicate_bypass_seconds = cfg.get("duplicate_bypass_seconds", 2)
        self.max_retry = cfg.get("max_retry", 5)
        self.update_callback = update_callback
        self.online = False
        
        self.dup_filter = DuplicateFilter(self.duplicate_filter_threshold, self.duplicate_bypass_seconds) if self.duplicate_filter_enabled else None
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")
        self.device = cfg.get("device")
        if not self.device or self.device == "auto":
            self.device = "cuda:0" if cuda_available else "cpu"
        if self.device.startswith("cuda") and not cuda_available:
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
        logger.info(f"Loading person model {self.person_model} on {self.device}")
        self.model_person = YOLO(self.person_model)
        self.email_cfg = cfg.get("email", {})
        if self.device.startswith("cuda"):
            self.model_person.model.to(self.device).half()
            torch.backends.cudnn.benchmark = True
        else:
            self.model_person.model.to(self.device)
        self.tracker = DeepSort(max_age=5)
        self.frame_queue = queue.Queue(maxsize=10)
        self.tracks = {}
        self.redis = redis.Redis.from_url(self.redis_url)
        key_prefix = f"cam:{self.cam_id}:"
        self.key_in = key_prefix + "in"
        self.key_out = key_prefix + "out"
        self.key_date = key_prefix + "date"
        self.groups = cfg.get("track_objects", ["person"])
        self.in_counts = {}
        self.out_counts = {}
        for g in self.groups:
            self.in_counts[g] = int(self.redis.get(f"{self.key_in}_{g}") or 0)
            self.out_counts[g] = int(self.redis.get(f"{self.key_out}_{g}") or 0)
        self.in_count = sum(self.in_counts.values())
        self.out_count = sum(self.out_counts.values())
        stored_date = self.redis.get(self.key_date)
        self.prev_date = (
            date.fromisoformat(stored_date.decode()) if stored_date else date.today()
        )
        init_data = {self.key_date: self.prev_date.isoformat()}
        for g in self.groups:
            init_data[f"{self.key_in}_{g}"] = self.in_counts[g]
            init_data[f"{self.key_out}_{g}"] = self.out_counts[g]
        self.redis.mset(init_data)
        today = date.today().isoformat()
        for item in ANOMALY_ITEMS:
            date_key = f'{item}_date'
            count_key = f'{item}_count'
            d_raw = self.redis.get(date_key)
            d = date.fromisoformat(d_raw.decode()) if d_raw else self.prev_date
            if d.isoformat() != today:
                self.redis.mset({count_key: 0, date_key: today})
        self.snap_dir = SNAP_DIR
        self.output_frame = None
        self.running = True

    @staticmethod
    def _clean_label(name: str) -> str:
        """Normalize a label to lowercase with underscores."""
        return name.lower().replace(' ', '_').replace('-', '_').replace('/', '_')

    def task_enabled(self, task: str) -> bool:
        """Check if a given task is enabled for this camera."""
        if isinstance(self.tasks, dict):
            if task == 'in_count':
                return 'in' in self.tasks.get('counting', [])
            if task == 'out_count':
                return 'out' in self.tasks.get('counting', [])
            return task in self.tasks.get('ppe', [])
        return task in (self.tasks or [])

    def update_cfg(self, cfg: dict):
        for k, v in cfg.items():
            setattr(self, k, v)
        # update object classes if provided
        if "object_classes" in cfg:
            self.classes = cfg["object_classes"]
        if "count_classes" in cfg:
            self.count_classes = cfg["count_classes"]
        if "ppe_classes" in cfg:
            self.ppe_classes = cfg["ppe_classes"]
        if "tasks" in cfg:
            self.tasks = cfg["tasks"]
        if "type" in cfg:
            self.src_type = cfg["type"]
        if "alert_anomalies" in cfg:
            self.alert_anomalies = cfg["alert_anomalies"]
        if "line_orientation" in cfg:
            self.line_orientation = cfg["line_orientation"]
        if "reverse" in cfg:
            self.reverse = bool(cfg["reverse"])
        if "resolution" in cfg:
            self.resolution = cfg["resolution"]
        if "helmet_conf_thresh" in cfg:
            self.helmet_conf_thresh = cfg["helmet_conf_thresh"]
        if "detect_helmet_color" in cfg:
            self.detect_helmet_color = cfg["detect_helmet_color"]
        if "track_misc" in cfg:
            self.track_misc = cfg["track_misc"]
        if "show_lines" in cfg:
            self.show_lines = cfg["show_lines"]
        if "show_ids" in cfg:
            self.show_ids = cfg["show_ids"]
        if "show_track_lines" in cfg:
            self.show_track_lines = cfg["show_track_lines"]
        if "duplicate_filter_enabled" in cfg:
            self.duplicate_filter_enabled = cfg["duplicate_filter_enabled"]
            self.dup_filter = DuplicateFilter(self.duplicate_filter_threshold, self.duplicate_bypass_seconds) if self.duplicate_filter_enabled else None
        if "duplicate_filter_threshold" in cfg:
            self.duplicate_filter_threshold = cfg["duplicate_filter_threshold"]
            if self.dup_filter:
                self.dup_filter.threshold = self.duplicate_filter_threshold
        if "duplicate_bypass_seconds" in cfg:
            self.duplicate_bypass_seconds = cfg["duplicate_bypass_seconds"]
            if self.dup_filter:
                self.dup_filter.bypass_seconds = self.duplicate_bypass_seconds
        if "person_model" in cfg and cfg["person_model"] != getattr(self, "person_model", None):
            self.person_model = cfg["person_model"]
            self.model_person = YOLO(self.person_model)
            if self.device.startswith("cuda"):
                self.model_person.model.to(self.device).half()
        if "email" in cfg:
            self.email_cfg = cfg["email"]


    def _open_capture(self):
        """Return a capture object or FFmpeg process stdout for RTSP streams."""
        if self.src_type == "rtsp" and self.duplicate_filter_enabled:
            import subprocess, shlex
            res_map = {"480p": "640x480", "720p": "1280x720", "1080p": "1920x1080"}
            size = res_map.get(self.resolution)
            cmd = [
                "ffmpeg",
                "-rtsp_transport",
                "tcp",
                "-i",
                self.src,
                "-vf",
                "mpdecimate,setpts=N/FRAME_RATE/TB",
            ]
            if size:
                cmd += ["-s", size]
            cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
            logger.info("Starting ffmpeg: %s", " ".join(shlex.quote(c) for c in cmd))
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10 ** 8,
            )
            return proc
        if self.src_type == "rtsp":
            cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif self.src_type == "local":
            try:
                index = int(self.src)
            except ValueError:
                index = self.src
            cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(self.src)
        if self.resolution != "original":
            res_map = {
                "480p": (640, 480),
                "720p": (1280, 720),
                "1080p": (1920, 1080),
            }
            if self.resolution in res_map:
                w, h = res_map[self.resolution]
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        return cap

    def capture_loop(self):
        failures = 0
        while self.running:
            try:
                cap = self._open_capture()
                using_ffmpeg = hasattr(cap, "stdout")
                if not using_ffmpeg and not cap.isOpened():
                    raise RuntimeError("open_failed")
                # reset tracker state on new connection to avoid ID reuse
                self.tracker = DeepSort(max_age=5)
                self.tracks.clear()
                self.online = True
                failures = 0
                logger.info(f"Stream opened: {self.src}")
                width = height = None
                if using_ffmpeg:
                    if self.resolution != "original":
                        res_map = {"480p": (640,480), "720p": (1280,720), "1080p": (1920,1080)}
                        width, height = res_map.get(self.resolution, (None, None))
                    else:
                        probe = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                        if probe.isOpened():
                            ret, fr = probe.read()
                            if ret:
                                height, width = fr.shape[:2]
                        probe.release()
                while self.running:
                    try:
                        if using_ffmpeg:
                            if width is None or height is None:
                                width = 640
                                height = 480
                            raw = cap.stdout.read(width * height * 3)
                            if len(raw) < width * height * 3:
                                raise RuntimeError("ffmpeg_eof")
                            import numpy as np
                            frame = np.frombuffer(raw, dtype="uint8").reshape(height, width, 3)
                        else:
                            ret, frame = cap.read()
                            if not ret:
                                raise RuntimeError("read_failed")
                    except Exception as e:
                        logger.error(f"Stream read error: {e}")
                        ret = False
                    if not using_ffmpeg and not ret:
                        logger.warning(f"Lost stream, retry in {self.retry_interval}s")
                        failures += 1
                        if failures >= self.max_retry:
                            logger.error(f"Max retries reached for {self.src}. stopping tracker")
                            self.running = False
                            break
                        break
                    if self.frame_queue.full():
                        _ = self.frame_queue.get()
                    self.frame_queue.put(frame)
                if using_ffmpeg:
                    cap.kill()
                else:
                    cap.release()
            except (ConnectionResetError, OSError) as e:
                self.online = False
                if isinstance(e, ConnectionResetError) or getattr(e, 'winerror', None) == 10054:
                    logger.warning(f"Connection reset for {self.src}")
                else:
                    logger.error(f"Cannot open stream: {self.src} ({e})")
                failures += 1
            if failures >= self.max_retry:
                logger.error(f"Max retries reached for {self.src}. stopping tracker")
                self.running = False
            if self.running:
                time.sleep(self.retry_interval)

    def process_loop(self):
        idx = 0
        while self.running or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            if frame is None:
                continue
            if self.dup_filter and self.dup_filter.is_duplicate(frame):
                continue
            idx += 1
            if date.today() != self.prev_date:
                self.in_count = 0
                self.out_count = 0
                self.tracks.clear()
                self.prev_date = date.today()
                init_data = {self.key_date: self.prev_date.isoformat()}
                for g in self.groups:
                    self.in_counts[g] = 0
                    self.out_counts[g] = 0
                    init_data[f"{self.key_in}_{g}"] = 0
                    init_data[f"{self.key_out}_{g}"] = 0
                self.in_count = 0
                self.out_count = 0
                self.redis.mset(init_data)
                for item in ANOMALY_ITEMS:
                    self.redis.mset({f'{item}_count': 0, f'{item}_date': self.prev_date.isoformat()})
                logger.info("Daily counts reset")
            if self.skip_frames and idx % self.skip_frames:
                continue
            res = self.model_person.predict(frame, device=self.device, verbose=False)[0]
            h, w = frame.shape[:2]
            if self.line_orientation == 'horizontal':
                line_pos = int(h * self.line_ratio)
                if self.show_lines:
                    cv2.line(frame, (0, line_pos), (w, line_pos), (255, 0, 0), 2)
            else:
                line_pos = int(w * self.line_ratio)
                if self.show_lines:
                    cv2.line(frame, (line_pos, 0), (line_pos, h), (255, 0, 0), 2)
            dets = []
            for *xyxy, conf, cls in res.boxes.data.tolist():
                raw = self.model_person.names[int(cls)] if isinstance(self.model_person.names, dict) else self.model_person.names[int(cls)]
                label = self._clean_label(raw)
                if label in self.classes and conf >= self.conf_thresh:
                    dets.append([
                        [int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])],
                        conf,
                        label,
                    ])
            try:
                tracks = self.tracker.update_tracks(dets, frame=frame)
            except ValueError as e:
                logger.warning(f"tracker update error: {e}")
                continue
            now = time.time()
            active_ids = set()
            for tr in tracks:
                if not tr.is_confirmed():
                    continue
                tid = tr.track_id
                active_ids.add(tid)
                x1, y1, x2, y2 = map(int, tr.to_ltrb())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                if self.line_orientation == 'horizontal':
                    zone = 'top' if cy < line_pos else 'bottom'
                else:
                    zone = 'left' if cx < line_pos else 'right'
                label = getattr(tr, 'det_class', None)
                if tid not in self.tracks:
                    self.tracks[tid] = {
                        'zone': zone,
                        'cx': cx,
                        'time': now,
                        'last': None,
                        'alerted': False,
                        'label': label,
                        'best_conf': 0.0,
                        'best_img': None,
                        'images': [],
                        'first_zone': zone,
                        'trail': [(cx, cy)],
                    }
                prev = self.tracks[tid]
                conf = getattr(tr, 'det_conf', 0) or 0
                if conf > prev.get('best_conf', 0):
                    crop = frame[y1:y2, x1:x2]
                    if crop.size:
                        prev['best_conf'] = conf
                        prev['best_img'] = crop.copy()
                if conf >= 0.5:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size:
                        imgs = prev.setdefault('images', [])
                        if len(imgs) < 20:
                            imgs.append((conf, crop.copy()))
                if label is not None:
                    prev['label'] = label
                if zone != prev['zone'] and abs(cx - prev['cx']) > self.v_thresh and now - prev['time'] > self.debounce:
                    direction = None
                    if self.line_orientation == 'horizontal':
                        if prev['zone'] == 'top' and zone == 'bottom':
                            direction = 'Entering'
                        elif prev['zone'] == 'bottom' and zone == 'top':
                            direction = 'Exiting'
                    else:
                        if prev['zone'] == 'left' and zone == 'right':
                            direction = 'Entering'
                        elif prev['zone'] == 'right' and zone == 'left':
                            direction = 'Exiting'
                    if self.reverse and direction:
                        direction = 'Exiting' if direction == 'Entering' else 'Entering'
                    lbl = prev.get('label')
                    grp = None
                    for g, labels in COUNT_GROUPS.items():
                        if lbl in labels:
                            grp = g
                            break
                    if direction and grp in self.groups and lbl in self.count_classes:
                        allowed = True
                        if direction == 'Entering' and not self.task_enabled('in_count') and not self.task_enabled('full_monitor'):
                            allowed = False
                        if direction == 'Exiting' and not self.task_enabled('out_count') and not self.task_enabled('full_monitor'):
                            allowed = False
                        if allowed:
                            if prev['last'] is None:
                                if direction == 'Entering':
                                    self.in_counts[grp] += 1
                                    self.in_count += 1
                                else:
                                    self.out_counts[grp] += 1
                                    self.out_count += 1
                                self.redis.mset({
                                    f"{self.key_in}_{grp}": self.in_counts[grp],
                                    f"{self.key_out}_{grp}": self.out_counts[grp]
                                })
                                if self.update_callback:
                                    self.update_callback()
                                prev['last'] = direction
                                prev['direction'] = direction
                                prev['cross_time'] = now
                                logger.info(
                                    f"{direction} ID{tid} ({grp}) In={self.in_counts[grp]} Out={self.out_counts[grp]}"
                                )
                            elif prev['last'] != direction:
                                if prev['last'] == 'Entering':
                                    self.in_counts[grp] = max(0, self.in_counts[grp]-1)
                                    self.in_count = max(0, self.in_count-1)
                                else:
                                    self.out_counts[grp] = max(0, self.out_counts[grp]-1)
                                    self.out_count = max(0, self.out_count-1)
                                self.redis.mset({
                                    f"{self.key_in}_{grp}": self.in_counts[grp],
                                    f"{self.key_out}_{grp}": self.out_counts[grp]
                                })
                                if self.update_callback:
                                    self.update_callback()
                                prev['last'] = None
                                prev['direction'] = None
                                prev['cross_time'] = None
                                logger.info(f"Reversed flow for ID{tid}")
                            prev['time'] = now
                prev['zone'], prev['cx'] = zone, cx
                prev['last_seen'] = now
                trail = prev.setdefault('trail', [])
                trail.append((cx, cy))
                if len(trail) > 20:
                    trail.pop(0)
                color = (0, 255, 0) if zone == 'right' else (0, 0, 255)
                if self.show_track_lines:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if self.show_track_lines and len(trail) > 1:
                    for i in range(1, len(trail)):
                        cv2.line(frame, trail[i-1], trail[i], (0,0,255), 2)
                if self.show_ids:
                    cv2.putText(frame, f"ID{tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # process tracks that have disappeared
            gone_ids = [tid for tid in list(self.tracks.keys()) if tid not in active_ids]
            for tid in gone_ids:
                info = self.tracks.pop(tid)
                first_zone = info.get('first_zone')
                last_zone = info.get('zone')
                images = []
                best_img = info.get('best_img')
                if best_img is not None and best_img.size:
                    images.append(best_img)
                import random
                candidates = [img for c, img in info.get('images', []) if c >= 0.5 and img is not best_img]
                random.shuffle(candidates)
                images.extend(candidates[:4])
                label = info.get('label')
                if label in COUNT_GROUPS.get('vehicle', []):
                    ts = int(time.time())
                    snap = best_img if best_img is not None else frame
                    fname = f"{self.cam_id}_{tid}_vehicle_{ts}.jpg"
                    path = self.snap_dir / fname
                    cv2.imwrite(str(path), snap)
                    entry = {
                        'ts': ts,
                        'cam_id': self.cam_id,
                        'track_id': tid,
                        'label': 'vehicle',
                        'path': str(path),
                    }
                    self.redis.zadd('vehicle_logs', {json.dumps(entry): ts})
                    limit = self.cfg.get('ppe_log_limit', 1000)
                    self.redis.zremrangebyrank('vehicle_logs', 0, -limit-1)
                    continue
                # fallback count using ROI zones when no crossing detected
                if info.get('last') is None and first_zone and last_zone and first_zone != last_zone and info.get('label') in self.count_classes:
                    direction = None
                    if self.line_orientation == 'horizontal':
                        if first_zone == 'top' and last_zone == 'bottom':
                            direction = 'Entering'
                        elif first_zone == 'bottom' and last_zone == 'top':
                            direction = 'Exiting'
                    else:
                        if first_zone == 'left' and last_zone == 'right':
                            direction = 'Entering'
                        elif first_zone == 'right' and last_zone == 'left':
                            direction = 'Exiting'
                    if self.reverse and direction:
                        direction = 'Exiting' if direction == 'Entering' else 'Entering'
                    if direction:
                        allowed = True
                        if direction == 'Entering' and not self.task_enabled('in_count') and not self.task_enabled('full_monitor'):
                            allowed = False
                        if direction == 'Exiting' and not self.task_enabled('out_count') and not self.task_enabled('full_monitor'):
                            allowed = False
                    if direction and allowed:
                        grp = None
                        lbl = info.get('label')
                        for g, labels in COUNT_GROUPS.items():
                            if lbl in labels:
                                grp = g
                                break
                        if grp in self.groups:
                            if direction == 'Entering':
                                self.in_counts[grp] += 1
                                self.in_count += 1
                            else:
                                self.out_counts[grp] += 1
                                self.out_count += 1
                            self.redis.mset({
                                f"{self.key_in}_{grp}": self.in_counts[grp],
                                f"{self.key_out}_{grp}": self.out_counts[grp]
                            })
                            if self.update_callback:
                                self.update_callback()
                            logger.info(
                                f"ROI {direction} ID{tid} ({grp}) In={self.in_counts[grp]} Out={self.out_counts[grp]}"
                            )
                            info['direction'] = direction
                            info['cross_time'] = now


                if label in COUNT_GROUPS.get('person', []):
                    ct = info.get('cross_time')
                    cross_ts = int(ct) if ct is not None else int(time.time())
                    direction = info.get('direction')
                    if direction:
                        snap = best_img if best_img is not None else frame
                        fname = f"{self.cam_id}_{tid}_{direction.lower()}_{cross_ts}.jpg"
                        path = self.snap_dir / fname
                        cv2.imwrite(str(path), snap)
                        entry = {
                            'ts': cross_ts,
                            'cam_id': self.cam_id,
                            'track_id': tid,
                            'direction': direction,
                            'path': str(path),
                            'needs_ppe': bool(self.tasks.get('ppe')),
                            'ppe_tasks': self.tasks.get('ppe', [])
                        }
                        self.redis.zadd('person_logs', {json.dumps(entry): cross_ts})
                        limit = self.cfg.get('ppe_log_limit', 1000)
                        self.redis.zremrangebyrank('person_logs', 0, -limit-1)


            cv2.putText(frame, f"Entering: {self.in_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Exiting: {self.out_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            with lock:
                self.output_frame = frame.copy()
            time.sleep(1 / self.fps)


