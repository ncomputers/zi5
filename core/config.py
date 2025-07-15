"""Configuration loading and saving utilities."""
from __future__ import annotations
import json
import os
import redis
from typing import Any

MODEL_CLASSES = [
    "dust_mask",
    "face_shield",
    "helmet",
    "protective_gloves",
    "safety_glasses",
    "safety_shoes",
    "vest_jacket",
    "head",
]
PPE_ITEMS = [
    "helmet",
    "safety_shoes",
    "safety_glasses",
    "protective_gloves",
    "dust_mask",
    "face_shield",
    "vest_jacket",
]
ANOMALY_ITEMS = [
    "no_helmet",
    "no_safety_shoes",
    "no_safety_glasses",
    "no_protective_gloves",
    "no_dust_mask",
    "no_face_shield",
    "no_vest_jacket",
    "yellow_alert",
    "red_alert",
]
COUNT_GROUPS = {
    "person": ["person"],
    "vehicle": ["car", "truck", "bus", "motorcycle", "bicycle"],
}
AVAILABLE_CLASSES = (
    MODEL_CLASSES + ANOMALY_ITEMS + [c for cl in COUNT_GROUPS.values() for c in cl]
)
# Camera task options. Counting directions are now defined as tasks and
# can be combined with PPE detection tasks.
CAMERA_TASKS = ["in_count", "out_count", "full_monitor"] + MODEL_CLASSES


def sync_detection_classes(cfg: dict) -> None:
    object_classes: list[str] = []
    count_classes: list[str] = []
    for group in cfg.get("track_objects", ["person"]):
        count_classes.extend(COUNT_GROUPS.get(group, [group]))
    object_classes.extend(count_classes)
    ppe_classes: list[str] = []
    for item in cfg.get("track_ppe", []):
        if item in MODEL_CLASSES:
            ppe_classes.append(item)
        neg = f"no_{item}"
        if neg in AVAILABLE_CLASSES:
            ppe_classes.append(neg)
    cfg["object_classes"] = object_classes
    cfg["ppe_classes"] = ppe_classes
    cfg["count_classes"] = count_classes


def load_config(path: str, r: redis.Redis) -> dict:
    if os.path.exists(path):
        data = json.load(open(path))
        data.setdefault("track_ppe", [])
        data.setdefault("alert_anomalies", [])
        data.setdefault("track_objects", ["person"])
        data.setdefault("helmet_conf_thresh", 0.5)
        data.setdefault("detect_helmet_color", False)
        data.setdefault("show_lines", True)
        data.setdefault("show_ids", True)
        data.setdefault("preview_anomalies", [])
        data.setdefault("email_enabled", True)
        data.setdefault("show_track_lines", False)
        data.setdefault("duplicate_filter_enabled", False)
        data.setdefault("duplicate_filter_threshold", 0.1)
        data.setdefault("duplicate_bypass_seconds", 2)
        data.setdefault("ppe_log_limit", 1000)
        data.setdefault("max_retry", 5)
        data.setdefault("person_model", "yolov8n.pt")
        data.setdefault("ppe_model", "mymodalv5.pt")
        data.setdefault("users", [
            {"username": "admin", "password": "rapidadmin", "role": "admin"},
            {"username": "viewer", "password": "viewer", "role": "viewer"}
        ])
        data.setdefault("settings_password", "000")
        sync_detection_classes(data)
        r.set("config", json.dumps(data))
        return data
    raise FileNotFoundError(path)


def save_config(cfg: dict, path: str, r: redis.Redis) -> None:
    sync_detection_classes(cfg)
    def _ser(o: Any):
        from pathlib import Path
        if isinstance(o, Path):
            return str(o)
        raise TypeError(str(o))
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, default=_ser)
    r.set("config", json.dumps(cfg, default=_ser))
