"""Crowd management system version 78."""
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
import asyncio

from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
import uvicorn
import redis
import cv2
import torch
import sys

# allow imports relative to this version directory without hardcoding its name
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# log to file with rotation
logger.add(str(Path(__file__).resolve().parent / "app.log"), rotation="1 MB", retention=5)



from core.config import load_config, save_config
from core.tracker_manager import (
    load_cameras,
    start_tracker,
    count_log_loop,
)
from modules.person_tracker import PersonTracker
from modules.alerts import AlertWorker
from modules.ppe_worker import PPEDetector
from modules.utils import lock, SNAP_DIR
from config import set_config, config as shared_config

# Globals
BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI()
app.mount("/snapshots", StaticFiles(directory=str(SNAP_DIR)), name="snapshots")
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Context holders
config: dict
config_path: str
redis_client: redis.Redis
cameras: list
a = []
trackers: dict[int, PersonTracker] = {}
ppe_worker: PPEDetector | None = None
alert_worker: AlertWorker | None = None

# Routers
from routers import dashboard, settings, cameras as cam_routes, reports, ppe_reports, alerts, auth

log_task = None

@app.on_event("startup")
async def on_startup_event():
    global log_task
    if 'redis_client' in globals():
        log_task = asyncio.create_task(count_log_loop(redis_client, trackers))

@app.on_event("shutdown")
async def on_shutdown_event():
    global log_task
    if log_task:
        log_task.cancel()


def init_app():
    global config, config_path, redis_client, cameras, alert_worker
    parser = argparse.ArgumentParser()
    parser.add_argument("stream_url", nargs="?")
    parser.add_argument("-c", "--config", default="config.json")
    parser.add_argument("-w", "--workers", type=int, default=None)
    args = parser.parse_args()

    config_path_local = args.config if os.path.isabs(args.config) else str(BASE_DIR / args.config)
    temp_cfg = json.load(open(config_path_local))
    redis_client_local = redis.Redis.from_url(temp_cfg.get("redis_url", "redis://localhost:6379/0"))
    cfg = load_config(config_path_local, redis_client_local)
    set_config(cfg)
    cams = load_cameras(redis_client_local, cfg["stream_url"])
    app.add_middleware(SessionMiddleware, secret_key=cfg["secret_key"])
    if args.stream_url:
        cams = [{"id": 1, "name": "CameraCLI", "url": args.stream_url, "tasks": ["both"], "enabled": True}]
    # start trackers
    for cam in cams:
        if cam.get("enabled", True):
            start_tracker(cam, cfg, trackers, redis_client_local)
    alert_worker = AlertWorker(cfg, cfg["redis_url"], BASE_DIR)
    from core.stats import broadcast_stats
    ppe_worker = PPEDetector(cfg, cfg["redis_url"], SNAP_DIR,
                             lambda: broadcast_stats(trackers, redis_client_local))
    ppe_worker.start()

    # set globals
    globals().update(config=cfg, config_path=config_path_local, redis_client=redis_client_local, cameras=cams, ppe_worker=ppe_worker)

    # routers init
    dashboard.init_context(cfg, trackers, cams, redis_client_local)
    settings.init_context(cfg, trackers, cams, redis_client_local, str(TEMPLATE_DIR), config_path_local)
    cam_routes.init_context(cfg, cams, trackers, redis_client_local, str(TEMPLATE_DIR))
    reports.init_context(cfg, trackers, redis_client_local, str(TEMPLATE_DIR))
    ppe_reports.init_context(cfg, trackers, redis_client_local, str(TEMPLATE_DIR))
    alerts.init_context(cfg, trackers, redis_client_local, str(TEMPLATE_DIR), config_path_local)
    auth.init_context(cfg, str(TEMPLATE_DIR))

    app.include_router(dashboard.router)
    app.include_router(settings.router)
    app.include_router(cam_routes.router)
    app.include_router(reports.router)
    app.include_router(ppe_reports.router)
    app.include_router(alerts.router)
    app.include_router(auth.router)

    cores = os.cpu_count() or 1
    workers = args.workers if args.workers is not None else cfg["default_workers"]
    w = max((cores - 1 if workers == -1 else (1 if workers == 0 else workers)), 1)
    cv2.setNumThreads(w)
    torch.set_num_threads(w)
    logger.info(f"Threads={w}, cores={cores}")

    redis_client_local  # store to avoid flake

    return cfg


def main():
    cfg = init_app()
    logger.info(f"Server http://0.0.0.0:{cfg['port']}")
    uvicorn.run(app, host="0.0.0.0", port=cfg["port"], log_config=None)

if __name__ == "__main__":
    main()
