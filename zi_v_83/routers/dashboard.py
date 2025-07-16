"""Dashboard and websocket routes."""
from __future__ import annotations
import os
import json
from datetime import datetime
from typing import Dict
from pathlib import Path

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import cv2
import asyncio

from core.tracker_manager import handle_status_change
from modules.utils import lock, require_roles
from core.config import ANOMALY_ITEMS
from config import config

router = APIRouter()

def init_context(config: dict, trackers: Dict[int, "PersonTracker"], cameras: list, redis_client):
    global cfg, trackers_map, cams, templates, redis
    cfg = config
    trackers_map = trackers
    cams = cameras
    redis = redis_client
    templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / 'templates'))

@router.get('/')
async def index(request: Request):
    res = require_roles(request, ['admin','viewer'])
    if isinstance(res, RedirectResponse):
        return res
    groups = cfg.get('track_objects', ['person'])
    group_counts = {}
    for g in groups:
        in_g = sum(t.in_counts.get(g, 0) for t in trackers_map.values())
        out_g = sum(t.out_counts.get(g, 0) for t in trackers_map.values())
        group_counts[g] = {'in': in_g, 'out': out_g, 'current': in_g - out_g}
    current = group_counts.get('person', {'current': 0})['current']
    in_c = group_counts.get('person', {'in': 0})['in']
    out_c = group_counts.get('person', {'out': 0})['out']
    max_cap = cfg['max_capacity']
    warn_lim = max_cap * cfg['warn_threshold'] / 100
    status = 'green' if current < warn_lim else 'yellow' if current < max_cap else 'red'
    active = [c for c in cams if c.get('show', True)]
    anomaly_counts = {item: int(redis.get(f'{item}_count') or 0) for item in ANOMALY_ITEMS}
    return templates.TemplateResponse('dashboard.html', {
        'request': request,
        'max_capacity': max_cap,
        'status': status,
        'current': current,
        'cameras': active,
        'cfg': config,
        'anomaly_counts': anomaly_counts,
        'group_counts': group_counts,
    })

@router.get('/video_feed/{cam_id}')
async def video_feed(cam_id: int, request: Request):
    res = require_roles(request, ['admin','viewer'])
    if isinstance(res, RedirectResponse):
        return res
    tr = trackers_map.get(cam_id)
    if not tr:
        return HTMLResponse('Not found', status_code=404)

    async def gen():
        while True:
            with lock:
                frame = tr.output_frame
            if frame is None:
                await asyncio.sleep(0.1)
                continue
            _, buf = cv2.imencode('.jpg', frame)
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
            await asyncio.sleep(1 / tr.fps)
    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')

@router.websocket('/ws/stats')
async def ws_stats(ws: WebSocket):
    """WebSocket endpoint that streams stats using Redis Streams."""
    await ws.accept()
    from core.stats import gather_stats
    await ws.send_json(gather_stats(trackers_map, redis))
    last_id = '$'
    try:
        while True:
            res = await asyncio.to_thread(
                redis.xread,
                {'people_counter': last_id},
                block=10000,
                count=1,
            )
            if res:
                _, messages = res[0]
                for sid, fields in messages:
                    last_id = sid.decode() if isinstance(sid, bytes) else sid
                    data = fields.get(b'data') or fields.get('data')
                    if data:
                        await ws.send_text(data.decode() if isinstance(data, bytes) else data)
    except WebSocketDisconnect:
        pass

@router.get('/sse/stats')
async def sse_stats():
    async def event_gen():
        last_id = '$'
        try:
            while True:
                res = await asyncio.to_thread(
                    redis.xread, {'people_counter': last_id}, block=10000, count=1
                )
                if res:
                    _, messages = res[0]
                    for sid, fields in messages:
                        last_id = sid.decode() if isinstance(sid, bytes) else sid
                        data = fields.get(b'data') or fields.get('data')
                        if data:
                            val = data.decode() if isinstance(data, bytes) else data
                            yield f"data: {val}\n\n"
                await asyncio.sleep(0)
        finally:
            pass
    return StreamingResponse(event_gen(), media_type='text/event-stream')

@router.get('/latest_images')
async def latest_images(status: str = 'no_helmet', count: int = 5):
    """Return paths of the most recent snapshots for the given status."""
    entries = redis.zrevrange('ppe_logs', 0, 1000)
    imgs: list[str] = []
    for item in entries:
        e = json.loads(item)
        if e.get('status') == status and e.get('path'):
            fname = os.path.basename(e['path'])
            imgs.append(f'/snapshots/{fname}')
            if len(imgs) >= count:
                break
    return {'images': imgs}
