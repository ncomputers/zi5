"""Camera management routes."""
from __future__ import annotations
from typing import Dict, List

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from modules.utils import require_roles
import cv2

from core.tracker_manager import start_tracker, stop_tracker, save_cameras
from core.config import CAMERA_TASKS
from config import config

router = APIRouter()

def init_context(config: dict, cameras: List[dict], trackers: Dict[int, "PersonTracker"], redis_client, templates_path):
    global cfg, cams, trackers_map, redis, templates
    cfg = config
    cams = cameras
    trackers_map = trackers
    redis = redis_client
    templates = Jinja2Templates(directory=templates_path)

@router.get('/cameras')
async def cameras_page(request: Request):
    res = require_roles(request, ['admin'])
    if isinstance(res, RedirectResponse):
        return res
    cam_list = []
    for c in cams:
        tr = trackers_map.get(c['id'])
        cam_copy = c.copy()
        cam_copy['online'] = tr.online if tr else False
        cam_list.append(cam_copy)
    return templates.TemplateResponse('cameras.html', {
        'request': request,
        'cams': cam_list,
        'model_classes': CAMERA_TASKS,
        'cfg': config,
    })

@router.post('/cameras')
async def add_camera(request: Request):
    res = require_roles(request, ['admin'])
    if isinstance(res, RedirectResponse):
        return res
    data = await request.json()
    name = data.get('name') or f"Camera{len(cams)+1}"
    url = data.get('url')
    src_type = data.get('type', 'http')
    tasks = data.get('tasks', {})
    reverse = bool(data.get('reverse'))
    orientation = data.get('line_orientation', 'vertical')
    resolution = data.get('resolution', 'original')
    if isinstance(tasks, list):
        cnt, ppe = [], []
        for t in tasks:
            if t == 'in_count':
                cnt.append('in')
            elif t == 'out_count':
                cnt.append('out')
            else:
                ppe.append(t)
        tasks = {'counting': cnt, 'ppe': ppe}
    if not isinstance(tasks, dict):
        tasks = {'counting': ['in', 'out'], 'ppe': []}
    if not url:
        return {'error': 'Missing URL'}
    cam_id = max([c['id'] for c in cams], default=0) + 1
    cam = {
        'id': cam_id,
        'name': name,
        'url': url,
        'type': src_type,
        'tasks': tasks,
        'enabled': True,
        'show': True,
        'reverse': reverse,
        'line_orientation': orientation,
        'resolution': resolution,
    }
    cams.append(cam)
    save_cameras(cams, redis)
    start_tracker(cam, cfg, trackers_map, redis)
    return {'added': True, 'camera': cam}

@router.delete('/cameras/{cam_id}')
async def delete_camera(cam_id: int, request: Request):
    res = require_roles(request, ['admin'])
    global cams
    remaining = [c for c in cams if c['id'] != cam_id]
    if len(remaining) == len(cams):
        return {'error': 'Not found'}
    cams[:] = remaining
    stop_tracker(cam_id, trackers_map)
    save_cameras(cams, redis)
    return {'deleted': True}

@router.patch('/cameras/{cam_id}/show')
async def toggle_show(cam_id: int, request: Request):
    res = require_roles(request, ['admin'])
    if isinstance(res, RedirectResponse):
        return res
    for cam in cams:
        if cam['id'] == cam_id:
            cam['show'] = not cam.get('show', True)
            save_cameras(cams, redis)
            return {'show': cam['show']}
    return {'error': 'Not found'}

@router.put('/cameras/{cam_id}')
async def update_camera(cam_id: int, request: Request):
    res = require_roles(request, ['admin'])
    if isinstance(res, RedirectResponse):
        return res
    data = await request.json()
    for cam in cams:
        if cam['id'] == cam_id:
            if 'tasks' in data:
                tasks_upd = data['tasks']
                if isinstance(tasks_upd, list):
                    cnt, ppe = [], []
                    for t in tasks_upd:
                        if t == 'in_count':
                            cnt.append('in')
                        elif t == 'out_count':
                            cnt.append('out')
                        else:
                            ppe.append(t)
                    tasks_upd = {'counting': cnt, 'ppe': ppe}
                if isinstance(tasks_upd, dict):
                    cam['tasks'] = tasks_upd
            if 'url' in data:
                cam['url'] = data['url']
            if 'type' in data:
                cam['type'] = data['type']
            if 'show' in data:
                cam['show'] = bool(data['show'])
            if 'reverse' in data:
                cam['reverse'] = bool(data['reverse'])
            if 'line_orientation' in data:
                cam['line_orientation'] = data['line_orientation']
            if 'resolution' in data:
                cam['resolution'] = data['resolution']
            save_cameras(cams, redis)
            tr = trackers_map.get(cam_id)
            if tr:
                tr.update_cfg({
                    'tasks': cam['tasks'],
                    'type': cam['type'],
                    'reverse': cam['reverse'],
                    'line_orientation': cam['line_orientation'],
                    'resolution': cam['resolution'],
                })
            return {'updated': True}
    return {'error': 'Not found'}

@router.get('/cameras/export')
async def export_cameras(request: Request):
    """Export camera list as JSON."""
    res = require_roles(request, ['admin'])
    if isinstance(res, RedirectResponse):
        return res
    from fastapi.responses import JSONResponse
    return JSONResponse(cams)

@router.post('/cameras/import')
async def import_cameras(request: Request):
    """Replace cameras with uploaded list."""
    res = require_roles(request, ['admin'])
    if isinstance(res, RedirectResponse):
        return res
    data = await request.json()
    if not isinstance(data, list):
        return {'error': 'invalid'}
    # stop existing trackers
    for cid in list(trackers_map.keys()):
        stop_tracker(cid, trackers_map)
    cams[:] = data
    save_cameras(cams, redis)
    for cam in cams:
        if cam.get('enabled', True):
            start_tracker(cam, cfg, trackers_map, redis)
    return {'imported': True}

@router.post('/cameras/test')
async def test_camera(request: Request):
    """Return a single frame for previewing a camera stream."""
    res = require_roles(request, ['admin'])
    if isinstance(res, RedirectResponse):
        return res
    data = await request.json()
    url = data.get('url')
    if not url:
        return {'error': 'missing url'}
    cap = cv2.VideoCapture(int(url) if url.isdigit() else url)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return {'error': 'unable to read'}
    _, buf = cv2.imencode('.jpg', frame)
    from fastapi.responses import Response
    return Response(content=buf.tobytes(), media_type='image/jpeg')
