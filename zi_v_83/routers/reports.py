"""Count report routes."""
from __future__ import annotations
from typing import Dict
import os
from pathlib import Path
from datetime import datetime
import io
import csv
import json

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from modules.utils import require_roles
from config import config

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent

def init_context(config: dict, trackers: Dict[int, "PersonTracker"], redis_client, templates_path):
    global cfg, trackers_map, redis, templates
    cfg = config
    trackers_map = trackers
    redis = redis_client
    templates = Jinja2Templates(directory=templates_path)

@router.get('/report')
async def report_page(request: Request):
    res = require_roles(request, ['admin'])
    if isinstance(res, RedirectResponse):
        return res
    return templates.TemplateResponse('report.html', {
        'request': request,
        'vehicle_enabled': 'vehicle' in cfg.get('track_objects', []),
        'cfg': config,
    })

@router.get('/report_data')
async def report_data(start: str, end: str, type: str = 'person', view: str = 'graph', rows: int = 50, request: Request = None):
    res = require_roles(request, ['admin'])
    if isinstance(res, RedirectResponse):
        return res
    try:
        start_ts = int(datetime.fromisoformat(start).timestamp())
        end_ts = int(datetime.fromisoformat(end).timestamp())
    except Exception:
        return {"error": "invalid range"}
    if view == 'graph':
        entries = [json.loads(e) for e in redis.zrangebyscore('history', start_ts, end_ts)]
        times, ins, outs, currents = [], [], [], []
        key_in = f'in_{type}'
        key_out = f'out_{type}'
        for entry in entries:
            ts = entry.get('ts')
            times.append(datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M'))
            i = entry.get(key_in, 0)
            o = entry.get(key_out, 0)
            ins.append(i)
            outs.append(o)
            currents.append(i - o)
        return {'times': times, 'ins': ins, 'outs': outs, 'current': currents}
    else:
        key = 'person_logs' if type == 'person' else 'vehicle_logs'
        entries = [json.loads(e) for e in redis.zrevrangebyscore(key, end_ts, start_ts, start=0, num=rows)]
        rows_out = []
        for e in entries:
            ts = e.get('ts')
            img_url = None
            path = e.get('path')
            if path:
                img_url = f"/snapshots/{os.path.basename(path)}"
            row = {
                'time': datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M'),
                'cam_id': e.get('cam_id'),
                'track_id': e.get('track_id'),
                'direction': e.get('direction'),
                'path': img_url,
                'label': e.get('label'),
            }
            rows_out.append(row)
        return {'rows': rows_out}

@router.get('/report/export')
async def report_export(start: str, end: str, type: str = 'person', view: str = 'graph', rows: int = 50, request: Request = None):
    res = require_roles(request, ['admin'])
    if isinstance(res, RedirectResponse):
        return res
    data = await report_data(start, end, type, view, rows, request)
    if 'error' in data:
        return JSONResponse(data, status_code=400)
    if view == 'graph':
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Time', 'In', 'Out', 'Current'])
        for row in zip(data['times'], data['ins'], data['outs'], data['current']):
            writer.writerow(row)
        return StreamingResponse(io.BytesIO(output.getvalue().encode()), media_type='text/csv', headers={'Content-Disposition': 'attachment; filename=report.csv'})
    else:
        from openpyxl import Workbook
        from openpyxl.drawing.image import Image as XLImage
        wb = Workbook()
        ws = wb.active
        ws.append(['Time', 'Camera', 'Track', 'Direction', 'Label', 'Image'])
        for row in data['rows']:
            ws.append([row['time'], row['cam_id'], row['track_id'], row.get('direction') or '', row.get('label') or '', ''])
            img_path = row.get('path')
            if img_path:
                img_file = os.path.join(BASE_DIR, img_path.lstrip('/'))
                if os.path.exists(img_file):
                    img = XLImage(img_file)
                    img.width = 80
                    img.height = 60
                    ws.add_image(img, f'F{ws.max_row}')
        bio = io.BytesIO()
        wb.save(bio)
        bio.seek(0)
        return StreamingResponse(bio, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', headers={'Content-Disposition': 'attachment; filename=report.xlsx'})
