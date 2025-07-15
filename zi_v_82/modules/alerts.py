from __future__ import annotations
import json
import io
import time
import threading
from loguru import logger
from .utils import send_email
import redis
from datetime import datetime, timedelta
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from pathlib import Path

class AlertWorker:
    def __init__(self, cfg: dict, redis_url: str, base_dir: Path):
        self.cfg = cfg
        self.redis = redis.Redis.from_url(redis_url)
        self.base_dir = base_dir
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)

    def loop(self):
        while self.running:
            try:
                self.check_rules()
            except Exception as exc:
                logger.error("alert loop error: %s", exc)
            time.sleep(60)

    def _collect_rows(self, start_ts: int, end_ts: int, metric: str):
        """Return log rows matching the metric within the time window."""
        entries = self.redis.lrange("ppe_logs", 0, -1)
        rows = []
        for item in entries:
            e = json.loads(item)
            ts = e.get("ts")
            if ts is None or ts <= start_ts or ts > end_ts:
                continue
            if metric and e.get("status") != metric:
                continue
            rows.append(e)
        rows.sort(key=lambda r: r["ts"])  # ensure chronological order
        return rows

    def _send_report(self, rows, recipients, subject, attach=True):
        wb = Workbook()
        ws = wb.active
        ws.append(['Time','Camera','Track','Status','Conf','Color'])
        for r in rows:
            ws.append([
                datetime.fromtimestamp(r['ts']).strftime('%Y-%m-%d %H:%M'),
                r.get('cam_id'),
                r.get('track_id'),
                r.get('status'),
                round(r.get('conf',0),2),
                r.get('color') or ''
            ])
            path = r.get('path')
            if path and Path(path).exists():
                img = XLImage(path)
                img.width = 80
                img.height = 60
                ws.add_image(img, f'F{ws.max_row}')
        bio = io.BytesIO()
        wb.save(bio)
        bio.seek(0)
        attachment = bio.getvalue() if attach else None
        send_email(
            subject,
            'See attached report' if attach else 'Alert',
            recipients,
            None,
            self.cfg.get('email', {}),
            attachment=attachment,
            attachment_name='report.xlsx' if attach else None,
            attachment_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' if attach else None
        )

    def check_rules(self):
        if not self.cfg.get('email_enabled', True):
            return
        rules = self.cfg.get('alert_rules', [])
        if not rules:
            return
        now = int(time.time())
        for i, rule in enumerate(rules):
            metric = rule.get('metric')
            rtype = rule.get('type', 'event')
            value = int(rule.get('value', 1))
            attach = rule.get('attach', True)
            recipients = [a.strip() for a in rule.get('recipients', '').split(',') if a.strip()]
            if not metric or not recipients:
                continue
            last_key = f'alert_rule_{i}_last'
            last_ts = int(float(self.redis.get(last_key) or 0))
            if rtype == 'frequency':
                interval = value * 60
                if now - last_ts >= interval:
                    rows = self._collect_rows(last_ts, now, metric)
                    if rows:
                        self._send_report(rows, recipients, f'Alert: {metric}', attach)
                    self.redis.set(last_key, now)
                continue

            rows = self._collect_rows(last_ts, now, metric)
            if not rows:
                continue
            if rtype == 'event':
                if len(rows) >= value:
                    send_rows = rows[:value]
                    self._send_report(send_rows, recipients, f'Alert: {metric}', attach)
                    self.redis.set(last_key, send_rows[-1]['ts'])
            elif rtype == 'threshold':
                prev_key = f'alert_rule_{i}_pending'
                pending = int(self.redis.get(prev_key) or 0) + len(rows)
                if pending >= value:
                    send_rows = rows[:value]
                    self._send_report(send_rows, recipients, f'Alert: {metric}', attach)
                    pending -= value
                    self.redis.set(last_key, send_rows[-1]['ts'])
                self.redis.set(prev_key, pending)
