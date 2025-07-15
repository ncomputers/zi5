import asyncio
import json
import time
from datetime import datetime
import sys
from pathlib import Path
import pytest
import fakeredis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault('cv2', type('cv2', (), {}))
sys.modules.setdefault('torch', type('torch', (), {'cuda': type('cuda', (), {'is_available': lambda: False})}))
sys.modules.setdefault('ultralytics', type('ultralytics', (), {'YOLO': object}))
sys.modules.setdefault('deep_sort_realtime', type('ds', (), {}))
sys.modules['deep_sort_realtime.deepsort_tracker'] = type('t', (), {'DeepSort': object})
sys.modules.setdefault('imagehash', type('imagehash', (), {}))

from routers import reports, ppe_reports

class DummyRequest:
    def __init__(self):
        self.session = {'user': {'role': 'admin'}}

@pytest.fixture
def redis_client():
    return fakeredis.FakeRedis()

@pytest.fixture
def cfg():
    return {
        'track_objects': ['person'],
        'helmet_conf_thresh': 0.5,
        'track_ppe': ['helmet'],
    }

@pytest.fixture(autouse=True)
def setup_context(redis_client, cfg, tmp_path):
    reports.init_context(cfg, {}, redis_client, str(tmp_path))
    ppe_reports.init_context(cfg, {}, redis_client, str(tmp_path))


def test_report_data_graph(redis_client):
    now = int(time.time())
    entry1 = {'ts': now-120, 'in_person': 1, 'out_person': 0}
    entry2 = {'ts': now-60, 'in_person': 2, 'out_person': 1}
    redis_client.zadd('history', {json.dumps(entry1): entry1['ts']})
    redis_client.zadd('history', {json.dumps(entry2): entry2['ts']})
    start = datetime.fromtimestamp(now-180).isoformat()
    end = datetime.fromtimestamp(now).isoformat()
    res = asyncio.run(reports.report_data(start, end, 'person', 'graph', 50, DummyRequest()))
    assert res['ins'] == [1, 2]
    assert res['outs'] == [0, 1]
    assert res['current'] == [1, 1]


def test_ppe_report_data(redis_client):
    now = int(time.time())
    entry = {'ts': now-30, 'cam_id': 1, 'track_id': 2, 'status': 'no_helmet', 'conf': 0.9, 'color': None, 'path': 'snap.jpg'}
    redis_client.zadd('ppe_logs', {json.dumps(entry): entry['ts']})
    start = datetime.fromtimestamp(now-60).isoformat()
    end = datetime.fromtimestamp(now).isoformat()
    res = asyncio.run(ppe_reports.ppe_report_data(start, end, 'no_helmet', None, None))
    assert len(res['rows']) == 1
    assert res['rows'][0]['status'] == 'no_helmet'
