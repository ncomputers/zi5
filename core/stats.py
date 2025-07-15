from __future__ import annotations
import json
import redis
from typing import Dict
from .config import COUNT_GROUPS, ANOMALY_ITEMS


def gather_stats(trackers: Dict[int, 'PersonTracker'], r: redis.Redis) -> dict:
    """Collect aggregated counts and anomaly metrics."""
    group_counts = {}
    for g in COUNT_GROUPS.keys():
        in_c = sum(t.in_counts.get(g, 0) for t in trackers.values())
        out_c = sum(t.out_counts.get(g, 0) for t in trackers.values())
        group_counts[g] = {'in': in_c, 'out': out_c, 'current': in_c - out_c}
    anomaly_counts = {item: int(r.get(f'{item}_count') or 0) for item in ANOMALY_ITEMS}
    data = {
        'group_counts': group_counts,
        'anomaly_counts': anomaly_counts,
        'people_in': group_counts.get('person', {}).get('current', 0),
        'people_entered': group_counts.get('person', {}).get('in', 0),
        'people_exited': group_counts.get('person', {}).get('out', 0),
        'vehicles_in': group_counts.get('vehicle', {}).get('current', 0),
    }
    return data


def broadcast_stats(trackers: Dict[int, 'PersonTracker'], r: redis.Redis) -> None:
    """Publish the latest stats to listeners via Redis pub/sub."""
    data = gather_stats(trackers, r)
    r.publish('stats_updates', json.dumps(data))
