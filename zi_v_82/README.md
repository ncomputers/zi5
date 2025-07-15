# Crowd Management System v82

Version 82 separates the person counting and PPE detection logic into two
independent modules.  The basic **PersonTracker** detects and tracks people and
logs entry/exit events to `person_logs`.  A new **PPEDetector** reads those log
entries that require PPE checks and stores the results in `ppe_logs`.  Camera
configuration now uses grouped tasks for counting and PPE detection.

Duplicate frame removal and all other features from the previous release are
still available.

## Features
- **Multiple camera sources**: Add HTTP or RTSP cameras via the settings page.
- **Person counting and PPE checks**: YOLOv8 is used for person detection and, when enabled, for verifying required PPE.
- **Counting and alerts**: Tracks entries/exits and can send email alerts based on customizable rules.
- **Duplicate frame filter**: Skips nearly identical frames to reduce GPU/CPU load.
- **Dashboard and reports**: Live counts, recent anomalies, and historical reports are available in the web interface.
- **Per-camera resolution**: Choose 480p, 720p, 1080p, or original when adding a camera.
- **Real-time updates via SSE**: Dashboard values refresh automatically without manual reload.
- **Camera status**: Online/offline indicators appear in the Cameras page for quick troubleshooting.
- **Secure logins**: User passwords are stored as PBKDF2 hashes and verified using passlib.
- **Rotating log file**: `app.log` captures runtime logs with automatic rotation.
- **Historical reports**: A background task records per-minute counts to Redis so
  the reports page can graph occupancy over time. Log entries are stored in Redis
  sorted sets for efficient range queries.

## Installation
1. Install Python 3.10+ and Redis.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Install PHP if you want to use the sample PHP pages in `public/`.

## Configuration
Edit `config.json` to set camera URLs, model paths, thresholds, and email settings. Most options can also be adjusted in the web UI under **Settings**. Key fields include:

- `stream_url` – Default video source when none are configured.
- `person_model`, `ppe_model` – Paths to YOLO models.
- `device` – `auto`, `cpu`, or `cuda:0`.
- `max_capacity` and `warn_threshold` – Occupancy limits.
- `redis_url` – Location of the Redis instance.

## Running
Launch the FastAPI application:
```bash
python3 app.py
```
Then open `http://localhost:5002` in your browser. Use the **Cameras** page to add streams (HTTP, RTSP or local webcams) and **Settings** to adjust options. Tests can be executed with `pytest`:
```bash
python3 -m pytest -q tests
```

## Directory Structure
- `app.py` – FastAPI entry point.
- `core/` – Helper modules such as configuration and tracker manager.
- `modules/` – Tracking, alerts, and utilities.
- `routers/` – API routes for dashboard, settings, reports, and cameras.
- `templates/` – HTML templates rendered by FastAPI.
- `public/` – Optional PHP pages.
- `tests/` – Simple unit tests.

