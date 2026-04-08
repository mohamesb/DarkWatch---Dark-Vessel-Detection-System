# DARKWATCH — Dark Vessel Detection System

Real-time AIS silence monitoring for Finnish maritime waters with Kalman Filter ghost trajectory prediction.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DARKWATCH SYSTEM                            │
├──────────────────────┬──────────────────────────────────────────────┤
│                      │                                              │
│  Digitraffic API     │   Flask Backend (server.py)                  │
│  (AIS Feed)          │                                              │
│  meri.digitraffic.fi ──→  Background Thread (polls every 30s)      │
│                      │        │                                     │
│                      │        ▼                                     │
│                      │   VesselStore (thread-safe in-memory DB)     │
│                      │    ├─ Track accumulation (30 pts/vessel)     │
│                      │    ├─ Dark detection (silence + SOG + port)  │
│                      │    └─ Stale vessel purge (1hr timeout)       │
│                      │        │                                     │
│                      │        ▼                                     │
│                      │   REST API Endpoints                        │
│                      │    ├─ GET /api/vessels                       │
│                      │    ├─ GET /api/vessels/dark                  │
│                      │    ├─ GET /api/vessel/<mmsi>                 │
│                      │    ├─ GET /api/predict/<mmsi>                │
│                      │    ├─ GET /api/stats                         │
│                      │    └─ GET /api/ports                         │
│                      │        │                                     │
│                      │        ▼                                     │
│                      │   Frontend (static/index.html)               │
│                      │    ├─ Leaflet.js dark map                    │
│                      │    ├─ Animated vessel markers                │
│                      │    ├─ Dark vessel alerts                     │
│                      │    ├─ Track history visualization            │
│                      │    └─ Ghost trajectory rendering             │
└──────────────────────┴──────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Clone / download
cd darkwatch

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python server.py
```

Open **http://localhost:5000** in your browser.

The backend immediately starts polling the Digitraffic API. Within ~3 seconds you'll see live vessel positions on the map.

## Project Structure

```
darkwatch/
├── server.py              # Flask backend + AIS ingestion + Kalman filter
├── static/
│   └── index.html         # Full frontend (single-file, no build step)
├── requirements.txt
└── README.md
```

## How Dark Vessel Detection Works

A vessel is classified as **dark** when all three conditions are met:

| Condition | Default Threshold | Rationale |
|-----------|-------------------|-----------|
| AIS silence duration | ≥ 15 minutes | Normal AIS transmits every 2–30 seconds |
| Speed Over Ground | ≥ 1.0 knots | Stationary vessels aren't evading |
| Not near a known port | Outside port radius | Docked vessels don't need continuous AIS |

The silence threshold is adjustable in real-time via the sidebar slider (5–60 min).

## Ghost Trajectory Prediction

The system uses a **Kalman Filter** with a constant-velocity state model:

**State vector:** `x = [lat, lon, v_lat, v_lon]ᵀ`

**Transition model:**
```
x(k+1) = F · x(k) + w

F = | 1  0  Δt  0  |
    | 0  1  0   Δt |
    | 0  0  1   0  |
    | 0  0  0   1  |
```

**Process noise** uses the continuous white-noise jerk model scaled by σ = 5×10⁻⁹, which allows moderate course/speed changes between observations.

**When fewer than 2 track points exist**, the system falls back to dead-reckoning using the Haversine-based rhumb line projection with cos(φ) correction for meridian convergence at ~60°N.

## API Reference

| Endpoint | Params | Description |
|----------|--------|-------------|
| `GET /api/vessels` | `silence`, `min_sog`, `category`, `dark_only` | All vessels with detection |
| `GET /api/vessels/dark` | `silence`, `min_sog` | Dark vessels only |
| `GET /api/vessel/<mmsi>` | — | Single vessel + track history |
| `GET /api/predict/<mmsi>` | `steps`, `interval` | Kalman ghost trajectory |
| `GET /api/stats` | — | System statistics |
| `GET /api/ports` | — | Known port positions |

## Frontend Features

- **Dark ops-center aesthetic** — CRT scanline overlay, JetBrains Mono typography, cyan/red color scheme
- **Smooth marker animation** — vessels glide to new positions on each update cycle
- **Type filter chips** — toggle tanker/cargo/passenger/fishing/other
- **Search** — filter by vessel name or MMSI
- **Mode tabs** — ALL / DARK ONLY / UNDERWAY
- **Detail panel** — click any vessel for full telemetry, track visualization, and prediction
- **Ghost trajectory** — orange dashed line with waypoint markers projected from the vessel's last known state

## Business Value

### Environmental Protection
- **IUU Fishing**: Vessels disabling AIS near MPAs may be fishing illegally. Ghost trajectories help predict interception points.
- **Illegal Dumping**: Dark periods near sensitive areas trigger alerts for the Finnish Environment Institute (SYKE).
- **Oil Spill Attribution**: Correlating dark vessel trajectories with satellite SAR detections identifies responsible parties.

### Sovereign Security
- **Shadow Fleet Monitoring**: Sanctions-evading tankers routinely disable AIS for ship-to-ship transfers. DARKWATCH flags these gaps in Finnish waters and the EEZ.
- **Search & Rescue**: A vessel going silent may indicate distress. Predicted trajectories narrow the search area for MRCC Turku.
- **NATO Maritime Domain Awareness**: Dark vessel alerts feed into alliance COP frameworks via MARSUR and the EU's CISE platform.

## License

MIT
