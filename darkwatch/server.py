"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DARKWATCH — Dark Vessel Detection System                                  ║
║  Backend: Flask + Background AIS Ingestion Thread                          ║
║  Data:    Fintraffic Digitraffic Marine API (no key required)              ║
║  Model:   Kalman Filter (constant-velocity) for ghost trajectory           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Endpoints:
  GET /                        → Serves the frontend
  GET /api/vessels              → All tracked vessels (filtered, paginated)
  GET /api/vessels/dark         → Dark vessels only
  GET /api/vessel/<mmsi>        → Single vessel detail + track history
  GET /api/predict/<mmsi>       → Kalman-predicted ghost trajectory
  GET /api/stats                → System-wide statistics
  GET /api/ports                → Known port locations
"""

import math
import os
import time
import threading
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import requests
from flask import Flask, jsonify, send_from_directory, request

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

API_URL = "https://meri.digitraffic.fi/api/ais/v1/locations"
METADATA_URL = "https://meri.digitraffic.fi/api/ais/v1/vessels"
VESSEL_META_URL = "https://meri.digitraffic.fi/api/ais/v1/vessels/{mmsi}"
POLL_INTERVAL = 10          # Seconds between API polls
META_POLL_INTERVAL = 300    # Seconds between metadata refreshes (5 min)
MAX_HISTORY = 30            # Max track points kept per vessel
STALE_TIMEOUT = 3600        # Remove vessels not seen for 1 hour

# Default detection thresholds (can be overridden per-request via query params)
DEFAULT_SILENCE_MIN = 15    # Minutes without AIS update to flag as dark
DEFAULT_MIN_SOG = 1.0       # Minimum SOG (knots) to be considered "underway"

# Prediction
PRED_STEPS = 10
PRED_INTERVAL_S = 300       # 5 minutes between predicted waypoints

# Known ports: (name, lat, lon, exclusion_radius_nm)
PORTS = [
    ("Helsinki",       60.1550, 24.9530, 2.0),
    ("Turku",          60.4350, 22.2250, 2.0),
    ("Kotka",          60.4700, 26.9450, 2.0),
    ("Rauma",          61.1280, 21.4620, 1.5),
    ("Hamina",         60.5630, 27.1840, 1.5),
    ("Hanko",          59.8230, 22.9680, 1.5),
    ("Naantali",       60.4680, 22.0240, 1.5),
    ("Pori",           61.5940, 21.4710, 1.5),
    ("Oulu",           65.0080, 25.4280, 1.5),
    ("Vaasa",          63.0900, 21.6100, 1.5),
    ("Kokkola",        63.8380, 23.0460, 1.5),
    ("Tallinn",        59.4500, 24.7650, 2.0),
    ("Stockholm",      59.3300, 18.0500, 2.0),
    ("St. Petersburg", 59.9300, 30.2500, 3.0),
    ("Riga",           56.9500, 24.1050, 2.0),
    ("Gdansk",         54.3520, 18.6570, 2.0),
]

# Ship type ranges (AIS standard)
SHIP_TYPES = {
    "tanker":    range(80, 90),
    "cargo":     range(70, 80),
    "passenger": range(60, 70),
    "fishing":   [30],
    "tug":       [31, 32, 52],
    "military":  [35],
    "sailing":   [36],
    "pleasure":  [37],
}

# AIS navigational status codes
NAV_STATUS = {
    0: "Under way using engine",
    1: "At anchor",
    2: "Not under command",
    3: "Restricted manoeuvrability",
    4: "Constrained by draught",
    5: "Moored",
    6: "Aground",
    7: "Engaged in fishing",
    8: "Under way sailing",
    9: "Reserved (HSC)",
    10: "Reserved (WIG)",
    11: "Power-driven towing astern",
    12: "Power-driven pushing/towing",
    14: "AIS-SART / MOB / EPIRB",
    15: "Not defined",
}


def decode_eta(eta_raw):
    """
    Decode AIS ETA from the packed integer format used by Digitraffic.
    Returns a human-readable string or None.
    """
    if not eta_raw or eta_raw == 0:
        return None
    try:
        eta = int(eta_raw)
        # Digitraffic bit packing: month(4b) day(5b) hour(5b) minute(6b)
        minute = eta & 0x3F
        hour = (eta >> 6) & 0x1F
        day = (eta >> 11) & 0x1F
        month = (eta >> 16) & 0xF
        if 1 <= month <= 12 and 1 <= day <= 31 and hour < 24 and minute < 60:
            return f"{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC"
        # Fallback: plain integer MMDDHHMM
        minute = eta % 100
        hour = (eta // 100) % 100
        day = (eta // 10000) % 100
        month = (eta // 1000000) % 100
        if 1 <= month <= 12 and 1 <= day <= 31 and hour < 24 and minute < 60:
            return f"{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC"
    except (ValueError, TypeError):
        pass
    return str(eta_raw)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("darkwatch")


# ═══════════════════════════════════════════════════════════════════════════════
# SPATIOTEMPORAL MATH
# ═══════════════════════════════════════════════════════════════════════════════

def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance via the Haversine formula, returned in nautical miles.

        a = sin²(Δφ/2) + cos(φ₁)·cos(φ₂)·sin²(Δλ/2)
        d = 2·R·atan2(√a, √(1−a))     where R = 3440.065 NM
    """
    R = 3440.065
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ, Δλ = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def project_position(lat, lon, cog_deg, sog_kn, dt_s):
    """
    Dead-reckoning projection along a rhumb line.

        dist_nm  = SOG × (Δt / 3600)
        Δlat°    = (dist / 60) × cos(COG)
        Δlon°    = (dist / 60) × sin(COG) / cos(φ)

    The 1/cos(φ) factor corrects for meridian convergence — critical at ~60°N.
    """
    dist = sog_kn * (dt_s / 3600.0)
    cog = math.radians(cog_deg)
    dlat = (dist / 60.0) * math.cos(cog)
    dlon = (dist / 60.0) * math.sin(cog) / max(math.cos(math.radians(lat)), 1e-6)
    return lat + dlat, lon + dlon


def near_port(lat, lon):
    """Check if a position falls within any known port exclusion zone."""
    return any(haversine_nm(lat, lon, plat, plon) <= r for _, plat, plon, r in PORTS)


def classify_ship(ship_type):
    """Map AIS numeric ship type to a category string."""
    if ship_type is None:
        return "other"
    t = int(ship_type)
    for cat, codes in SHIP_TYPES.items():
        if t in codes:
            return cat
    return "other"


# ═══════════════════════════════════════════════════════════════════════════════
# KALMAN FILTER — CONSTANT-VELOCITY MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def kalman_predict(history: list[dict], steps=PRED_STEPS, interval=PRED_INTERVAL_S):
    """
    2-D Kalman Filter for maritime trajectory prediction.

    State vector:   x = [lat, lon, v_lat, v_lon]ᵀ
    Transition:     x(k+1) = F·x(k) + w     (constant velocity)
    Observation:    z(k) = H·x(k) + v        (position only)

    F = | 1  0  Δt  0  |       H = | 1  0  0  0 |
        | 0  1  0   Δt |           | 0  1  0  0 |
        | 0  0  1   0  |
        | 0  0  0   1  |

    Process noise Q uses the continuous white-noise jerk model, scaled by σ:

        Q = σ × | Δt³/3  0      Δt²/2  0     |
                | 0      Δt³/3  0      Δt²/2 |
                | Δt²/2  0      Δt     0     |
                | 0      Δt²/2  0      Δt   |

    When < 2 observations exist, falls back to dead-reckoning from last COG/SOG.
    """
    pts = sorted(history, key=lambda p: p["timestamp"])

    # --- Fallback: dead reckoning ---
    if len(pts) < 2:
        p = pts[-1]
        path = []
        lat, lon = p["lat"], p["lon"]
        for _ in range(steps):
            lat, lon = project_position(lat, lon, p.get("cog", 0), p.get("sog", 0), interval)
            path.append({"lat": round(lat, 6), "lon": round(lon, 6)})
        return path

    # --- Initialise from first two observations ---
    p0, p1 = pts[0], pts[1]
    dt0 = max((p1["timestamp"] - p0["timestamp"]) / 1000.0, 1.0)
    vlat = (p1["lat"] - p0["lat"]) / dt0
    vlon = (p1["lon"] - p0["lon"]) / dt0

    x = np.array([p1["lat"], p1["lon"], vlat, vlon])
    P = np.diag([1e-6, 1e-6, 1e-8, 1e-8])
    R = np.diag([1e-8, 1e-8])
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    σ = 5e-9

    def make_FQ(dt):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        Q = σ * np.array([
            [dt**3/3, 0, dt**2/2, 0],
            [0, dt**3/3, 0, dt**2/2],
            [dt**2/2, 0, dt, 0],
            [0, dt**2/2, 0, dt],
        ])
        return F, Q

    # --- Assimilate observations forward ---
    for i in range(1, len(pts) - 1):
        dt_k = max((pts[i + 1]["timestamp"] - pts[i]["timestamp"]) / 1000.0, 1.0)
        F, Q = make_FQ(dt_k)
        x = F @ x
        P = F @ P @ F.T + Q
        z = np.array([pts[i + 1]["lat"], pts[i + 1]["lon"]])
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(4) - K @ H) @ P

    # --- Forward prediction ---
    path = []
    for _ in range(steps):
        F, Q = make_FQ(interval)
        x = F @ x
        P = F @ P @ F.T + Q
        path.append({"lat": round(float(x[0]), 6), "lon": round(float(x[1]), 6)})

    return path


# ═══════════════════════════════════════════════════════════════════════════════
# VESSEL STORE (Thread-safe in-memory database)
# ═══════════════════════════════════════════════════════════════════════════════

class VesselStore:
    """Thread-safe vessel state manager with track accumulation and metadata cache."""

    def __init__(self):
        self._lock = threading.Lock()
        self._vessels = {}        # mmsi (str) -> dict
        self._metadata = {}       # mmsi (str) -> dict (from /api/ais/v1/vessels)
        self._meta_lock = threading.Lock()
        self._cycle = 0
        self._last_fetch = None
        self._fetch_ms = 0

    def ingest_metadata(self, vessel_list: list):
        """
        Ingest bulk vessel metadata from /api/ais/v1/vessels.

        Each entry contains: mmsi, name, imo, callSign, destination, draught,
        eta, shipType, refA, refB, refC, refD, posType, timestamp.
        refA+refB = overall length, refC+refD = overall beam.
        Draught is in 1/10 meter units.
        """
        with self._meta_lock:
            for entry in vessel_list:
                mmsi = str(entry.get("mmsi", ""))
                if not mmsi:
                    continue
                ref_a = entry.get("refA") or 0
                ref_b = entry.get("refB") or 0
                ref_c = entry.get("refC") or 0
                ref_d = entry.get("refD") or 0
                draught_raw = entry.get("draught") or 0
                self._metadata[mmsi] = {
                    "name": entry.get("name") or None,
                    "imo": entry.get("imo") or None,
                    "call_sign": entry.get("callSign") or None,
                    "destination": entry.get("destination") or None,
                    "draught": round(draught_raw / 10.0, 1) if draught_raw else None,
                    "eta": decode_eta(entry.get("eta")),
                    "length": ref_a + ref_b if (ref_a or ref_b) else None,
                    "beam": ref_c + ref_d if (ref_c or ref_d) else None,
                    "ref_a": ref_a, "ref_b": ref_b,
                    "ref_c": ref_c, "ref_d": ref_d,
                    "nav_status": entry.get("navStat"),
                    "nav_status_text": NAV_STATUS.get(entry.get("navStat"), None),
                    "heading": entry.get("heading"),
                    "rot": entry.get("rot"),
                    "pos_type": entry.get("posType"),
                    "meta_timestamp": entry.get("timestamp"),
                }

    def get_metadata(self, mmsi: str) -> dict:
        """Return cached metadata for a vessel, or empty dict."""
        with self._meta_lock:
            return dict(self._metadata.get(mmsi, {}))

    def ingest(self, features: list):
        """Process a batch of GeoJSON features from the Digitraffic API."""
        now_ms = int(time.time() * 1000)
        with self._lock:
            self._cycle += 1
            seen = set()
            for feat in features:
                props = feat.get("properties", {})
                coords = feat.get("geometry", {}).get("coordinates")
                if not coords or coords[0] is None:
                    continue

                mmsi = str(props.get("mmsi", ""))
                if not mmsi:
                    continue
                seen.add(mmsi)

                ts = props.get("timestampExternal", 0)
                if ts > 1e12:
                    ts = int(ts)
                else:
                    ts = int(ts * 1000)

                lat, lon = coords[1], coords[0]
                sog = props.get("sog") or 0
                cog = props.get("cog") or 0
                ship_type = props.get("shipType")
                name = props.get("name") or f"MMSI {mmsi}"

                pos = {"lat": lat, "lon": lon, "sog": sog, "cog": cog, "timestamp": ts}

                if mmsi not in self._vessels:
                    self._vessels[mmsi] = {
                        "mmsi": mmsi, "name": name,
                        "ship_type": ship_type,
                        "category": classify_ship(ship_type),
                        "history": [],
                        "first_seen": now_ms,
                    }
                v = self._vessels[mmsi]
                v["lat"] = lat
                v["lon"] = lon
                v["sog"] = sog
                v["cog"] = cog
                v["timestamp"] = ts
                v["name"] = name
                v["ship_type"] = ship_type
                v["category"] = classify_ship(ship_type)

                # Append to track history (deduplicate by timestamp)
                hist = v["history"]
                if not hist or hist[-1]["timestamp"] != ts:
                    hist.append(pos)
                    if len(hist) > MAX_HISTORY:
                        v["history"] = hist[-MAX_HISTORY:]

            # Purge stale vessels
            cutoff = now_ms - (STALE_TIMEOUT * 1000)
            stale = [m for m, v in self._vessels.items() if v["timestamp"] < cutoff]
            for m in stale:
                del self._vessels[m]

            self._last_fetch = datetime.now(timezone.utc).isoformat()

    def get_all(self, silence_min=DEFAULT_SILENCE_MIN, min_sog=DEFAULT_MIN_SOG):
        """Return all vessels with dark-detection applied."""
        now_ms = int(time.time() * 1000)
        with self._lock:
            result = []
            for v in self._vessels.values():
                silence = (now_ms - v["timestamp"]) / 60000.0
                is_dark = (
                    silence >= silence_min
                    and v["sog"] >= min_sog
                    and not near_port(v["lat"], v["lon"])
                )
                result.append({
                    "mmsi": v["mmsi"],
                    "name": v["name"],
                    "lat": round(v["lat"], 6),
                    "lon": round(v["lon"], 6),
                    "sog": round(v["sog"], 1),
                    "cog": round(v["cog"], 1),
                    "ship_type": v["ship_type"],
                    "category": v["category"],
                    "timestamp": v["timestamp"],
                    "silence_min": round(silence, 1),
                    "is_dark": is_dark,
                    "track_points": len(v["history"]),
                })

        # Enrich names from metadata cache (outside vessel lock)
        with self._meta_lock:
            for entry in result:
                meta = self._metadata.get(entry["mmsi"])
                if meta and meta.get("name"):
                    if entry["name"].startswith("MMSI ") or not entry["name"]:
                        entry["name"] = meta["name"]

        return result

    def get_vessel(self, mmsi: str):
        """Return single vessel detail with full track history + metadata."""
        now_ms = int(time.time() * 1000)
        with self._lock:
            v = self._vessels.get(mmsi)
            if not v:
                return None
            silence = (now_ms - v["timestamp"]) / 60000.0
            result = {
                "mmsi": v["mmsi"],
                "name": v["name"],
                "lat": round(v["lat"], 6),
                "lon": round(v["lon"], 6),
                "sog": round(v["sog"], 1),
                "cog": round(v["cog"], 1),
                "ship_type": v["ship_type"],
                "category": v["category"],
                "timestamp": v["timestamp"],
                "silence_min": round(silence, 1),
                "is_dark": (
                    silence >= DEFAULT_SILENCE_MIN
                    and v["sog"] >= DEFAULT_MIN_SOG
                    and not near_port(v["lat"], v["lon"])
                ),
                "track": list(v["history"]),
            }

        # Merge metadata (outside vessel lock to avoid deadlock)
        meta = self.get_metadata(mmsi)
        # Prefer metadata name when position-API name is just "MMSI ..." fallback
        meta_name = meta.get("name")
        if meta_name and (result["name"].startswith("MMSI ") or not result["name"]):
            result["name"] = meta_name
        result["imo"] = meta.get("imo")
        result["call_sign"] = meta.get("call_sign")
        result["destination"] = meta.get("destination")
        result["draught"] = meta.get("draught")
        result["eta"] = meta.get("eta")
        result["length"] = meta.get("length")
        result["beam"] = meta.get("beam")
        result["heading"] = meta.get("heading")
        result["rot"] = meta.get("rot")
        result["nav_status"] = meta.get("nav_status")
        result["nav_status_text"] = meta.get("nav_status_text")
        result["has_metadata"] = bool(meta)
        return result

    def get_history(self, mmsi: str):
        """Return raw track history for prediction."""
        with self._lock:
            v = self._vessels.get(mmsi)
            return list(v["history"]) if v else []

    def stats(self):
        with self._lock:
            now_ms = int(time.time() * 1000)
            total = len(self._vessels)
            dark = sum(
                1 for v in self._vessels.values()
                if (now_ms - v["timestamp"]) / 60000 >= DEFAULT_SILENCE_MIN
                and v["sog"] >= DEFAULT_MIN_SOG
                and not near_port(v["lat"], v["lon"])
            )
            return {
                "total": total,
                "active": total - dark,
                "dark": dark,
                "cycle": self._cycle,
                "last_fetch": self._last_fetch,
            }


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND INGESTION THREAD
# ═══════════════════════════════════════════════════════════════════════════════

store = VesselStore()


def poll_loop():
    """Continuously poll the Digitraffic locations API in a background thread."""
    while True:
        try:
            t0 = time.time()
            resp = requests.get(API_URL, timeout=20, headers={"Accept-Encoding": "gzip"})
            resp.raise_for_status()
            data = resp.json()
            features = data.get("features", [])
            store.ingest(features)
            elapsed = round((time.time() - t0) * 1000)
            log.info(f"Cycle {store._cycle}: ingested {len(features)} vessels in {elapsed}ms")
        except Exception as e:
            log.error(f"Position poll failed: {e}")
        time.sleep(POLL_INTERVAL)


def meta_poll_loop():
    """Periodically fetch bulk vessel metadata (name, IMO, destination, dimensions)."""
    while True:
        try:
            t0 = time.time()
            resp = requests.get(METADATA_URL, timeout=30, headers={"Accept-Encoding": "gzip"})
            resp.raise_for_status()
            data = resp.json()
            store.ingest_metadata(data)
            elapsed = round((time.time() - t0) * 1000)
            log.info(f"Metadata refresh: {len(data)} entries in {elapsed}ms")
        except Exception as e:
            log.error(f"Metadata poll failed: {e}")
        time.sleep(META_POLL_INTERVAL)


def fetch_single_metadata(mmsi: str) -> dict:
    """On-demand fetch of metadata for a single vessel (fallback if bulk cache misses)."""
    try:
        url = VESSEL_META_URL.format(mmsi=mmsi)
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            # Ingest this single entry into the cache
            store.ingest_metadata([data])
            return store.get_metadata(mmsi)
    except Exception as e:
        log.warning(f"Single metadata fetch for {mmsi} failed: {e}")
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/vessels")
def api_vessels():
    """
    GET /api/vessels?silence=15&min_sog=1.0&category=tanker&dark_only=false

    Returns all tracked vessels with dark detection applied.
    Optional query params override thresholds or filter by category.
    """
    silence = float(request.args.get("silence", DEFAULT_SILENCE_MIN))
    min_sog = float(request.args.get("min_sog", DEFAULT_MIN_SOG))
    category = request.args.get("category")
    dark_only = request.args.get("dark_only", "false").lower() == "true"

    vessels = store.get_all(silence_min=silence, min_sog=min_sog)

    if category:
        vessels = [v for v in vessels if v["category"] == category]
    if dark_only:
        vessels = [v for v in vessels if v["is_dark"]]

    return jsonify(vessels)


@app.route("/api/vessels/dark")
def api_dark():
    """GET /api/vessels/dark — Dark vessels only."""
    silence = float(request.args.get("silence", DEFAULT_SILENCE_MIN))
    min_sog = float(request.args.get("min_sog", DEFAULT_MIN_SOG))
    vessels = store.get_all(silence_min=silence, min_sog=min_sog)
    return jsonify([v for v in vessels if v["is_dark"]])


@app.route("/api/vessel/<mmsi>")
def api_vessel(mmsi):
    """GET /api/vessel/<mmsi> — Single vessel detail + track history + metadata."""
    v = store.get_vessel(mmsi)
    if not v:
        return jsonify({"error": "Vessel not found"}), 404
    # If metadata cache missed, try on-demand fetch
    if not v.get("has_metadata"):
        meta = fetch_single_metadata(mmsi)
        if meta:
            v.update({
                "imo": meta.get("imo"),
                "call_sign": meta.get("call_sign"),
                "destination": meta.get("destination"),
                "draught": meta.get("draught"),
                "eta": meta.get("eta"),
                "length": meta.get("length"),
                "beam": meta.get("beam"),
                "heading": meta.get("heading"),
                "rot": meta.get("rot"),
                "nav_status": meta.get("nav_status"),
                "nav_status_text": meta.get("nav_status_text"),
                "has_metadata": True,
            })
    return jsonify(v)


@app.route("/api/predict/<mmsi>")
def api_predict(mmsi):
    """
    GET /api/predict/<mmsi>?steps=10&interval=300

    Run Kalman filter prediction for the given vessel.
    Returns predicted ghost trajectory waypoints.
    """
    history = store.get_history(mmsi)
    if not history:
        return jsonify({"error": "No track history for this vessel"}), 404

    steps = int(request.args.get("steps", PRED_STEPS))
    interval = int(request.args.get("interval", PRED_INTERVAL_S))
    steps = min(steps, 30)  # cap

    predictions = kalman_predict(history, steps=steps, interval=interval)
    return jsonify({
        "mmsi": mmsi,
        "method": "kalman_cv",
        "steps": steps,
        "interval_s": interval,
        "predictions": predictions,
    })


@app.route("/api/stats")
def api_stats():
    """GET /api/stats — System-wide statistics."""
    return jsonify(store.stats())


@app.route("/api/ports")
def api_ports():
    """GET /api/ports — Known port positions."""
    return jsonify([
        {"name": n, "lat": la, "lon": lo, "radius_nm": r}
        for n, la, lo, r in PORTS
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("Starting background AIS ingestion thread…")
    t1 = threading.Thread(target=poll_loop, daemon=True)
    t1.start()

    log.info("Starting background metadata refresh thread…")
    t2 = threading.Thread(target=meta_poll_loop, daemon=True)
    t2.start()

    # Wait for first data cycle
    log.info("Waiting for initial data…")
    time.sleep(4)

    log.info("Starting Flask on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
