# TECH_RULES.md — FatigueGuard Platform
**Technical Architecture & Coding Rules**
**Version:** 2.0

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     User's Device (Browser)                  │
│                                                              │
│   getUserMedia() → webcam frames                            │
│   MediaPipe Face Landmarker (WASM) → 478 landmarks          │
│   React UI → webcam feed + dashboard display                │
│   Web Audio API → alarm sounds                              │
│   Browser Geolocation API → GPS coordinates                  │
└──────────────────────┬──────────────────────────────────────┘
                       │ WebSocket (landmarks per frame)
                       │ REST (auth, profile, sessions)
                       │ HTTPS only
┌──────────────────────▼──────────────────────────────────────┐
│               Azure Container Apps                           │
│               FastAPI Backend (Python)                       │
│                                                              │
│   WebSocket handler → receives landmarks                     │
│   features.py (REUSED) → EAR, MAR, tilt, pitch             │
│   tracker.py (REUSED) → PERCLOS, blink rate, fatigue score  │
│   LightGBM .pkl (REUSED) → Hybrid model inference           │
│   Threshold logic (REUSED) → rule-based classification      │
│   Twilio → WhatsApp emergency alerts                        │
│   Firebase Admin → FCM push to mobile app                   │
│   Auth → Supabase JWT verification                          │
└──────────────────────┬──────────────────────────────────────┘
                       │ PostgreSQL protocol
┌──────────────────────▼──────────────────────────────────────┐
│                     Supabase                                 │
│   PostgreSQL → users, sessions, emergency_contacts,         │
│                medical_profiles, alert_events               │
└─────────────────────────────────────────────────────────────┘
                       ▲
                       │ REST + Supabase JS client
┌──────────────────────┴──────────────────────────────────────┐
│               React Native Mobile App                        │
│   Expo SDK — Android + iOS                                   │
│   Session history, analytics, profile management            │
│   "Open Web App" deeplink button                            │
│   FCM push notifications                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Code Reuse Map

This section is based on the actual source files (`threshold_model/` and `hybrid_model/`), verified directly against `launcher.py`, which confirms the real entry points:
```python
HYBRID_MAIN = os.path.join(BASE_DIR, "hybrid_model", "main.py")
THRESHOLD_MAIN = os.path.join(BASE_DIR, "threshold_model", "main.py")
```

**Important — the two models are not variants of shared code. They are two distinct pipelines** with separate `config.py`, `features.py`, `tracker.py`, `ui.py` files. `hybrid_model/main_hybrid_3state_final.py` is a legacy, self-contained single-file draft — **not used by `launcher.py`** and not ported to the backend.

> ⚠️ **Known discrepancy:** `README.md` instructs running `python hybrid_model/main_hybrid3state_final.py` directly, which contradicts `launcher.py`'s actual code (`hybrid_model/main.py`). This document treats `launcher.py` as authoritative since it's the real, currently-running entry point — but the README itself should be corrected separately so the repo doesn't mislead future contributors or anyone evaluating the project.

| Existing File | Web Platform Fate | Notes |
|---------------|-------------------|-------|
| `threshold_model/features.py` | **REUSED 100%** | `calculate_ear`, `calculate_mar`, `calculate_roll_tilt`, `calculate_pitch`, redness functions. Uses `compute_fatigue_score()` with hardcoded 35/65 thresholds. |
| `threshold_model/tracker.py` | **REUSED 100%** | `FatigueTracker` — single rule-based score, CSV logging removed for web (replaced by DB writes). |
| `hybrid_model/features.py` | **REUSED 100%** | Same geometry functions as threshold, plus `safe_stats()`, `compute_rule_fatigue_score_exact()` (imports `ALERT_MAX`/`MILD_FATIGUE_MAX` from config — must be passed as parameters in backend, not module-level config import), and `compute_display_probability()`. |
| `hybrid_model/tracker.py` | **REUSED 100%** | `Hybrid3StateTracker` — critically different from threshold's tracker: tracks `valid_face_ratio`, has streak-based state stabilization (`alert_streak`/`mild_streak`/`fatigue_streak`), and `build_feature_dict()` produces the **exact stats dict** (mean/std/min/max per metric) the LightGBM model expects. |
| `hybrid_model/predictor.py` | **REUSED 100%** | `FatigueMLPredictor` — loads 3 `.pkl` files (model, feature column order, threshold) via `joblib`, predicts via `predict_proba()`. Must run server-side; `joblib`/`pandas` are backend-only dependencies. |
| `fatigue_lightgbm_model_final.pkl`, `fatigue_lightgbm_feature_columns_final.pkl`, `fatigue_lightgbm_final_threshold.pkl` | **REUSED directly** | All three required together — `feature_columns_path` defines exact column order `predictor.py` expects from the feature dict. |
| `whatsapp_notifier.py` | **REUSED — minor modification** | Verified identical in both models. `send_whatsapp_alert(sid, token, from_number, to_numbers, message_text)` — simple Twilio `Client.messages.create()` loop, returns SIDs. Does **no** phone number formatting/validation itself — E.164 formatting must be enforced wherever numbers are entered (mobile Profile screen), not assumed here. |
| `location_helper.py` + `get_location.html` | **REPLACED — see §5 below** | Browser `navigator.geolocation` replaces this entirely; no manual file-based flow needed for the web app. |
| `detector.py` (both) | **REPLACED by MediaPipe JS** | Verified: hybrid's `detect_face_landmarks(frame, landmarker, timestamp_ms)` takes an explicit timestamp (computed from frame `dt` in `main.py`); threshold's version computes its own timestamp internally via `cv2.getTickCount()`. Neither matters for the web port — browser MediaPipe JS handles its own timing — but confirms the two pipelines really do diverge at this layer too, not just at the tracker/model level. |
| `threshold_model/main.py` | **PORTED to `/ws/detect?mode=threshold`** | Single `tracker.update_summary_if_needed()` call per 8s window → one fatigue score. |
| `hybrid_model/main.py` | **PORTED to `/ws/detect?mode=hybrid`** | Per 8s window: `tracker.build_feature_dict()` → `predictor.predict()` → `tracker.update_summary()` (which calls `combine_rule_and_model()` internally). This is the verified real flow, confirmed against `launcher.py`. |
| `hybrid_model/main_hybrid_3state_final.py` | **NOT PORTED — legacy reference only** | Not used by `launcher.py`. Logic is duplicated inline (no module imports) and may drift from the real modular version over time. Use only to resolve ambiguity if `hybrid_model/`'s split files ever disagree with each other. |
| `ui.py` (both — different signatures) | **REPLACED by React** | Hybrid's `build_dashboard_panel()` has 3 extra fields threshold's doesn't: `display_prob`, `raw_model_prob`. React dashboard must conditionally render these only in Hybrid mode. |
| `alarm_manager.py` | **REPLACED by Web Audio API** | Verified identical in both models — `pygame.mixer.music.play(-1)` (loop forever) / `.stop()`. Clean 1:1 mapping: `audio.loop = true; audio.play()` / `audio.pause(); audio.currentTime = 0`. No fade, volume ramp, or other logic to replicate. |
| `launcher.py` | **REPLACED by React Launcher screen** | — |
| `config.py` (both — different) | **REPLACED by `.env` + FastAPI Settings** | Hybrid's config has additional fields: `MIN_OBSERVATION_BEFORE_DECISION`, `MODEL_MILD_THRESHOLD`, `MODEL_FATIGUE_THRESHOLD`, `CONSEC_WINDOWS_FOR_*`, `ALERT_MAX`/`MILD_FATIGUE_MAX`. All must be ported as backend settings. |

---

## 3. Tech Stack

### 3.1 Backend

The existing `requirements.txt` lists no version pins (just package names), so the versions below are recommended targets, not values carried over from the current project. Two stray entries in the existing `requirements.txt` — `features` and `detector` — are local module names, not real packages, and should be removed regardless of the web migration; they'll fail or silently no-op on `pip install`.

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.11+ | Runtime |
| FastAPI | 0.111+ | REST + WebSocket API framework |
| Uvicorn | 0.29+ | ASGI server |
| python-jose | 3.3+ | JWT verification |
| lightgbm | 4.x | Model inference |
| mediapipe | 0.10+ | (Development only — for testing landmark format) |
| twilio | 8.x | WhatsApp alerts |
| firebase-admin | 6.x | FCM push notifications |
| supabase-py | 2.x | Database client |
| python-dotenv | 1.x | Environment variable loading |
| numpy | 1.26+ | Feature calculations (used by features.py) |
| scipy | 1.12+ | (If used by features.py) |

### 3.2 Web Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18+ | UI framework |
| Vite | 5+ | Build tool |
| TypeScript | 5+ | Type safety |
| @mediapipe/tasks-vision | 0.10+ | Face Landmarker WASM — browser-side landmark detection |
| Zustand | 4+ | Global state management |
| React Router | 6+ | Client-side routing |
| Axios | 1.6+ | REST API calls |
| Recharts | 2.x | Charts on analytics pages |
| Lucide React | 0.383 | Icons |

### 3.3 Mobile App

| Technology | Version | Purpose |
|------------|---------|---------|
| React Native | 0.74+ | Cross-platform mobile |
| Expo SDK | 51+ | Managed workflow |
| React Navigation | 6+ | Navigation |
| Zustand | 4+ | State management |
| TanStack Query | 5+ | Server state + caching |
| Victory Native | 40+ | Charts |
| expo-secure-store | — | Secure token storage |
| expo-notifications | — | FCM push notifications |

### 3.4 Infrastructure

| Technology | Purpose |
|------------|---------|
| Docker | Single container: FastAPI backend + React build (static) |
| Azure Container Apps | Hosts the Docker container |
| Supabase | PostgreSQL database + Auth (email OTP + JWT) |
| Firebase Cloud Messaging | Push notifications to mobile app |
| Twilio | WhatsApp emergency alerts |
| GitHub Actions | CI/CD: push to main → build → push to Azure Container Registry → deploy |

---

## 4. Project Structure

### 4.1 Backend (Python — FastAPI)

```
backend/
├── app/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # Settings (from .env)
│   ├── dependencies.py          # JWT auth dependency
│   │
│   ├── routers/
│   │   ├── auth.py              # /auth/* endpoints
│   │   ├── profile.py           # /profile/* endpoints
│   │   ├── sessions.py          # /sessions/* endpoints
│   │   └── ws.py                # /ws/detect WebSocket endpoint
│   │
│   ├── services/
│   │   ├── detection_service.py # Orchestrates features.py + tracker.py + model
│   │   ├── alert_service.py     # Twilio + FCM
│   │   └── session_service.py   # Session save logic
│   │
│   ├── detection/               # REUSED CORE — copied from existing project as-is
│   │   ├── threshold/
│   │   │   ├── features.py      # ← COPIED FROM threshold_model/ (no changes)
│   │   │   └── tracker.py       # ← COPIED FROM threshold_model/ (no changes)
│   │   └── hybrid/
│   │       ├── features.py      # ← COPIED FROM hybrid_model/ (no changes)
│   │       ├── tracker.py       # ← COPIED FROM hybrid_model/ (no changes)
│   │       ├── predictor.py     # ← COPIED FROM hybrid_model/ (no changes)
│   │       └── models/
│   │           ├── fatigue_lightgbm_model_final.pkl
│   │           ├── fatigue_lightgbm_feature_columns_final.pkl
│   │           └── fatigue_lightgbm_final_threshold.pkl
│   │
│   ├── models/                  # Pydantic schemas
│   │   ├── user.py
│   │   ├── session.py
│   │   └── websocket.py
│   │
│   └── db/
│       └── supabase_client.py
│
├── static/                      # React build output (served by FastAPI)
│   └── (populated by Vite build step in Dockerfile)
│
├── Dockerfile
├── requirements.txt
└── .env.example
```

### 4.2 Web Frontend (React)

```
frontend/
├── src/
│   ├── pages/
│   │   ├── Landing.tsx          # Public landing page
│   │   ├── Login.tsx
│   │   ├── Register.tsx
│   │   ├── OTPVerification.tsx
│   │   ├── ProfileSetup.tsx     # Post-register onboarding
│   │   ├── Launcher.tsx         # Model selection screen
│   │   └── Detection.tsx        # Live webcam dashboard
│   │
│   ├── components/
│   │   ├── landing/
│   │   │   ├── HeroSection.tsx
│   │   │   ├── HowItWorks.tsx
│   │   │   ├── WhyUnique.tsx
│   │   │   └── TeamSection.tsx
│   │   ├── detection/
│   │   │   ├── WebcamFeed.tsx       # <video> + <canvas> overlay
│   │   │   ├── FatigueDashboard.tsx # Right panel — all metrics
│   │   │   ├── MetricRow.tsx        # Single EAR/MAR/etc. row
│   │   │   ├── StateIndicator.tsx   # Alert/Mild Fatigue/Fatigued badge
│   │   │   ├── AlarmOverlay.tsx     # "FATIGUE DETECTED" overlay
│   │   │   └── EmergencyModal.tsx   # Emergency escalation modal
│   │   └── ui/
│   │       ├── Button.tsx
│   │       └── Card.tsx
│   │
│   ├── hooks/
│   │   ├── useMediaPipe.ts      # Initialises MediaPipe Face Landmarker
│   │   ├── useWebSocket.ts      # Manages WS connection + message handling
│   │   ├── useAlarm.ts          # Web Audio API alarm management
│   │   └── useGeolocation.ts    # Browser GPS
│   │
│   ├── store/
│   │   ├── useAuthStore.ts
│   │   └── useDetectionStore.ts # Live metrics state during session
│   │
│   ├── services/
│   │   ├── api.ts               # Axios instance
│   │   └── authService.ts
│   │
│   └── constants/
│       └── colors.ts            # Shared design tokens
│
├── index.html
├── vite.config.ts
└── tsconfig.json
```

### 4.3 Mobile App (React Native)

```
mobile/
├── app/
│   ├── (auth)/
│   │   ├── index.tsx            # Splash
│   │   ├── login.tsx
│   │   ├── register.tsx
│   │   ├── otp.tsx
│   │   └── profile-setup.tsx
│   └── (tabs)/
│       ├── dashboard/
│       │   ├── index.tsx
│       │   └── [sessionId].tsx
│       ├── profile/
│       │   └── index.tsx
│       └── settings/
│           └── index.tsx
├── components/
├── store/
├── services/
└── constants/
```

---

## 5. WebSocket Protocol

### 5.1 Connection
```
ws://api.fatigueguard.app/ws/detect?token=<jwt>&mode=threshold|hybrid
```
JWT verified on connection. Connection rejected if token invalid. `mode` determines which tracker class and summary logic the backend uses (see §8).

### 5.2 Browser → Backend (per frame, ~30fps target)

```json
{
  "type": "frame",
  "timestamp_ms": 1712345678901,
  "landmarks": [
    [0.512, 0.334],
    [0.498, 0.341],
    ...
  ],
  "frame_width": 1280,
  "frame_height": 720
}
```

Notes:
- `landmarks` — 478 pairs of normalized (0–1) coordinates from MediaPipe JS
- `frame_width`, `frame_height` — used by backend to scale to pixel coordinates before `features.py`
- Send `"type": "no_face"` when MediaPipe returns no landmarks — backend calls `tracker.update_no_face()` (hybrid) or simply skips the frame (threshold, which has no equivalent method)
- **Location is no longer sent per-frame.** See §5.6 — it's only sent once, at session start, since GPS doesn't need 30fps updates.

### 5.3 Backend → Browser (per-frame response — both modes)

```json
{
  "type": "live",
  "ear": 0.289,
  "mar": 0.045,
  "tilt": -2.06
}
```
Sent every frame regardless of mode — mirrors the green live-metric text drawn on the desktop camera view (`draw_text(frame, f"Live EAR: ...")`). `redness` is intentionally omitted from this payload, not sent as `0.0` — the field is dropped for v1 (see §8 resolved decision), so there's nothing for the frontend to render or hide.

### 5.4 Backend → Browser (8-second summary response — mode-dependent)

**Threshold mode** — matches `threshold_model/tracker.py`'s `update_summary_if_needed()` output exactly:
```json
{
  "type": "summary",
  "mode": "threshold",
  "summary": { "avg_ear": 0.231, "avg_mar": 0.787, "avg_tilt": 4.58 },
  "indicators": { "perclos": 31.54, "blink_per_min": 37.39, "yawn_per_min": 14.96, "nod_per_min": 0.0, "eyes_closed_sec": 0.0 },
  "result": { "fatigue_score": 56.30, "state": "Mild Fatigue" },
  "alarm": "mild"
}
```

**Hybrid mode** — matches `hybrid_model/tracker.py`'s `update_summary()` output, includes two extra fields the threshold model has no equivalent for:
```json
{
  "type": "summary",
  "mode": "hybrid",
  "summary": { "avg_ear": 0.231, "avg_mar": 0.787, "avg_tilt": 4.58 },
  "indicators": { "perclos": 31.54, "blink_per_min": 37.39, "yawn_per_min": 14.96, "nod_per_min": 0.0, "eyes_closed_sec": 0.0 },
  "result": {
    "fatigue_score": 56.30,
    "state": "Mild Fatigue",
    "raw_model_prob": 0.6421,
    "display_prob": 0.612
  },
  "alarm": "mild"
}
```
React's `FatigueDashboard.tsx` must conditionally render `raw_model_prob`/`display_prob` only when `mode === "hybrid"`, matching `ui.py`'s extra two lines (`Display Prob`, `Raw Model Prob`) that only appear in `hybrid_model/ui.py`, not `threshold_model/ui.py`.

`alarm` values: `"none"` | `"mild"` | `"fatigue"` — derived the same way `main.py` decides in both models (state-based alarm start/stop logic).

### 5.5 Backend → Browser (emergency trigger)

```json
{
  "type": "emergency",
  "location": {
    "city": "Browser Detected",
    "region": "",
    "country": "India",
    "coordinates": "13.0982405,80.257549",
    "map_link": "https://www.google.com/maps?q=13.0982405,80.257549"
  },
  "contacts_notified": ["Amma", "Appa"],
  "message": "Emergency escalation triggered. WhatsApp alerts sent."
}
```
Matches the structure `location_helper.py`'s dict already returns — kept identical so `whatsapp_notifier.py`'s message-building logic in `main.py` ports over unchanged.

### 5.6 Browser → Backend (location — sent once, not per-frame)

```json
{
  "type": "location",
  "lat": 13.0982405,
  "lng": 80.257549
}
```
Sent once when the user grants browser geolocation permission (see §9 below) — typically right when the Detection screen mounts. The backend stores this on the WebSocket connection's session state and reuses it if/when an emergency triggers later. If the user denies permission, no `location` message is ever sent — backend falls back to the same `"Unknown City"` default `location_helper.py` already returns when `location.json` doesn't exist.

### 5.7 Browser → Backend (session end)

```json
{ "type": "end_session", "duration_seconds": 3621 }
```
Backend calculates final stats from tracker state, saves session record, sends FCM push to mobile app.

---

## 6. REST API Endpoints

All endpoints under `/api/v1/`. Frontend served from `/` (FastAPI mounts React static build).

### Auth
```
POST   /api/v1/auth/register
POST   /api/v1/auth/login
POST   /api/v1/auth/logout
POST   /api/v1/auth/refresh
POST   /api/v1/auth/forgot-password
POST   /api/v1/auth/verify-otp
POST   /api/v1/auth/reset-password
```

### Profile
```
GET    /api/v1/profile
PUT    /api/v1/profile
GET    /api/v1/profile/contacts
POST   /api/v1/profile/contacts
PUT    /api/v1/profile/contacts/{id}
DELETE /api/v1/profile/contacts/{id}
```

### Sessions
```
GET    /api/v1/sessions                # paginated list
GET    /api/v1/sessions/{id}           # session detail
GET    /api/v1/sessions/stats/weekly   # chart data
```

---

## 7. Database Schema (PostgreSQL — Supabase)

```sql
-- Users (managed by Supabase Auth)
-- Additional profile data:

CREATE TABLE profiles (
    id          UUID PRIMARY KEY REFERENCES auth.users(id),
    name        TEXT NOT NULL,
    phone       TEXT,
    age         INTEGER,
    blood_group TEXT,
    allergies   TEXT,
    location_permission BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE medical_conditions (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID REFERENCES profiles(id) ON DELETE CASCADE,
    condition  TEXT NOT NULL  -- 'diabetes', 'hypertension', 'heart_condition', etc.
);

CREATE TABLE emergency_contacts (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID REFERENCES profiles(id) ON DELETE CASCADE,
    name         TEXT NOT NULL,
    phone        TEXT NOT NULL,   -- E.164 format: +91XXXXXXXXXX
    relationship TEXT NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE sessions (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id              UUID REFERENCES profiles(id) ON DELETE CASCADE,
    start_time           TIMESTAMPTZ NOT NULL,
    end_time             TIMESTAMPTZ,
    duration_seconds     INTEGER,
    detection_mode       TEXT NOT NULL,   -- 'threshold' | 'hybrid'
    peak_state           TEXT,            -- 'Alert' | 'Mild Fatigue' | 'Fatigued'
    peak_score           FLOAT,
    avg_ear              FLOAT,
    avg_mar              FLOAT,
    perclos              FLOAT,
    blink_per_min        FLOAT,
    yawn_per_min         FLOAT,
    alert_triggered      BOOLEAN DEFAULT FALSE,
    alert_triggered_at   TIMESTAMPTZ,
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE alert_events (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID REFERENCES sessions(id) ON DELETE CASCADE,
    user_id     UUID REFERENCES profiles(id) ON DELETE CASCADE,
    triggered_at TIMESTAMPTZ NOT NULL,
    location_city TEXT,
    location_coords TEXT,
    contacts_notified INTEGER  -- count of WhatsApp messages sent
);
```

---

## 8. Backend Detection Handler (Pseudocode)

This shows how the **verified real** `features.py`, `tracker.py`, and (for hybrid) `predictor.py` are used inside the WebSocket handler. Threshold and hybrid modes branch early since they use different tracker classes and different summary logic.

```python
# app/routers/ws.py

from app.detection.threshold.features import calculate_ear as t_calculate_ear, ...
from app.detection.threshold.tracker import FatigueTracker
from app.detection.hybrid.features import calculate_ear as h_calculate_ear, ...
from app.detection.hybrid.tracker import Hybrid3StateTracker
from app.detection.hybrid.predictor import FatigueMLPredictor

# Loaded once at startup (hybrid only)
predictor = FatigueMLPredictor(
    model_path="detection/hybrid/models/fatigue_lightgbm_model_final.pkl",
    feature_columns_path="detection/hybrid/models/fatigue_lightgbm_feature_columns_final.pkl",
    threshold_path="detection/hybrid/models/fatigue_lightgbm_final_threshold.pkl",
)

@app.websocket("/ws/detect")
async def detection_ws(websocket: WebSocket, token: str, mode: str):
    user = verify_jwt(token)
    await websocket.accept()

    tracker = Hybrid3StateTracker() if mode == "hybrid" else FatigueTracker()
    session_location = None        # set once via a "location" message, reused for emergency
    fatigued_start_time = None
    emergency_sent = False

    async for message in websocket.iter_json():

        if message["type"] == "location":
            session_location = {"lat": message["lat"], "lng": message["lng"]}
            continue

        if message["type"] == "end_session":
            await save_session(user.id, tracker, mode)
            await send_fcm_push(user.id, "Session recorded")
            break

        if message["type"] == "no_face":
            if mode == "hybrid":
                tracker.update_no_face()   # threshold's FatigueTracker has no equivalent — just skip
            continue

        # Scale normalized landmarks to pixel space
        w, h = message["frame_width"], message["frame_height"]
        landmarks = [(x * w, y * h) for x, y in message["landmarks"]]

        left_eye  = get_points_by_index(landmarks, LEFT_EYE_IDX)
        right_eye = get_points_by_index(landmarks, RIGHT_EYE_IDX)
        mouth     = get_points_by_index(landmarks, MOUTH_IDX)

        ear   = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
        mar   = calculate_mar(mouth)
        tilt  = calculate_roll_tilt(landmarks[LEFT_EYE_OUTER], landmarks[RIGHT_EYE_OUTER])
        pitch = calculate_pitch(landmarks[CHIN_IDX], landmarks[FOREHEAD_IDX])
        # ⚠️ RESOLVED (was UNCLEAR): redness dropped for v1. Browser-only landmark detection has
        # no eye-pixel data to compute it from (calculate_eye_redness() needs actual cropped BGR
        # patches, not just landmark coordinates). redness stays 0.0 internally so the existing
        # compute_fatigue_score()/compute_rule_fatigue_score_exact() formulas need zero changes —
        # the 8% redness weight simply always contributes 0, redistributing weight implicitly
        # across the other 7 factors. Frontend hides the Redness row entirely rather than
        # displaying a static "0.000" (see DESIGN.md §3.4) — candidate v2 feature via canvas
        # pixel sampling, tracked as a known scope cut, not an oversight.
        redness = 0.0

        tracker.update_frame_metrics(ear=ear, mar=mar, tilt=tilt, redness=redness, pitch=pitch)

        await websocket.send_json({"type": "live", "ear": ear, "mar": mar, "tilt": tilt, "redness": redness})

        # ---- Summary branch — genuinely different per mode ----
        should_summarize = (
            tracker.is_time_to_summarize() if mode == "hybrid"
            else tracker.update_summary_if_needed()   # threshold computes AND returns bool in one call
        )

        if mode == "hybrid" and should_summarize:
            feature_dict = tracker.build_feature_dict(fps_value=30.0)  # fps from frame timing if tracked
            raw_model_prob = predictor.predict(feature_dict)
            tracker.update_summary(raw_model_prob)   # internally calls combine_rule_and_model()
            tracker.reset_window()

        # threshold mode: update_summary_if_needed() already updated tracker.latest_* internally

        if should_summarize:
            alarm = determine_alarm_state(tracker.latest_state)   # mirrors main.py's mild/fatigue alarm logic

            result = {"fatigue_score": tracker.latest_fatigue_score, "state": tracker.latest_state}
            if mode == "hybrid":
                result["raw_model_prob"] = tracker.latest_raw_model_probability
                result["display_prob"] = tracker.latest_display_probability

            await websocket.send_json({
                "type": "summary",
                "mode": mode,
                "summary": {
                    "avg_ear": tracker.latest_avg_ear, "avg_mar": tracker.latest_avg_mar,
                    "avg_tilt": tracker.latest_avg_tilt,
                    # avg_redness intentionally omitted — see §8 resolved decision
                },
                "indicators": {
                    "perclos": tracker.latest_perclos, "blink_per_min": tracker.latest_blink_rate,
                    "yawn_per_min": tracker.latest_yawn_rate, "nod_per_min": tracker.latest_nod_rate,
                    "eyes_closed_sec": tracker.continuous_eye_closure_sec,
                },
                "result": result,
                "alarm": alarm,
            })

            # ---- Emergency escalation — identical logic in both desktop main.py files ----
            if tracker.latest_state == "Fatigued":
                if fatigued_start_time is None:
                    fatigued_start_time = time.time()
                elapsed = time.time() - fatigued_start_time
                if elapsed >= EMERGENCY_COUNTDOWN_SEC and not emergency_sent:
                    contacts = await get_emergency_contacts(user.id)
                    loc = format_location(session_location)  # falls back to "Unknown City" if None,
                                                               # matching location_helper.py's default
                    await trigger_whatsapp_alert(contacts, loc)
                    await send_fcm_push(user.id, "Emergency alert triggered")
                    emergency_sent = True
                    await websocket.send_json({
                        "type": "emergency", "location": loc,
                        "contacts_notified": [c["name"] for c in contacts],
                        "message": "Emergency escalation triggered. WhatsApp alerts sent.",
                    })
            else:
                fatigued_start_time = None
```

---

## 9. Geolocation Handling (Replaces `get_location.html` + `location.json`)

The desktop flow requires a user to manually open `get_location.html`, click a button, and a `location.json` file gets downloaded for `location_helper.py` to read. **This manual step does not exist in the web app** — it's replaced entirely by the browser's native Geolocation API, called automatically from inside the React app itself.

```typescript
// hooks/useGeolocation.ts
export function useGeolocation(onLocation: (lat: number, lng: number) => void) {
  useEffect(() => {
    if (!navigator.geolocation) return;  // gracefully degrade — no location sent, backend defaults apply

    navigator.geolocation.getCurrentPosition(
      (pos) => onLocation(pos.coords.latitude, pos.coords.longitude),
      (err) => console.warn("Location permission denied or unavailable:", err.message),
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
    );
  }, []);
}
```

Called once when the Detection screen mounts. On success, sends a single `{"type": "location", lat, lng}` WebSocket message (§5.6) — not per-frame, since GPS coordinates don't need 30fps updates and a driver's location is effectively constant across an 8-second summary window.

**Permission denied case:** detection still proceeds normally. The emergency WhatsApp message falls back to the same default `location_helper.py` already returns when no `location.json` exists (`"Unknown City"`, empty coordinates) — so no new failure mode is introduced, just a less detailed alert message.

`location_helper.py` and `get_location.html` are **not ported to the backend at all** for the web platform — the backend's only job becomes formatting whatever `{lat, lng}` it received (or didn't) into the same dict shape `location_helper.py` already returns, so `whatsapp_notifier.py`'s message template needs zero changes.

---

## 10. Docker Container

```dockerfile
# Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./backend/

# Copy React build (produced by CI pipeline before Docker build)
COPY frontend/dist/ ./static/

# FastAPI serves React static build from /static
# React router uses HTML5 history — FastAPI catchall serves index.html

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

FastAPI mounts static files:
```python
app.mount("/", StaticFiles(directory="static", html=True), name="static")
```
All API routes at `/api/v1/*` are handled before the static mount catches them.

---

## 11. GitHub Actions CI/CD

```yaml
# .github/workflows/deploy.yml

on:
  push:
    branches: [main]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build React frontend
        run: |
          cd frontend
          npm ci
          npm run build
          cp -r dist ../frontend/dist

      - name: Build Docker image
        run: docker build -t fatigueguard:${{ github.sha }} .

      - name: Push to Azure Container Registry
        run: |
          az acr login --name fatigueguardacr
          docker tag fatigueguard:${{ github.sha }} fatigueguardacr.azurecr.io/fatigueguard:latest
          docker push fatigueguardacr.azurecr.io/fatigueguard:latest

      - name: Deploy to Azure Container Apps
        run: |
          az containerapp update \
            --name fatigueguard-app \
            --resource-group fatigueguard-rg \
            --image fatigueguardacr.azurecr.io/fatigueguard:latest
```

---

## 12. Environment Variables

### Backend (.env)
```
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...
JWT_SECRET=your-jwt-secret
TWILIO_ACCOUNT_SID=ACxxx
TWILIO_AUTH_TOKEN=xxx
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
FIREBASE_CREDENTIALS_JSON={"type":"service_account",...}
EMERGENCY_COUNTDOWN_SEC=60
```

### Frontend (.env)
```
VITE_API_BASE_URL=https://fatigueguard.azurecontainerapps.io
VITE_WS_URL=wss://fatigueguard.azurecontainerapps.io
VITE_SUPABASE_URL=https://xxx.supabase.co
VITE_SUPABASE_ANON_KEY=eyJ...
```

### Mobile (.env)
```
EXPO_PUBLIC_API_BASE_URL=https://fatigueguard.azurecontainerapps.io/api/v1
EXPO_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=eyJ...
```

---

## 13. Coding Rules

### Python (Backend)
- FastAPI with async handlers (`async def`)
- Pydantic v2 for all request/response models
- Type hints on every function signature
- `features.py` and `tracker.py` imported without modification — if a bug is found, fix it in the original file and copy again
- One `FatigueTracker` instance per WebSocket connection — never shared across connections
- Model loaded once at startup using `@app.on_event("startup")`, stored as app state
- All database calls through `supabase_client.py` service layer
- Errors: always raise typed `HTTPException`; WebSocket errors sent as `{"type": "error", "message": "..."}` then connection closed

### React Web (Frontend)
- TypeScript strict mode
- MediaPipe Face Landmarker initialized once in `useMediaPipe` hook, reused across frames
- WebSocket connection managed entirely in `useWebSocket` hook — no raw WebSocket calls in components
- Detection store (`useDetectionStore`) holds live metric values — updated by `useWebSocket`
- Dashboard components are purely reactive — read from store, render, no logic
- Alarm state driven by `alarm` field in WebSocket response — `useAlarm` hook plays/stops sounds
- No hardcoded colors — all from `constants/colors.ts`

### React Native (Mobile)
- Same rules as TECH_RULES.md v1.0 (TypeScript strict, Zustand, TanStack Query, expo-secure-store)
- "Open Web App" button: `Linking.openURL('https://fatigueguard.azurecontainerapps.io')`

---

## 14. Team Responsibilities

Three people, three platform layers — each with full, independent ownership end-to-end. No one is responsible for a subset of another person's layer; this keeps each contribution clean and individually verifiable for resumes/LinkedIn.

| Person | Owns | Primary Tech | Scope |
|--------|------|-------------|-------|
| **Yashwant Vadhan M** | Cloud + backend platform: FastAPI WebSocket handler, `detection_service.py` (wiring `features.py`/`tracker.py`/LightGBM into the API), auth + profile + session REST endpoints, database schema, Twilio + FCM alert pipeline, Docker containerization, Azure Container Apps deployment, CI/CD pipeline | Python, FastAPI, SQL, Supabase, Docker, Azure, GitHub Actions | Entire backend — API, database, infra, deployment |
| **M A Sushil Kumar** | Web frontend: landing page (hero, how-it-works, why-unique, team section), auth screens, launcher screen, live detection dashboard, MediaPipe Face Landmarker JS integration, WebSocket client hook, Web Audio alarm handling | React, TypeScript, Vite, MediaPipe JS | Entire web app — every screen, every browser-side feature |
| **Rithika G V** | Mobile app: all React Native screens (auth, dashboard, session detail, profile, settings), Zustand stores, API integration, push notification handling, "Open Web App" deeplink | React Native, Expo, TypeScript | Entire mobile app — every screen, every feature |

**Why this split works:** each person owns one full vertical slice of the platform (cloud/backend, web, mobile) rather than a horizontal fragment of one layer. This means each can describe their contribution as "I built X" rather than "I helped with part of X" — a stronger, more defensible claim for a resume or interview.

Naveena M S is not assigned a role in this phase of the platform expansion.

---

*End of TECH_RULES.md v2.0*
