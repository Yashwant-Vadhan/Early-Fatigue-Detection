# PRD — FatigueGuard Platform
**Product Requirements Document**
**Version:** 2.0
**Project:** Early Fatigue Detection System — Full Platform (Web + Mobile + Backend)
**Authors:** Yashwant Vadhan M, M A Sushil Kumar, Naveena M S, Rithika G V
**Mentor:** Dr. P. AnandhaKumar
**Status:** Pre-Development

---

## 1. Overview

### 1.1 Problem Statement
The Early Fatigue Detection System (desktop) detects driver fatigue and sends Twilio WhatsApp alerts. It works, but it is locked to a local machine, hardcodes emergency contacts in source code, stores no session history, and is inaccessible to anyone without a Python environment.

### 1.2 Solution
**FatigueGuard** is a three-part platform:

| Component | What it is |
|-----------|-----------|
| **Web App** | Browser-accessible fatigue detection — any device with a camera. Landing page + live detection dashboard. Hosted on Azure. |
| **Backend API** | FastAPI server running on Azure Container Apps. Hosts the LightGBM model, processes landmarks, manages users, sessions, and emergency alerts. |
| **Mobile App** | React Native companion app. Session history, analytics, emergency contacts, profile management. Syncs with web sessions. |

A user creates one account. Their emergency contacts, medical profile, and session history are shared seamlessly between the web app and mobile app.

### 1.3 What Makes This Unique
- The core detection engine (`features.py`, `tracker.py`, LightGBM model) runs on a cloud server — the same logic as the desktop system, no reimplementation
- Facial landmark extraction runs entirely in the browser using MediaPipe Face Landmarker WASM — no video ever leaves the device
- Works on any device with a camera: laptop, desktop, tablet, mobile browser
- Emergency contacts managed from the mobile app are automatically used by the web detection system for Twilio alerts
- One account, three surfaces: browser detection, mobile history, downloadable desktop app

### 1.4 Target Users
- Primary: Individual drivers, long-haul truckers, night-shift vehicle operators
- Secondary: Fleet safety supervisors
- Tertiary: Family members listed as emergency contacts

---

## 2. Goals

| Goal | Metric |
|------|--------|
| Any device with a camera can run fatigue detection via browser | Works on Chrome/Firefox/Safari — Android, iOS, Windows, Mac |
| Session history synced from web → mobile within 5 seconds of session end | Latency < 5s on 4G |
| Emergency contacts set in mobile app are used in web detection alerts | No manual config needed anywhere |
| Landing page clearly communicates project value | Visitor can understand the system in under 30 seconds |

---

## 3. Scope — Version 1.0

### 3.1 In Scope
**Web App**
- Public landing page (project overview, how it works, team, GitHub link)
- Sign in / create account (shared auth with mobile app)
- Launcher screen (select Hybrid or Threshold model)
- Live detection dashboard (webcam feed + real-time metrics)
- Emergency alert modal (browser equivalent of Tkinter popup)
- Browser audio alarm (mild + fatigue states)
- Twilio WhatsApp alert on emergency escalation
- Session auto-saved to backend on session end
- Link to download mobile app

**Backend API**
- Auth endpoints (register, login, refresh, logout)
- User profile + emergency contacts CRUD
- WebSocket endpoint for real-time landmark → metric processing
- LightGBM and Threshold model inference
- Session record storage and retrieval
- Twilio WhatsApp notification trigger
- Weekly analytics aggregation

**Mobile App**
- Login / register (shared auth)
- Session history log with filtering
- Session detail screen (metrics, state timeline, alert info)
- Weekly dashboard (trend chart, state distribution)
- Profile management (personal info, medical, emergency contacts)
- Settings (theme, language, notifications)
- "Open Web App" button (shareable link)
- Push notification when emergency alert fires

### 3.2 Out of Scope — Version 1.0
- Live session streaming to mobile while session is running on web (v2.0)
- Fleet management / multi-driver monitoring (v2.0)
- Mobile app as the detection device using phone camera (v2.0)
- AI-driven fatigue prediction from historical patterns (v2.0)
- Emergency services (100/108) direct integration (v3.0)

---

## 4. Web App — Feature Details

### 4.1 Landing Page
| Section | Content |
|---------|---------|
| Hero | Project name, tagline, "Try Now" CTA (→ sign in), "Download App" CTA |
| What is this | 2-3 sentence explanation + 3 key stats (metrics tracked, detection modes, alert speed) |
| How it works | 3-step visual: Open link → Sign in → Start session |
| Why it's unique | MediaPipe in browser (no video upload), LightGBM on cloud, dual detection modes, cross-device |
| Team | 4 member cards with name, role, contribution |
| GitHub | Link to repo |
| Footer | Sign In link + Download App link |

### 4.2 Authentication (Web)
- Sign In: email + password
- Create Account: name, email, phone, password
- Shared JWT tokens with mobile app — same account works everywhere
- "Forgot Password" via email OTP

### 4.3 Launcher Screen (Post Sign-In)
- Visually mirrors the Tkinter launcher: dark background, card, two mode buttons
- "Hybrid Model" (green button) — uses LightGBM
- "Threshold Model" (blue button) — uses rule-based thresholds
- User's name shown: "Welcome back, [Name]"
- Short description of each model for user clarity

### 4.4 Live Detection Dashboard
This is the core screen. Layout mirrors the desktop banner: webcam feed left, metrics panel right.

**Left panel — Camera Feed**
- `<video>` element showing live webcam (mirrored)
- Canvas overlay drawing facial landmarks (green dots on eyes, blue on mouth, yellow line between eyes)
- State overlay text on camera: "FATIGUE DETECTED!" in red when fatigued
- Countdown text: "Emergency escalation in: Xs"
- "Stop Alarm" button replaces keyboard `S` key

**Right panel — Fatigue Dashboard**
Exact metric layout from the desktop banner:
- Live Metrics: EAR, MAR, Head Tilt (Redness dropped for v1 — see §8)
- 8-Second Summary: Avg EAR, Avg MAR, Avg Tilt (Avg Redness dropped for v1 — see §8)
- Fatigue Indicators: PERCLOS, Blink/min, Yawn/min, Nod/min, Eyes Closed duration
- Overall Result: Fatigue Score (large), State badge (Alert / Mild Fatigue / Fatigued)
- **Hybrid mode only:** two additional rows — Display Probability and Raw Model Probability — matching fields the desktop hybrid dashboard shows that threshold's dashboard does not

**Session controls**
- "End Session" button → saves session → shows summary modal → redirects to launcher
- Session timer (top right)

### 4.5 Emergency Escalation (Web)
- When state is "Fatigued" for > 60 seconds without alarm being stopped:
  - Browser plays fatigue alarm sound (Web Audio API)
  - Countdown renders on screen
  - At 0: Backend triggers Twilio WhatsApp to all emergency contacts
  - Browser shows modal (equivalent of Tkinter popup): location info, map link, contacts notified
  - Mobile app receives FCM push notification

### 4.6 Session End
- On "End Session" or browser tab close:
  - Final session summary POSTed to backend
  - Modal shown: duration, peak state, peak score, alert triggered Y/N
  - "View in App" deeplink

---

## 5. Mobile App — Feature Details

*(See DESIGN.md for full screen specifications)*

| Screen | Key Features |
|--------|-------------|
| Dashboard | Session history log, weekly trend chart, summary cards |
| Session Detail | State timeline, metric grid, alert info |
| Profile | Personal info, medical details, emergency contacts (up to 3) |
| Settings | Theme, language, notifications |
| All screens | "Open Web App" — prominent button linking to web detection |

---

## 6. User Flows

### 6.1 New User — Web
```
Landing Page → "Try Now"
  → Create Account (name, email, phone, password)
    → Email OTP Verification
      → Profile Setup (emergency contacts, medical info, location permission)
        → Launcher Screen (model selection)
          → Live Detection Dashboard
            → [Session ends] → Session Summary Modal
              → "Download the app" CTA
```

### 6.2 Returning User — Web
```
Landing Page → "Sign In"
  → Launcher Screen
    → Live Detection Dashboard
```

### 6.3 Mobile — Session Sync
```
Web session ends
  → Backend saves session record
    → FCM push to mobile: "Session recorded"
      → Mobile dashboard shows new session
```

### 6.4 Emergency Alert
```
Fatigued state > 60 seconds, alarm not stopped
  → Backend fetches emergency contacts for this user
    → Twilio WhatsApp sent to each contact
      → Backend logs alert event
        → FCM push to mobile: "Emergency alert was triggered"
          → Browser shows emergency modal with location + contacts notified
```

---

## 7. Non-Functional Requirements

| Category | Requirement |
|----------|-------------|
| Webcam access | Camera accessed via browser `getUserMedia` — no video ever sent to server |
| Landmark privacy | Only 478 (x,y) coordinate pairs sent per frame — not image pixels |
| WebSocket latency | Round-trip landmark → score < 100ms on standard broadband |
| Browser support | Chrome 90+, Firefox 88+, Safari 15+, Edge 90+ |
| Device support | Any device with camera and supported browser: Android, iOS, Windows, Mac |
| Auth security | JWT access tokens (15min expiry) + refresh tokens (30 days); stored in httpOnly cookies on web |
| HTTPS | All traffic over HTTPS only; no HTTP in production |
| Container | Single Docker container; Azure Container Apps deployment |

---

## 8. Resolved Decisions (from source code review)

These were open questions in v2.0 — now resolved after reviewing the actual `threshold_model/` and `hybrid_model/` source:

1. **Single endpoint, mode param — confirmed.** `/ws/detect?mode=threshold|hybrid` on one endpoint. Threshold and hybrid use genuinely different tracker classes (`FatigueTracker` vs `Hybrid3StateTracker`) and different summary payloads server-side, but share the same connection/auth flow.
2. **User association** — JWT passed as a query param on WebSocket connect, verified before `accept()`.
3. **Redness** — ✅ Resolved: dropped entirely for v1, not shown as a zeroed stat. `calculate_eye_redness()` needs actual cropped pixel data from the eye region, which a landmarks-only browser pipeline doesn't have. Rather than display a permanently-static "0.000" (which reads as broken to anyone watching closely), the Redness row is hidden from the dashboard entirely. The fatigue score formula needs no changes — `redness` stays internally `0.0`, so its 8% weight in `compute_fatigue_score()`/`compute_rule_fatigue_score_exact()` simply contributes nothing, with the other 7 factors implicitly carrying the full weight. Tracked as a deliberate v1 scope cut and a clean v2 candidate feature (canvas-based pixel sampling), not an oversight.
4. **WebSocket disconnect handling** — Auto-reconnect with session state preserved is the goal, but the tracker's in-memory state lives only on the backend connection object. If the connection drops, a simple v1 approach is: client shows "Reconnecting..." and starts a **new** tracker on reconnect (some data loss in the gap is acceptable for a demo-stage product); true mid-session resume is a v2 enhancement.

## 9. Location Handling — Confirmed Approach

The original desktop system requires manually running `get_location.html`, clicking a button, and placing a downloaded `location.json` next to the script for `location_helper.py` to read. **This manual step is fully eliminated in the web app.** The browser's native Geolocation API is called automatically when the Detection screen loads — the user sees a standard one-time permission prompt ("fatigueguard.app wants your location"), and from then on their coordinates are available to the session without any file, download, or extra action. If permission is denied, detection still works normally; only the emergency WhatsApp message loses location detail, falling back to the same default the desktop app already uses when no location file exists.

---

*End of PRD v2.0*
