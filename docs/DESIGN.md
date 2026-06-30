# DESIGN.md — FatigueGuard Platform
**UI/UX Design Specification — Web App + Mobile App**
**Version:** 2.0

---

## 1. Design Philosophy

FatigueGuard must feel **serious, trustworthy, and clinical-grade** — this is safety software, not a lifestyle app. The desktop Tkinter system's dark theme (`#0f172a` background, cyan accents) is the brand anchor across all three surfaces (desktop, web, mobile). A user should recognize the dashboard whether they're looking at the original Tkinter window, the browser, or their phone.

Design priorities, in order: **clarity of critical data → trust → polish**. A fatigue score must be unmistakable even in a quick glance — this is not a place for subtle UI.

---

## 2. Shared Design System

This token set is identical across web and mobile — implemented as `colors.ts` in both `frontend/` and `mobile/`.

### 2.1 Core Palette

| Token | Hex | Usage |
|-------|-----|-------|
| `color-bg-primary` | `#0f172a` | Main background |
| `color-bg-surface` | `#1e293b` | Cards, panels, modals |
| `color-bg-elevated` | `#253248` | Inputs, hover states |
| `color-accent-cyan` | `#22d3ee` | Brand accent, links, primary highlights |
| `color-accent-cyan-dim` | `#0e7490` | Borders, secondary accents |
| `color-text-primary` | `#f8fafc` | Headlines, body |
| `color-text-secondary` | `#94a3b8` | Captions, labels |
| `color-text-muted` | `#475569` | Disabled, metadata |

### 2.2 Fatigue State Colors (must match desktop exactly)

| State | Hex | Source |
|-------|-----|--------|
| Alert | `#22c55e` | Green — matches "State: Alert" in banner |
| Mild Fatigue | `#f59e0b` | Amber — matches "State: Mild Fatigue" |
| Fatigued | `#ef4444` | Red — matches "State: Fatigued" |
| No Data | `#64748b` | Grey — idle state |

### 2.3 Mode Colors (from launcher banner)

| Mode | Hex |
|------|-----|
| Hybrid Model | `#16a34a` (green, hover `#15803d`) |
| Threshold Model | `#2563eb` (blue, hover `#1d4ed8`) |
| Exit / Destructive | `#dc2626` (hover `#b91c1c`) |

### 2.4 Typography

| Role | Font | Weight | Web Size | Mobile Size |
|------|------|--------|----------|-------------|
| Display | Inter | 700 | 34px | 28sp |
| Heading 1 | Inter | 600 | 24px | 22sp |
| Heading 2 | Inter | 600 | 20px | 18sp |
| Body | Inter | 400 | 16px | 15sp |
| Caption | Inter | 400 | 13px | 12sp |
| Metric Value | Inter | 700 | 28px | 32sp |
| Metric Label | Inter | 400 | 13px | 12sp |

### 2.5 Spacing & Radius

- Base unit: 4px/4dp
- Card radius: 16px
- Button radius: 12px
- Input radius: 10px
- Card padding: 20px web / 16dp mobile

---

## 3. Web App — Screens

### 3.1 Landing Page

**Layout:** Full-width sections, max content width 1200px, centered.

**Hero Section**
- Background: `color-bg-primary` with subtle radial cyan glow behind headline
- Headline: "Catch Fatigue Before It Catches You" (Display, white)
- Subheadline: "Real-time driver fatigue detection — right from your browser." (Body, secondary)
- Two CTAs side by side: "Try Now" (primary cyan button → sign in) and "Download App" (ghost button)
- Below CTAs: small live demo GIF/video loop of the dashboard in action (autoplay, muted)

**What Is This Section**
- Centered heading: "What is FatigueGuard?"
- Description (adapted from the project's own README, so messaging stays consistent with how the team already describes it): "Driver fatigue is one of the major causes of road accidents. FatigueGuard monitors facial behavior in real time and detects early signs of drowsiness — entirely from your browser, no install required."
- 3-stat row: "8 Metrics Tracked" | "2 Detection Modes" | "< 100ms Response Time"

**How It Works Section**
- 3-step horizontal layout (stacks vertically on mobile breakpoint)
- Step 1: "Sign In" — icon: user
- Step 2: "Choose a Model" — icon: sliders
- Step 3: "Start Monitoring" — icon: camera
- Each step: icon in cyan circle, short title, 1-line description

**Why It's Unique Section**
- 2x2 grid of feature cards:
  - "Browser-Based Detection" — MediaPipe runs locally, no video upload
  - "Cloud ML Inference" — LightGBM model on Azure
  - "Cross-Device" — works on laptop, tablet, or phone
  - "Connected Ecosystem" — synced with mobile app automatically

**Team Section**
- Heading: "Built By"
- 4-column card grid (responsive to 2x2 on tablet, 1-column on mobile)
- Each card: avatar/initial circle, name, role tag, one-line contribution
- Example: "Yashwant Vadhan M" / "Cloud & Backend" / "Built the FastAPI detection engine and Azure deployment pipeline"

**GitHub + Footer**
- "View Source on GitHub" — ghost button with GitHub icon
- Footer: Sign In | Download App | Mentor credit (Dr. P. AnandhaKumar) | Anna University, MIT Campus

### 3.2 Auth Screens (Web)
Centered card on `color-bg-primary`, matching mobile auth visually:
- Sign In: email, password, "Sign In" button, "Forgot Password?" link, "Create Account" link
- Create Account: name, email, phone, password, confirm password
- OTP: 6-box input, resend timer
- Profile Setup: same fields as mobile (medical, emergency contacts, location permission) — laid out in a 2-column form on desktop widths

### 3.3 Launcher Screen (Web)
Direct visual port of the Tkinter launcher banner:
- Centered card on dark background
- "Welcome back, [Name]" replaces generic title
- Subtitle: "Choose a detection mode to continue"
- Two large buttons side by side: "Hybrid Model" (green) / "Threshold Model" (blue)
- Below buttons: one-line description of each mode appears on hover/focus
- Small "Sign Out" link top right

### 3.4 Live Detection Dashboard (Web)
**This is the centerpiece screen — must closely mirror Figures 1–6 from the project banners.**

On mount, before the camera starts: browser shows two native permission prompts in sequence — camera access, then location access. Location is optional; a small dismissible toast ("Location helps emergency alerts include your position — you can continue without it") should explain why it's asked, since an unexplained second permission prompt right after the camera one can read as suspicious to a first-time user.

Two-column layout, matching desktop aspect ratio:

**Left column (≈65% width) — Camera View**
- `<video>` element, mirrored, rounded corners (16px)
- `<canvas>` overlay on top, drawing:
  - Green dots on eye landmarks
  - Blue dots on mouth landmarks
  - Yellow horizontal line between eyes (tilt reference)
- Top-left overlay text (only while no alarm active): Live EAR / MAR / Tilt in green monospace, matching desktop banner style
- When fatigue alarm active: full-width red banner overlay "FATIGUE DETECTED!" + "Emergency escalation in: Xs" + "Stop Alarm" button (replaces keyboard `S`)
- Session timer, top-right corner

**Right column (≈35% width) — Fatigue Dashboard panel**
Exact section order from the banner, in a dark card. Two pipelines (threshold/hybrid) render slightly different panels — confirmed from the real `ui.py` files in each:
1. "FATIGUE DASHBOARD" header, cyan, bold
2. **Live Metrics**: EAR, MAR, Head Tilt — three rows only. **Redness is hidden in v1**, not shown as a zeroed stat (decision finalized — see PRD §8). Showing a permanently-static "Redness: 0.000" would read as broken to anyone watching closely; cleaner to omit the row entirely and reintroduce it as a clearly-new v2 feature later.
3. **8-Second Summary**: Avg EAR, Avg MAR, Avg Tilt — three rows only, same reasoning as above (no Avg Redness row in v1)
4. **Fatigue Indicators**: PERCLOS, Blink/min, Yawn/min, Nod/min, Eyes Closed — cyan subheading
5. **Overall Result**: Fatigue Score (large, 28px bold, yellow), State badge (color-coded pill: green/amber/red)
6. **Hybrid mode only — two extra rows here:** "Display Prob" and "Raw Model Prob", smaller text, muted grey for Raw Model Prob (matching the dimmer styling in the desktop hybrid dashboard). Threshold mode does not render these two rows at all — not greyed out, simply absent, matching the real difference between `threshold_model/ui.py` and `hybrid_model/ui.py`
7. Footer: "End Session" button (ghost, red border)

### 3.5 Emergency Modal (Web)
Replaces the Tkinter `messagebox.showwarning`:
- Centered modal, red-tinted border, cannot be dismissed by clicking outside
- Warning icon (triangle) + "Emergency Alert" heading
- Body text: same structure as Tkinter popup — location, coordinates, map link, contacts notified
- Single "OK" button (cyan) to dismiss and reset session

### 3.6 Session Summary Modal (Web)
Shown when user clicks "End Session":
- Duration, peak state badge, peak fatigue score, whether alert was triggered
- "View in App" button (deeplinks to mobile, or shows QR if on desktop browser)
- "Start New Session" button → back to Launcher

---

## 4. Mobile App — Screens

*(Same as DESIGN.md v1.0 — retained for reference. Web is the new primary detection surface; mobile is the analytics + profile companion.)*

### 4.1 Navigation
```
Auth Stack: Splash → Login/Register → OTP → Profile Setup
Main Stack (bottom tabs): Dashboard | Profile | Settings
```

### 4.2 Dashboard Screen
- Summary cards row: This Week sessions, Peak State, Alerts Triggered
- Weekly trend bar chart (color-coded by score range)
- Recent sessions list (date, duration, peak state badge, score)
- **New:** Prominent "Start a Session" card at top → opens web app via `Linking.openURL()`, since mobile itself does not run detection in v1

### 4.3 Session Detail Screen
- State timeline (color blocks across session duration)
- Metric grid: Peak Score, Avg EAR, Avg MAR, PERCLOS, Blink/min, Yawn/min
- Alert banner if emergency was triggered, with timestamp and contacts notified

### 4.4 Profile Screen
- Personal info, medical history chips, emergency contacts (up to 3, editable)
- These contacts are the same ones used by the web app's Twilio alerts — call this out in the UI with a small note: "Used for emergency alerts during detection sessions"

### 4.5 Settings Screen
- Theme, language, notification toggles
- "Open Web App" button — large, cyan, near top of settings (this is the primary way users start a detection session)

---

## 5. Component Library (Shared Patterns)

### 5.1 Buttons
- Primary: cyan bg, dark text, 12px radius, 52px height (mobile) / 48px (web)
- Mode buttons (Hybrid/Threshold): colored per §2.3, bold white text, large touch target
- Ghost: transparent bg, cyan border + text
- Destructive: red bg, white text — used only for Stop Alarm, End Session, Delete Account

### 5.2 State Badge
Pill shape, 15% opacity background of state color, full-opacity text and border — identical on web and mobile.

### 5.3 Metric Display
Label (small, secondary) above value (large, bold, primary or state-colored) — consistent across desktop banner, web dashboard, and mobile session detail.

### 5.4 Cards
`color-bg-surface` background, 16px radius, no border, soft shadow `0 4px 12px rgba(0,0,0,0.35)` on web; mobile uses elevation 2.

---

## 6. Responsive Behavior (Web)

| Breakpoint | Layout change |
|-----------|---------------|
| > 1024px | Full two-column detection dashboard (camera + panel side by side) |
| 768–1024px | Camera on top, dashboard panel below, full width |
| < 768px | Single column throughout; landing page sections stack; team grid becomes 2-column then 1-column |

The detection dashboard must remain usable on a tablet or large phone browser, since "any device with a camera" is a core promise.

---

## 7. Iconography
Lucide icons throughout (web and mobile), consistent stroke weight:
- `camera` — detection / launcher
- `activity` — dashboard / metrics
- `alert-triangle` — emergency / fatigue alert
- `user` — profile
- `settings` — settings
- `phone-call` — emergency contacts
- `github` — repo link
- `external-link` — "Open Web App" from mobile

---

*End of DESIGN.md v2.0*
