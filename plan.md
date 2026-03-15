Web Interface Plan: Drosophila Brain Simulation

 Context

 We have a working embodied fly brain simulation (138,639 LIF neurons + MuJoCo physics body) that produces walking, escape flight, and NT
 logging. Currently it outputs video files and CSV logs — no real-time viewing. The goal is a web interface where users can:

 1. Watch the fly in a 3D room (Three.js)
 2. Switch to fly POV (MuJoCo render feed)
 3. Select scenarios (hunger, looming, etc)
 4. See real-time NT dashboard (GABA, ACh, dopamine, serotonin, etc)

 Tech Stack

 Already installed — no new dependencies needed:
 - Backend: FastAPI 0.129 + uvicorn + websockets 16.0
 - Frontend: Three.js (CDN) + vanilla JS (no build step)
 - Data: Pydantic models, JSON over WebSocket

 Architecture

 Simulation (Python)          Backend (FastAPI)           Frontend (Browser)
 ┌──────────────┐            ┌──────────────┐           ┌──────────────────┐
 │ BrainEngine  │  asyncio   │ /ws/sim      │  JSON/WS  │ Three.js room    │
 │ 138K neurons ├───queue───►│ broadcast    ├──────────►│ + fly avatar     │
 │ MuJoCo body  │            │              │           │ NT dashboard     │
 │ NT logging   │            │ /api/start   │◄──────────│ Scenario picker  │
 └──────────────┘            │ /api/stop    │   REST    │ Fly POV toggle   │
                             └──────────────┘           └──────────────────┘

 File Structure

 flywire-neuro/web/
 ├── backend/
 │   ├── main.py                 # FastAPI app, WebSocket, REST endpoints
 │   ├── simulation_bridge.py    # Runs simulation in thread, pushes to queue
 │   └── models.py               # Pydantic: SimFrame, Scenario, SimConfig
 │
 ├── frontend/
 │   ├── index.html              # Single page app
 │   ├── css/style.css           # Dark theme, dashboard layout
 │   └── js/
 │       ├── app.js              # Init, WebSocket, state management
 │       ├── room.js             # Three.js: room + furniture + fly
 │       ├── dashboard.js        # NT gauges, DN rates, behavior mode
 │       └── controls.js         # Scenario picker, play/pause, POV toggle
 │
 └── assets/
     └── (glTF models if needed, start with primitive geometries)

 Implementation Phases

 Phase 1: Backend skeleton (30 min)

 - main.py: FastAPI app with WebSocket endpoint /ws/sim
 - models.py: Pydantic SimFrame model (from Eon's mon_data structure)
 - simulation_bridge.py: Mock data generator (sin waves for NT levels)
 - Test: connect with wscat, verify JSON frames at 30Hz

 Phase 2: Frontend skeleton (1 hour)

 - index.html: Split layout — 3D viewport (left 70%) + dashboard (right 30%)
 - room.js: Three.js scene with box geometries (floor, walls, table, bed, fruits)
 - app.js: WebSocket client, receive frames, update state
 - dashboard.js: Simple bars for GF, P9, GABA, ACh levels
 - Test: open browser, see room + updating bars

 Phase 3: Connect real simulation (1 hour)

 - Modify simulation_bridge.py to import BrainEngine and run actual simulation
 - Run in background thread, push SimFrame to asyncio queue
 - Frontend receives real brain data
 - Add scenario picker (hunger→escape, sugar→feeding, etc)

 Phase 4: Polish (1 hour)

 - Fly avatar in Three.js (simple mesh moving by position data)
 - Wing animation when flight_state=FLYING
 - Behavior mode indicator (walking/escape/grooming/feeding/flight)
 - NT time-series chart (scrolling line graph)
 - Fly POV: stream MuJoCo render frames as base64 images

 Key Files to Reference

 - fly-brain-embodied/brain_monitor.py — mon_data structure, 30Hz update pattern
 - fly-brain-embodied/fly_embodied.py lines 927-1013 — mon_data fields
 - embodied/02_flight_escape_v2.py — our latest working simulation
 - fly-brain-embodied/brain_body_bridge.py — DN_NEURONS, DN_GROUPS, STIMULI

 Data Frame Structure (SimFrame)

 class SimFrame(BaseModel):
     t_ms: float
     phase: str                    # hunger, looming, free
     flight_state: str             # GROUNDED, TAKEOFF, FLYING, LANDING
     pos: list[float]              # [x, y, z] mm
     alt_mm: float

     # DN rates (0-1)
     dn_escape: float
     dn_forward: float
     dn_backward: float
     dn_turn_L: float
     dn_turn_R: float

     # Drive
     drive: list[float]            # [left, right]

     # NT populations (spike counts per interval)
     pam: int
     ppl1: int
     mbon_approach: int
     mbon_avoidance: int
     mbon_suppress: int
     serotonin: int
     octopamine: int
     gaba: int
     ach: int
     glut: int

     # Flight
     wing_freq: float
     behavior_mode: str            # walking, escape, grooming, feeding, flight

 ## Verification

 1. `cd web/backend && python main.py` → server starts on :8000
 2. Open browser → `http://localhost:8000` → 3D room visible
 3. Click "Start Simulation" → bars start updating
 4. Select "Escape" scenario → GF spikes, fly position changes
 5. NT dashboard shows real-time GABA/ACh/dopamine levels
