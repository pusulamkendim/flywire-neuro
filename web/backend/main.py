import os
os.environ['MUJOCO_GL'] = 'disabled'  # headless — no GL context needed

"""
FastAPI backend for the Drosophila brain simulation web interface.

Endpoints:
    GET  /                  → serves frontend
    GET  /api/scenarios     → list available scenarios
    POST /api/start         → start NT simulation (mock or real brain)
    POST /api/walk          → start CPG walking animation
    POST /api/stop          → stop current simulation
    WS   /ws/sim            → stream frames (SimFrame or walk poses)
"""

import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from models import SimConfig
from simulation_bridge import SimulationBridge, SCENARIOS

app = FastAPI(title="Drosophila Brain Simulation")

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Shared state
bridge = SimulationBridge()
walker = None  # lazy init
frame_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
connected_clients: list[WebSocket] = []


def _drain_queue():
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except asyncio.QueueEmpty:
            break


@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/api/scenarios")
async def get_scenarios():
    return [
        {"name": s["name"], "label": s["label"], "description": s["description"]}
        for s in SCENARIOS.values()
    ]


@app.post("/api/start")
async def start_sim(config: SimConfig):
    bridge.stop()
    walker.stop()
    await asyncio.sleep(0.1)
    _drain_queue()

    loop = asyncio.get_event_loop()
    duration = SCENARIOS.get(config.scenario, {}).get("duration_s", config.duration_s)
    bridge.start(config.scenario, duration, loop, frame_queue,
                 use_real_brain=config.use_real_brain)
    return {"status": "started", "scenario": config.scenario,
            "duration_s": duration, "real_brain": config.use_real_brain}


@app.post("/api/walk")
async def start_walk(duration_s: float = 5.0):
    global walker
    bridge.stop()
    if walker:
        walker.stop()
    await asyncio.sleep(0.1)
    _drain_queue()

    from walking_sim import WalkingBridge
    walker = WalkingBridge()
    loop = asyncio.get_running_loop()
    walker.start(loop, frame_queue, duration_s=duration_s)
    return {"status": "walking", "duration_s": duration_s}


@app.post("/api/stop")
async def stop_sim():
    bridge.stop()
    if walker:
        walker.stop()
    return {"status": "stopped"}


@app.websocket("/ws/sim")
async def ws_sim(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            data = await frame_queue.get()
            if data is None:
                await websocket.send_json({"event": "end"})
                continue

            # SimFrame (Pydantic) or dict (walking)
            if hasattr(data, 'model_dump'):
                payload = data.model_dump()
                payload["event"] = "frame"
            else:
                payload = data

            for client in list(connected_clients):
                try:
                    await client.send_json(payload)
                except Exception:
                    if client in connected_clients:
                        connected_clients.remove(client)
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
    except Exception:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
