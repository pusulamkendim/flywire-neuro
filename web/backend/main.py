import os
os.environ['MUJOCO_GL'] = 'disabled'

"""
FastAPI backend for the Drosophila brain simulation web interface.

Behavior simulations:
    POST /api/walk      → CPG tripod walking
    POST /api/groom     → antennal grooming
    POST /api/fly       → flight (takeoff + cruise + land)
    POST /api/feed      → foraging + feeding
    POST /api/escape    → startle escape run
    POST /api/backward  → moonwalk (backward + turn)
    POST /api/odor      → odor navigation (zigzag chemotaxis)
    POST /api/courtship → courtship song (wing extension + vibration)
    POST /api/stop      → stop any running simulation
"""

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from models import SimConfig
from simulation_bridge import SimulationBridge, SCENARIOS
from simulation_recorder import RecordingQueue, SimulationRecorder

app = FastAPI(title="Drosophila Brain Simulation")

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Shared state
bridge = SimulationBridge()
_active_sim = None   # currently running behavior sim
_brain = None        # interactive brain instance
simulation_recorder = SimulationRecorder()
frame_queue: asyncio.Queue = RecordingQueue(simulation_recorder, maxsize=200)
connected_clients: list[WebSocket] = []


def _drain_queue():
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except asyncio.QueueEmpty:
            break


async def _stop_all():
    global _active_sim, _brain
    simulation_recorder.request_stop()
    bridge.stop()
    if _active_sim:
        _active_sim.stop()
        _active_sim = None
    if _brain:
        _brain.stop()
        _brain = None
    await asyncio.sleep(0.1)
    _drain_queue()
    return simulation_recorder.finish("stopped")


async def _start_behavior(bridge_cls, **start_kwargs):
    global _active_sim
    await _stop_all()
    recording_id = simulation_recorder.start(
        bridge_cls.__name__, config=start_kwargs
    )
    _active_sim = bridge_cls()
    loop = asyncio.get_running_loop()
    try:
        _active_sim.start(loop, frame_queue, **start_kwargs)
    except Exception:
        simulation_recorder.finish("error")
        raise
    return recording_id


# --- Static ---
@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/api/scenarios")
async def get_scenarios():
    return [{"name": s["name"], "label": s["label"], "description": s["description"]}
            for s in SCENARIOS.values()]


@app.get("/api/episodes")
async def get_episodes():
    """List validated observable episode manifests."""
    from embodied_episode import list_episode_summaries
    return list_episode_summaries()


@app.get("/api/world/{world_id}")
async def get_world(world_id: str):
    """Return a validated millimetre-scale interactive world definition."""
    from world_config import WorldConfigError, load_world
    try:
        return load_world(world_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="World not found")
    except WorldConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/api/sensory-contract/{contract_id}")
async def get_sensory_contract(contract_id: str):
    """Return reusable environment-to-brain sensory channel definitions."""
    from world_config import WorldConfigError, load_sensory_contract
    try:
        return load_sensory_contract(contract_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Sensory contract not found")
    except WorldConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/api/neural-datasets")
async def get_neural_datasets():
    """List local response libraries and pending calibration plans."""
    from dataset_registry import list_dataset_summaries
    return list_dataset_summaries()


@app.get("/api/recordings")
async def get_recordings(limit: int = 100):
    """List compressed Start-to-Stop simulation recordings, newest first."""
    return simulation_recorder.list(limit=max(1, min(limit, 500)))


@app.get("/api/recordings/{recording_id}")
async def get_recording_manifest(recording_id: str):
    try:
        return simulation_recorder.get(recording_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Recording not found")


@app.get("/api/recordings/{recording_id}/data")
async def download_recording_data(recording_id: str):
    try:
        manifest = simulation_recorder.get(recording_id)
        path = simulation_recorder.data_path(recording_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Recording not found")
    if manifest.get("status") == "running":
        raise HTTPException(status_code=409, detail="Recording is still running")
    return FileResponse(
        str(path),
        media_type="application/gzip",
        filename=f"{recording_id}.jsonl.gz",
    )


# --- NT Simulation (mock/real brain) ---
@app.post("/api/start")
async def start_sim(config: SimConfig):
    await _stop_all()
    recording_id = simulation_recorder.start(
        f"scenario-{config.scenario}", config=config.model_dump()
    )
    loop = asyncio.get_running_loop()
    duration = SCENARIOS.get(config.scenario, {}).get("duration_s", config.duration_s)
    try:
        bridge.start(config.scenario, duration, loop, frame_queue,
                     use_real_brain=config.use_real_brain)
    except Exception:
        simulation_recorder.finish("error")
        raise
    return {
        "status": "started",
        "scenario": config.scenario,
        "recording_id": recording_id,
    }


# --- Behavior Simulations ---
@app.post("/api/walk")
async def start_walk(duration_s: float = 5.0):
    from walking_sim import WalkingBridge
    await _start_behavior(WalkingBridge, duration_s=duration_s)
    return {"status": "walking"}


@app.post("/api/walk-preview/{route_name}")
async def start_walk_preview(route_name: str):
    """Replay measured walking paths with gait phase retargeted to NeuromechFly."""
    from walking_data_preview import (
        AVAILABLE_ROUTES,
        CHAIN_ROUTE_NAME,
        WalkingDataPreviewBridge,
        load_preview_cache,
    )
    if route_name not in AVAILABLE_ROUTES:
        raise HTTPException(status_code=404, detail="Walking preview route not found")
    await _start_behavior(WalkingDataPreviewBridge, route_name=route_name)
    cached = load_preview_cache()
    if route_name == CHAIN_ROUTE_NAME:
        segment_count = 12
        trajectory_id = "12-measured-walking-snippets"
    else:
        segment_count = 1
        trajectory_id = cached["previews"][route_name]["trajectory_id"]
    return {
        "status": "walking_data_preview",
        "route": route_name,
        "trajectory_id": trajectory_id,
        "segment_count": segment_count,
        "sample_hz": 30,
    }


@app.post("/api/walk-explore")
async def start_walk_explore():
    """Start continuous boundary-aware coverage using measured walk clips."""
    from exploration_walking import ExplorationWalkingBridge
    await _start_behavior(ExplorationWalkingBridge)
    return {
        "status": "walking_data_explore",
        "strategy": "measured_maneuver_serpentine_coverage",
        "maneuver_source": "walking_imitation_hdf5",
        "stop": "global_stop",
    }

@app.post("/api/groom")
async def start_groom():
    from grooming_sim import GroomingBridge
    await _start_behavior(GroomingBridge)
    return {"status": "grooming"}

@app.post("/api/fly")
async def start_fly():
    from flying_sim import FlyingBridge
    await _start_behavior(FlyingBridge)
    return {"status": "flying"}


@app.post("/api/flight-preview/{route_name}")
async def start_flight_preview(route_name: str):
    """Replay a real measured flight-imitation body trajectory."""
    from flight_data_preview import AVAILABLE_ROUTES, FlightDataPreviewBridge, load_preview_cache
    if route_name not in AVAILABLE_ROUTES:
        raise HTTPException(status_code=404, detail="Flight preview route not found")
    await _start_behavior(FlightDataPreviewBridge, route_name=route_name)
    route = load_preview_cache()["previews"][route_name]
    return {
        "status": "flight_data_preview",
        "route": route_name,
        "trajectory_id": route["trajectory_id"],
        "playback_rate": 0.5,
        "segment_count": len(route.get("segments", [])) or 1,
    }


@app.post("/api/flight-explore")
async def start_flight_explore():
    """Start continuous coverage assembled from measured flight maneuvers."""
    from exploration_flight import ExplorationFlightBridge
    await _start_behavior(ExplorationFlightBridge)
    return {
        "status": "flight_explore",
        "strategy": "measured_maneuver_serpentine_coverage",
        "maneuver_source": "flight_imitation_hdf5",
        "landing": "on_demand",
    }


@app.post("/api/flight-explore/land")
async def land_flight_explore():
    """Request a leg-extension descent from the active exploration flight."""
    from exploration_flight import ExplorationFlightBridge
    if not isinstance(_active_sim, ExplorationFlightBridge):
        raise HTTPException(status_code=409, detail="Exploration flight is not active")
    _active_sim.request_land()
    return {"status": "landing_requested"}

@app.post("/api/feed")
async def start_feed():
    from feed_sim import FeedBridge
    await _start_behavior(FeedBridge)
    return {"status": "feeding"}

@app.post("/api/escape")
async def start_escape():
    from escape_sim import EscapeBridge
    await _start_behavior(EscapeBridge)
    return {"status": "escaping"}

@app.post("/api/backward")
async def start_backward():
    from backward_sim import BackwardBridge
    await _start_behavior(BackwardBridge)
    return {"status": "backward"}

@app.post("/api/odor")
async def start_odor():
    from odor_sim import OdorBridge
    await _start_behavior(OdorBridge)
    return {"status": "odor_tracking"}

@app.post("/api/walk_fb")
async def start_walk_fb():
    from walk_flybody_sim import WalkFlybodyBridge
    await _start_behavior(WalkFlybodyBridge)
    return {"status": "walk_flybody"}

@app.post("/api/courtship")
async def start_courtship():
    from courtship_sim import CourtshipBridge
    await _start_behavior(CourtshipBridge)
    return {"status": "courtship"}


@app.post("/api/episode/{episode_id}")
async def start_episode(episode_id: str):
    """Play a unified multi-behavior episode on the shared WebSocket."""
    from embodied_episode import EmbodiedEpisodeBridge, load_episode_manifest
    try:
        manifest = load_episode_manifest(episode_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Episode not found")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    await _start_behavior(EmbodiedEpisodeBridge, episode_id=episode_id)
    return {
        "status": "episode_started",
        "episode_id": episode_id,
        "title": manifest["title"],
    }


@app.post("/api/brain")
async def start_brain(body: dict = None):
    global _brain, _active_sim
    await _stop_all()
    from brain_interactive import InteractiveBrain
    from digital_life import DigitalLifeBridge
    loop = asyncio.get_running_loop()
    request = body or {}
    initial = request.get('stimuli', [])
    sync_to_brain_time = request.get('sync_to_brain_time', False) is True
    recording_id = simulation_recorder.start(
        "digital-life", config={
            "initial_stimuli": initial,
            "sync_to_brain_time": sync_to_brain_time,
        }
    )
    _active_sim = DigitalLifeBridge(sync_to_brain_time=sync_to_brain_time)
    try:
        _active_sim.start(loop, frame_queue)
        _brain = InteractiveBrain()
        _brain.start(
            loop,
            frame_queue,
            initial_stimuli=initial,
            motor_sink=_active_sim,
        )
    except Exception:
        simulation_recorder.finish("error")
        raise
    return {
        "status": "brain_started",
        "initial_stimuli": initial,
        "body_runtime": "digital_life_v1",
        "timing_mode": "brain_time_sync" if sync_to_brain_time else "wall_time",
        "recording_id": recording_id,
        "stop": "global_stop",
    }


@app.get("/api/walk_cache")
async def get_walk_cache():
    """Return cached walk frames for brain-driven animation."""
    import json
    cache_path = Path(__file__).parent / 'walk_cache' / 'walk_5.0s.json'
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {"geom_names": [], "frames": []}


@app.post("/api/stop")
async def stop_sim():
    recording_id = await _stop_all()
    return {"status": "stopped", "recording_id": recording_id}


# --- WebSocket ---
@app.websocket("/ws/sim")
async def ws_sim(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)

    # Task for receiving commands from client
    async def receive_commands():
        try:
            while True:
                msg = await websocket.receive_text()
                import json
                cmd = json.loads(msg)
                if cmd.get('cmd') == 'set_stimuli' and _brain:
                    stimuli = cmd.get('stimuli', [])
                    simulation_recorder.record({
                        "event": "stimulus_command",
                        "stimuli": stimuli,
                        "source": "websocket",
                    })
                    _brain.set_stimuli(stimuli)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    # Task for sending frames to client
    async def send_frames():
        try:
            while True:
                data = await frame_queue.get()
                if data is None:
                    await websocket.send_json({"event": "end"})
                    continue
                payload = data.model_dump() if hasattr(data, 'model_dump') else data
                if "event" not in payload:
                    payload["event"] = "frame"
                for client in list(connected_clients):
                    try:
                        await client.send_json(payload)
                    except Exception:
                        if client in connected_clients:
                            connected_clients.remove(client)
        except Exception:
            pass

    # Run both tasks concurrently
    try:
        await asyncio.gather(receive_commands(), send_frames())
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
