/**
 * App: WebSocket, stimulus control, brain simulation interface.
 */

let ws = null;
let frameCount = 0;
let lastFpsTime = performance.now();
let fpsDisplay = 0;
let simStartWall = 0;
let totalFrames = 0;
let lastBehavior = 'walk';
let activeEpisode = null;
let activeEpisodeBehavior = null;
let latestRecordingId = null;
let activeBrainTimeSync = false;

// Active stimuli (multiple can be on)
let activeStimuli = new Set();

window.addEventListener('DOMContentLoaded', async () => {
    Room.init(document.getElementById('three-canvas'));
    Dashboard.init();
    BrainVis.init();
    Controls.init();
    connectWebSocket();
    initStimulusButtons();
    refreshRecordings();
});

async function refreshRecordings() {
    try {
        const response = await fetch('/api/recordings?limit=100');
        if (!response.ok) throw new Error('recordings request failed');
        const recordings = await response.json();
        document.getElementById('record-count').textContent = recordings.length;
        const latest = recordings[0];
        latestRecordingId = latest ? latest.id : null;
        document.getElementById('record-latest').textContent = latest
            ? `${latest.simulation_type} · ${latest.status}` : '—';
        document.getElementById('record-events').textContent = latest
            ? latest.record_count : '0';
        document.getElementById('record-download').disabled =
            !latestRecordingId || latest.status === 'running';
    } catch (error) {
        document.getElementById('record-latest').textContent = 'Unavailable';
        console.error('Simulation records could not be loaded:', error);
    }
}

function downloadLatestRecording() {
    if (!latestRecordingId) return;
    window.location.href =
        `/api/recordings/${encodeURIComponent(latestRecordingId)}/data`;
}

// --- Stimulus toggle buttons ---
function initStimulusButtons() {
    const infoEl = document.getElementById('stim-info');

    document.querySelectorAll('.stim-btn').forEach(btn => {
        const stim = btn.dataset.stim;
        const color = btn.dataset.color;
        const info = btn.dataset.info;

        // Hover: show info tooltip
        btn.addEventListener('mouseenter', () => {
            if (infoEl && info) {
                infoEl.textContent = info;
                infoEl.style.display = 'block';
                infoEl.style.borderLeft = `3px solid ${color}`;
            }
        });
        btn.addEventListener('mouseleave', () => {
            if (infoEl) infoEl.style.display = 'none';
        });

        btn.addEventListener('click', () => {
            btn.classList.toggle('active');
            if (btn.classList.contains('active')) {
                btn.style.borderColor = color;
                btn.style.background = color + '33';
                btn.style.color = 'white';
                activeStimuli.add(stim);
            } else {
                btn.style.borderColor = '';
                btn.style.background = '';
                btn.style.color = '';
                activeStimuli.delete(stim);
            }
            // Send stimulus update to backend
            sendStimulusUpdate();
            updateActiveDisplay();
        });
    });
}

function sendStimulusUpdate() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            cmd: 'set_stimuli',
            stimuli: Array.from(activeStimuli),
        }));
    }
}

function updateActiveDisplay() {
    const el = document.getElementById('active-stim-display');
    if (el) {
        el.textContent = activeStimuli.size > 0
            ? Array.from(activeStimuli).join(' + ')
            : 'None';
    }
}

// --- WebSocket ---
function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws/sim`);

    ws.onopen = () => {
        document.getElementById('status-text').textContent = 'Connected';
        document.getElementById('status-dot').classList.add('connected');
    };

    ws.onclose = () => {
        document.getElementById('status-text').textContent = 'Disconnected';
        document.getElementById('status-dot').classList.remove('connected', 'running');
        setTimeout(connectWebSocket, 2000);
    };

    ws.onmessage = (evt) => {
        const data = JSON.parse(evt.data);

        if (data.event === 'episode_init') {
            _episodeInit(data);
            return;
        }
        if (data.event === 'episode_segment') {
            _episodeSegment(data);
            return;
        }
        if (data.event === 'episode_end') {
            _episodeEnd(data);
            refreshRecordings();
            return;
        }

        // End events
        if (data.event === 'end' || data.event === 'walk_end') {
            document.getElementById('status-dot').classList.remove('running');
            const elapsed = ((performance.now() - simStartWall) / 1000).toFixed(1);
            document.getElementById('status-text').textContent =
                `Done (${totalFrames} frames in ${elapsed}s)`;
            if (data.event === 'walk_end') Room.walkEnd(data);
            document.getElementById('behavior-badge').textContent = 'ENDED';
            refreshRecordings();
            return;
        }

        // Walk animation frames
        if (data.event === 'walk_init') {
            Room.walkInit(data);
            refreshRecordings();
            return;
        }
        if (data.event === 'walk_frame') {
            Room.walkUpdate(data);
            if (data.motor_decision) {
                const decision = data.motor_decision;
                const executed = decision.executed_controller || decision.controller || 'idle';
                document.getElementById('info-flight').textContent =
                    decision.flight_state || data.flight_state || 'GROUNDED';
                document.getElementById('info-behavior').textContent =
                    decision.behavior_intent || data.behavior_mode || 'idle';
                document.getElementById('behavior-badge').textContent =
                    `LIFE · ${executed.replaceAll('_', ' ').toUpperCase()}`;
                document.getElementById('status-text').textContent =
                    data.timing_mode === 'brain_time_sync'
                        ? `LIF sync | ${executed.replaceAll('_', ' ')} | ` +
                          `brain ${(data.brain_t_ms || 0).toFixed(0)}ms | ` +
                          `wall ${(data.t_ms / 1000).toFixed(1)}s`
                        : `Digital Life | ${executed.replaceAll('_', ' ')} | ` +
                          `${(data.t_ms / 1000).toFixed(1)}s`;
            } else if (data.flight_state) {
                document.getElementById('info-flight').textContent = data.flight_state;
                document.getElementById('info-behavior').textContent =
                    data.phase === 'landing' ? 'landing' : 'flight';
            }
            if (!data.motor_decision && data.walk_state) {
                document.getElementById('info-flight').textContent = 'GROUNDED';
                document.getElementById('info-behavior').textContent = 'walking';
            }
            if (Number.isFinite(data.t_ms)) {
                if (data.timing_mode === 'brain_time_sync') {
                    document.getElementById('info-time').textContent =
                        `${(data.body_simulation_ms || 0).toFixed(1)} motor ms`;
                    document.getElementById('time-display').textContent =
                        `brain = ${(data.brain_t_ms || 0).toFixed(1)} ms · ` +
                        `body = ${(data.body_simulation_ms || 0).toFixed(1)} ms · ` +
                        `wall = ${(data.t_ms / 1000).toFixed(1)} s`;
                } else {
                    document.getElementById('info-time').textContent = `${data.t_ms.toFixed(1)} ms`;
                    document.getElementById('time-display').textContent = `t = ${data.t_ms.toFixed(1)} ms`;
                }
            }
            if (data.phase === 'landing') {
                document.getElementById('behavior-badge').textContent = 'LANDING';
            } else if (data.preview_route === 'measured-room-coverage') {
                document.getElementById('behavior-badge').textContent =
                    data.boundary_avoidance
                        ? `BOUNDARY · ${data.segment_route || 'TURN'}`
                        : `MOCAP · ${data.segment_route || 'EXPLORE'}`;
            } else if (data.preview_route === 'measured-walk-room-coverage') {
                document.getElementById('behavior-badge').textContent =
                    data.boundary_avoidance
                        ? `WALK BOUNDARY · ${data.segment_route || 'TURN'}`
                        : `WALK MOCAP · ${data.segment_route || 'EXPLORE'}`;
            } else if (data.maneuver_source === 'measured_hdf5'
                    && data.phase === 'walking_data_preview') {
                document.getElementById('behavior-badge').textContent =
                    `WALK MOCAP · ${data.segment_route || 'CLIP'}`;
            }
            if (data.episode_id) _episodeFrame(data);
            totalFrames++;
            _updateFps();
            return;
        }

        // Brain simulation frames
        if (data.event === 'brain_frame') {
            Dashboard.update(data);
            BrainVis.update(data);
            if (!data.body_driven) Room.brainDrive(data);
            totalFrames++;
            _updateFps();

            const elapsedS = ((performance.now() - simStartWall) / 1000).toFixed(0);
            document.getElementById('status-text').textContent =
                activeBrainTimeSync
                    ? `LIF sync | brain ${data.t_ms.toFixed(0)}ms | wall ${elapsedS}s`
                    : `Brain | sim ${data.t_ms.toFixed(0)}ms | ${elapsedS}s`;
            return;
        }

        // Legacy NT simulation frames
        if (data.event === 'frame') {
            Dashboard.update(data);
            Room.updateFly(data);
            totalFrames++;
            _updateFps();
        }
    };
}

function _updateFps() {
    frameCount++;
    const now = performance.now();
    if (now - lastFpsTime >= 1000) {
        fpsDisplay = frameCount;
        frameCount = 0;
        lastFpsTime = now;
        document.getElementById('fps-display').textContent = fpsDisplay + ' fps';
    }
}

// --- Brain simulation ---
async function startBrain() {
    _hideEpisodeMonitor();
    Dashboard.resetChart();
    simStartWall = performance.now();
    totalFrames = 0;

    const syncToBrainTime = document.getElementById('brain-time-sync')?.checked === true;
    activeBrainTimeSync = syncToBrainTime;
    const res = await fetch('/api/brain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            stimuli: Array.from(activeStimuli),
            sync_to_brain_time: syncToBrainTime,
        }),
    });
    await res.json();
    refreshRecordings();

    document.getElementById('status-text').textContent =
        syncToBrainTime
            ? 'Starting 138K neurons + brain-time synchronized body...'
            : 'Starting 138K neurons + persistent digital body...';
    document.getElementById('status-dot').classList.add('running');
}

// --- Behavior replay ---
async function startBehavior(name) {
    _hideEpisodeMonitor();
    activeBrainTimeSync = false;
    lastBehavior = name;
    Room.resetTrail();
    Dashboard.resetChart();
    simStartWall = performance.now();
    totalFrames = 0;

    const res = await fetch(`/api/${name}`, { method: 'POST' });
    await res.json();

    document.getElementById('status-text').textContent = `Starting ${name}...`;
    document.getElementById('status-dot').classList.add('running');
}

async function startFlightPreview(routeName) {
    _hideEpisodeMonitor();
    Room.resetTrail();
    Dashboard.resetChart();
    simStartWall = performance.now();
    totalFrames = 0;

    const res = await fetch(`/api/flight-preview/${encodeURIComponent(routeName)}`, {
        method: 'POST',
    });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) {
        document.getElementById('status-text').textContent =
            payload.detail || 'Flight preview failed to start';
        return;
    }
    document.getElementById('status-text').textContent =
        `Flight data | ${routeName} #${payload.trajectory_id} | ` +
        `${payload.segment_count} segment | ${payload.playback_rate}×`;
    document.getElementById('behavior-badge').textContent =
        `MOCAP ${routeName.toUpperCase()} · ${payload.playback_rate}×`;
    document.getElementById('status-dot').classList.add('running');
}

async function startExplorationFlight() {
    _hideEpisodeMonitor();
    Room.resetTrail();
    Dashboard.resetChart();
    simStartWall = performance.now();
    totalFrames = 0;
    const res = await fetch('/api/flight-explore', { method: 'POST' });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) {
        document.getElementById('status-text').textContent =
            payload.detail || 'Exploration flight failed to start';
        return;
    }
    document.getElementById('status-text').textContent =
        'Explore flight | measured maneuvers | homogeneous coverage | boundary aware';
    document.getElementById('behavior-badge').textContent = 'MOCAP EXPLORE';
    document.getElementById('status-dot').classList.add('running');
}

async function landExplorationFlight() {
    const res = await fetch('/api/flight-explore/land', { method: 'POST' });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) {
        document.getElementById('status-text').textContent =
            payload.detail || 'Landing request failed';
        return;
    }
    document.getElementById('status-text').textContent = 'Landing requested...';
    document.getElementById('behavior-badge').textContent = 'LANDING';
}

async function startWalkingPreview(routeName) {
    _hideEpisodeMonitor();
    Room.resetTrail();
    Dashboard.resetChart();
    simStartWall = performance.now();
    totalFrames = 0;
    const res = await fetch(`/api/walk-preview/${encodeURIComponent(routeName)}`, {
        method: 'POST',
    });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) {
        document.getElementById('status-text').textContent =
            payload.detail || 'Walking preview failed to start';
        return;
    }
    document.getElementById('status-text').textContent =
        `Walking data | ${routeName} #${payload.trajectory_id} | ` +
        `${payload.segment_count} segment | ${payload.sample_hz} Hz`;
    document.getElementById('behavior-badge').textContent =
        `WALK MOCAP · ${routeName.toUpperCase()}`;
    document.getElementById('status-dot').classList.add('running');
}

async function startWalkingExploration() {
    _hideEpisodeMonitor();
    Room.resetTrail();
    Dashboard.resetChart();
    simStartWall = performance.now();
    totalFrames = 0;
    const res = await fetch('/api/walk-explore', { method: 'POST' });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) {
        document.getElementById('status-text').textContent =
            payload.detail || 'Walking exploration failed to start';
        return;
    }
    document.getElementById('status-text').textContent =
        'Explore walk | measured path | stride-locked NeuromechFly gait | boundary aware';
    document.getElementById('behavior-badge').textContent = 'WALK MOCAP EXPLORE';
    document.getElementById('status-dot').classList.add('running');
}

// --- Unified embodied episode replay ---
async function startEpisode(episodeId) {
    Room.resetTrail();
    Dashboard.resetChart();
    simStartWall = performance.now();
    totalFrames = 0;

    const res = await fetch(`/api/episode/${encodeURIComponent(episodeId)}`, {
        method: 'POST',
    });
    if (!res.ok) {
        const error = await res.json().catch(() => ({}));
        document.getElementById('status-text').textContent =
            error.detail || 'Episode failed to start';
        return;
    }
    const payload = await res.json();
    document.getElementById('status-text').textContent = `Starting ${payload.title}...`;
    document.getElementById('status-dot').classList.add('running');
}

function _episodeInit(data) {
    activeEpisode = data;
    const monitor = document.getElementById('episode-monitor');
    monitor.hidden = false;
    document.getElementById('episode-title').textContent = data.title;
    document.getElementById('episode-source').textContent =
        data.source_mode === 'recorded_lif_causal_motor'
            ? 'CAUSAL MOTOR · V3'
            : data.source_mode === 'recorded_lif_overlay'
            ? 'RECORDED LIF · V2'
            : (data.source_mode === 'cached_controller_replay'
                ? 'CACHED REPLAY · V1'
                : data.source_mode);
    document.getElementById('episode-progress-fill').style.width = '0%';
    document.getElementById('episode-time').textContent =
        `0.0 / ${(data.duration_ms / 1000).toFixed(1)}s`;
    _resetEpisodeNeuralReadout();
    _resetEpisodeMotorReadout(data.motor_control === 'dn_state_resolver');

    const steps = document.getElementById('episode-steps');
    steps.innerHTML = '';
    data.segments.forEach((segment, index) => {
        const el = document.createElement('div');
        el.className = 'episode-step';
        el.dataset.index = index;
        el.title = segment.label;
        el.textContent = segment.label;
        steps.appendChild(el);
    });
}

function _episodeSegment(data) {
    if (!activeEpisode) return;
    activeEpisodeBehavior = data.behavior;
    document.getElementById('episode-stage').textContent = data.label;
    document.getElementById('behavior-badge').textContent =
        data.motor_control === 'dn_state_resolver' ? 'RESOLVING' : data.behavior.toUpperCase();
    document.getElementById('status-text').textContent =
        `Episode | ${data.segment_index + 1}/${data.segment_count} | ${data.label}`;

    document.querySelectorAll('.episode-step').forEach((el, index) => {
        el.classList.toggle('active', index === data.segment_index);
        el.classList.toggle('done', index < data.segment_index);
    });

    const pipeline = data.pipeline || {};
    document.getElementById('episode-world').textContent = pipeline.world || '—';
    document.getElementById('episode-sensory').textContent = pipeline.sensory || '—';
    document.getElementById('episode-neural').textContent = pipeline.neural || '—';
    document.getElementById('episode-motor').textContent = pipeline.motor || '—';
}

function _episodeFrame(data) {
    if (!activeEpisode) return;
    const progress = Math.max(0, Math.min(1, data.episode_progress || 0));
    document.getElementById('episode-progress-fill').style.width = `${progress * 100}%`;
    document.getElementById('episode-time').textContent =
        `${(data.episode_t_ms / 1000).toFixed(1)} / ` +
        `${(activeEpisode.duration_ms / 1000).toFixed(1)}s`;

    if (data.neural) {
        const neuralFrame = {
            t_ms: data.episode_t_ms,
            brain_steps: data.neural.brain_steps,
            total_spikes: data.neural.total_spikes,
            behavior_mode: data.motor_decision
                ? data.motor_decision.behavior_intent
                : data.neural.behavior_mode,
            flight_state: data.motor_decision
                ? data.motor_decision.flight_state
                : data.flight_state,
            dn: data.neural.dn || {},
            pop: data.neural.pop || {},
        };
        Dashboard.update(neuralFrame);
        BrainVis.update(neuralFrame);
        _updateEpisodeNeuralReadout(data.neural);

        if (data.motor_decision) {
            _updateEpisodeMotorReadout(data.motor_decision);
        }

        // The body is controller-cache replay in Phase 2. Keep its label
        // separate from the recorded LIF classifier shown below.
        if (data.motor_decision) {
            document.getElementById('behavior-badge').textContent =
                data.motor_decision.behavior_intent.toUpperCase();
        } else if (activeEpisodeBehavior) {
            document.getElementById('behavior-badge').textContent =
                activeEpisodeBehavior.toUpperCase();
        }
    }
}

function _updateEpisodeMotorReadout(decision) {
    document.getElementById('episode-motor-intent').textContent =
        (decision.behavior_intent || '—').toUpperCase();
    document.getElementById('episode-flight-state').textContent =
        decision.flight_state || '—';
    document.getElementById('episode-controller').textContent =
        (decision.controller || '—').toUpperCase();
    document.getElementById('episode-queued-intent').textContent =
        (decision.queued_intent || '—').toUpperCase();
    document.getElementById('episode-motor-reason').textContent = decision.reason || '—';
}

function _resetEpisodeMotorReadout(isCausal) {
    ['episode-motor-intent', 'episode-flight-state', 'episode-controller',
        'episode-queued-intent'].forEach(id => {
        document.getElementById(id).textContent = '—';
    });
    document.getElementById('episode-motor-reason').textContent = isCausal
        ? 'Waiting for the first DN-driven motor decision'
        : 'Phase 1/2: motor selection is not causal';
}

function _updateEpisodeNeuralReadout(neural) {
    const dn = neural.dn || {};
    const turn = (dn.turn_L || 0) - (dn.turn_R || 0);
    document.getElementById('episode-dn-forward').textContent = (dn.forward || 0).toFixed(3);
    document.getElementById('episode-dn-turn').textContent = turn.toFixed(3);
    document.getElementById('episode-dn-groom').textContent = (dn.groom || 0).toFixed(3);
    document.getElementById('episode-dn-feed').textContent = (dn.feed || 0).toFixed(3);
    document.getElementById('episode-spikes').textContent = neural.total_spikes || 0;
    document.getElementById('episode-neural-stimulus').textContent =
        `STIM ${(neural.stimulus || '—').toUpperCase()}`;
    document.getElementById('episode-neural-behavior').textContent =
        `LIF ${(neural.behavior_mode || '—').toUpperCase()}`;
}

function _resetEpisodeNeuralReadout() {
    ['forward', 'turn', 'groom', 'feed'].forEach(key => {
        document.getElementById(`episode-dn-${key}`).textContent = '0.000';
    });
    document.getElementById('episode-spikes').textContent = '0';
    document.getElementById('episode-neural-stimulus').textContent = 'STIM —';
    document.getElementById('episode-neural-behavior').textContent = 'LIF —';
}

function _episodeEnd(data) {
    if (!activeEpisode) return;
    const completed = data.completed === true;
    document.getElementById('episode-stage').textContent = completed ? 'Complete' : 'Stopped';
    document.getElementById('episode-progress-fill').style.width =
        completed ? '100%' : `${(data.emitted_frames / data.frame_count) * 100}%`;
    document.querySelectorAll('.episode-step').forEach(el => {
        el.classList.remove('active');
        if (completed) el.classList.add('done');
    });
}

function _hideEpisodeMonitor() {
    activeEpisode = null;
    activeEpisodeBehavior = null;
    const monitor = document.getElementById('episode-monitor');
    if (monitor) monitor.hidden = true;
}

function toggleInfo(id) {
    const el = document.getElementById(id);
    if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

async function stopSim() {
    await fetch('/api/stop', { method: 'POST' });
    activeBrainTimeSync = false;
    document.getElementById('status-text').textContent = 'Stopped';
    document.getElementById('status-dot').classList.remove('running');
    refreshRecordings();
}
