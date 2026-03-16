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

// Active stimuli (multiple can be on)
let activeStimuli = new Set();

window.addEventListener('DOMContentLoaded', async () => {
    Room.init(document.getElementById('three-canvas'));
    Dashboard.init();
    BrainVis.init();
    Controls.init();
    connectWebSocket();
    initStimulusButtons();
});

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

        // End events
        if (data.event === 'end' || data.event === 'walk_end') {
            document.getElementById('status-dot').classList.remove('running');
            const elapsed = ((performance.now() - simStartWall) / 1000).toFixed(1);
            document.getElementById('status-text').textContent =
                `Done (${totalFrames} frames in ${elapsed}s)`;
            if (data.event === 'walk_end') Room.walkEnd();
            document.getElementById('behavior-badge').textContent = 'ENDED';
            return;
        }

        // Walk animation frames
        if (data.event === 'walk_init') {
            Room.walkInit(data);
            return;
        }
        if (data.event === 'walk_frame') {
            Room.walkUpdate(data);
            totalFrames++;
            _updateFps();
            return;
        }

        // Brain simulation frames
        if (data.event === 'brain_frame') {
            Dashboard.update(data);
            BrainVis.update(data);
            Room.brainDrive(data);
            totalFrames++;
            _updateFps();

            const elapsedS = ((performance.now() - simStartWall) / 1000).toFixed(0);
            document.getElementById('status-text').textContent =
                `Brain | sim ${data.t_ms.toFixed(0)}ms | ${elapsedS}s`;
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
    Dashboard.resetChart();
    simStartWall = performance.now();
    totalFrames = 0;

    Room.loadWalkCache();

    const res = await fetch('/api/brain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stimuli: Array.from(activeStimuli) }),
    });
    await res.json();

    document.getElementById('status-text').textContent = 'Starting 138K neurons...';
    document.getElementById('status-dot').classList.add('running');
}

// --- Behavior replay ---
async function startBehavior(name) {
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

function toggleInfo(id) {
    const el = document.getElementById(id);
    if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

async function stopSim() {
    await fetch('/api/stop', { method: 'POST' });
    document.getElementById('status-text').textContent = 'Stopped';
    document.getElementById('status-dot').classList.remove('running');
}
