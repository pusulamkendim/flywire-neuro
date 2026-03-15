/**
 * App: WebSocket connection, state management, glue.
 *
 * "Buffered playback" model: backend pushes frames as fast as it can,
 * frontend renders each frame immediately when it arrives.
 * Mock mode → ~30 fps (wall-clock paced).
 * Real brain → ~1-10 fps depending on CPU (data-paced, no waiting).
 */

let ws = null;
let frameCount = 0;
let lastFpsTime = performance.now();
let fpsDisplay = 0;
let simStartWall = 0;   // wall-clock ms when sim started
let totalFrames = 0;

window.addEventListener('DOMContentLoaded', async () => {
    Room.init(document.getElementById('three-canvas'));
    Dashboard.init();
    await Controls.init();
    connectWebSocket();
});

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

        if (data.event === 'end' || data.event === 'walk_end') {
            document.getElementById('status-dot').classList.remove('running');
            const elapsed = ((performance.now() - simStartWall) / 1000).toFixed(1);
            const tag = `Done (${totalFrames} frames in ${elapsed}s)`;
            document.getElementById('status-text').textContent = tag;

            if (data.event === 'walk_end') Room.walkEnd();

            if (document.getElementById('chk-loop').checked) {
                setTimeout(() => {
                    if (data.event === 'walk_end') startWalk();
                    else startSim();
                }, 500);
                return;
            }
            document.getElementById('behavior-badge').textContent = 'ENDED';
            return;
        }

        // Walking animation frames
        if (data.event === 'walk_init') {
            Room.walkInit(data.geom_names);
            return;
        }

        if (data.event === 'walk_frame') {
            Room.walkUpdate(data);
            totalFrames++;

            document.getElementById('behavior-badge').textContent = 'WALKING';
            document.getElementById('behavior-badge').style.borderColor = 'var(--accent-forward)';
            document.getElementById('behavior-badge').style.color = 'var(--accent-forward)';
            document.getElementById('time-display').textContent = `t = ${data.t_ms.toFixed(1)} ms`;

            // FPS counter
            frameCount++;
            const now = performance.now();
            if (now - lastFpsTime >= 1000) {
                fpsDisplay = frameCount;
                frameCount = 0;
                lastFpsTime = now;
                document.getElementById('fps-display').textContent = fpsDisplay + ' fps';
                const elapsedS = ((now - simStartWall) / 1000).toFixed(0);
                document.getElementById('status-text').textContent =
                    `Walking | sim ${data.t_ms.toFixed(0)}ms | ${elapsedS}s`;
            }
            return;
        }

        if (data.event === 'frame') {
            Dashboard.update(data);
            Room.updateFly(data);
            totalFrames++;

            // FPS counter (frames received per second)
            frameCount++;
            const now = performance.now();
            if (now - lastFpsTime >= 1000) {
                fpsDisplay = frameCount;
                frameCount = 0;
                lastFpsTime = now;
                document.getElementById('fps-display').textContent = fpsDisplay + ' fps';
            }

            // Status: show sim time + wall elapsed
            const elapsedS = ((now - simStartWall) / 1000).toFixed(0);
            const simT = data.t_ms.toFixed(1);
            const isReal = document.getElementById('chk-real-brain').checked;
            const mode = isReal ? '138K neurons' : 'mock';
            document.getElementById('status-text').textContent =
                `${mode} | sim ${simT}ms | ${elapsedS}s elapsed`;
        }
    };
}

async function startSim() {
    const scenario = Controls.getSelected();

    const knownScenarios = {
        escape: {
            duration_s: 1.5,
            phases: [
                { name: 'hunger', start_ms: 0, end_ms: 300 },
                { name: 'looming', start_ms: 300, end_ms: 600 },
                { name: 'free', start_ms: 600, end_ms: 1500 },
            ],
        },
        foraging: {
            duration_s: 1.5,
            phases: [
                { name: 'search', start_ms: 0, end_ms: 500 },
                { name: 'sugar', start_ms: 500, end_ms: 1200 },
                { name: 'feeding', start_ms: 1200, end_ms: 1500 },
            ],
        },
        grooming: {
            duration_s: 1.2,
            phases: [
                { name: 'walking', start_ms: 0, end_ms: 300 },
                { name: 'touch', start_ms: 300, end_ms: 900 },
                { name: 'free', start_ms: 900, end_ms: 1200 },
            ],
        },
    };
    Controls.buildPhaseBar(knownScenarios[scenario] || knownScenarios.escape);

    Dashboard.resetChart();
    Room.resetTrail();

    const useRealBrain = document.getElementById('chk-real-brain').checked;
    simStartWall = performance.now();
    totalFrames = 0;

    const res = await fetch('/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            scenario,
            duration_s: knownScenarios[scenario]?.duration_s || 1.5,
            use_real_brain: useRealBrain,
        }),
    });
    await res.json();

    const mode = useRealBrain ? '138K neurons' : 'mock';
    document.getElementById('status-text').textContent = `Starting ${mode}...`;
    document.getElementById('status-dot').classList.add('running');
}

async function startWalk() {
    Room.resetTrail();
    simStartWall = performance.now();
    totalFrames = 0;

    const res = await fetch('/api/walk?duration_s=5', { method: 'POST' });
    await res.json();

    document.getElementById('status-text').textContent = 'Starting walk...';
    document.getElementById('status-dot').classList.add('running');
}

async function stopSim() {
    await fetch('/api/stop', { method: 'POST' });
    document.getElementById('status-text').textContent = 'Stopped';
    document.getElementById('status-dot').classList.remove('running');
}
