/**
 * Brain Dorsal View — real FlyWire connectome (139,244 neuron positions).
 * 10,000 sampled neurons as scatter + key population highlights.
 * Style reference: FlyWire brain emulation visualization (bright optic lobes,
 * scattered central brain, black background).
 */

const BrainVis = (() => {
    let ctx = null;
    let data = null;
    let popActivity = {};

    // NT index → color [r, g, b]
    const NT_COLORS = [
        [200, 230, 180],  // 0: ACh (warm white-green)
        [100, 140, 220],  // 1: GABA (blue)
        [200, 120, 180],  // 2: Glut (pink)
        [240, 220, 80],   // 3: DA (yellow)
        [180, 120, 220],  // 4: 5HT (purple)
        [220, 160, 80],   // 5: OA (orange)
        [80, 80, 100],    // 6: unknown (dim)
    ];

    // Population glow colors
    const POP_COLORS = {
        PAM:  [80, 255, 80],
        PPL1: [255, 60, 60],
        MBON: [60, 220, 120],
        KC:   [255, 240, 100],
    };

    // DN key → which region(s) light up
    const DN_REGION_MAP = {
        escape:  ['optic', 'visual_projection', 'central'],
        forward: ['central', 'motor', 'descending'],
        backward:['central', 'motor'],
        turn_L:  ['central', 'descending'],
        turn_R:  ['central', 'descending'],
        groom:   ['sensory', 'central'],
        feed:    ['sensory', 'central', 'motor'],
    };

    let regionActivity = {};

    function init() {
        const canvas = document.getElementById('brain-canvas');
        if (!canvas) return;
        ctx = canvas.getContext('2d');

        fetch('/static/assets/brain_map.json')
            .then(r => r.json())
            .then(d => {
                data = d;
                _render();
                console.log(`Brain map: ${d.total.toLocaleString()} neurons, ${d.sampled} rendered`);
            });
    }

    function update(frame) {
        if (!ctx || !data) return;

        const dn = frame.dn || {};
        const pop = frame.pop || {};

        // Population activity
        for (const name of Object.keys(POP_COLORS)) {
            const key = name.toLowerCase();
            const raw = pop[key] || 0;
            const target = Math.min(1, raw / 30);
            popActivity[name] = (popActivity[name] || 0) * 0.6 + target * 0.4;
        }

        // Region activity from DN rates
        for (const [dnKey, regions] of Object.entries(DN_REGION_MAP)) {
            const val = dn[dnKey] || 0;
            for (const reg of regions) {
                regionActivity[reg] = Math.max(regionActivity[reg] || 0, val);
            }
        }
        // Decay all regions
        for (const reg of Object.keys(regionActivity)) {
            regionActivity[reg] *= 0.85;
        }

        _render();
    }

    function _render() {
        if (!ctx || !data) return;

        const canvas = ctx.canvas;
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        ctx.scale(dpr, dpr);

        // Black background
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, w, h);

        // Padding to keep brain centered with fly-like proportions
        const padX = w * 0.03;
        const padY = h * 0.05;
        const bw = w - padX * 2;
        const bh = h - padY * 2;

        // --- Scatter: 10K neuron positions ---
        const pts = data.points;
        for (let i = 0; i < pts.length; i++) {
            const [px, py, nt] = pts[i];
            const x = padX + px * bw;
            const y = padY + py * bh;
            const col = NT_COLORS[nt] || NT_COLORS[6];

            // Check if this point's region is active
            let boost = 0;
            // Rough region assignment by X position
            if (px < 0.2 || px > 0.8) boost = regionActivity['optic'] || 0;
            else if (px > 0.35 && px < 0.65) boost = regionActivity['central'] || 0;

            const alpha = 0.08 + boost * 0.6;
            const size = 1 + boost * 1.5;

            ctx.fillStyle = `rgba(${col[0]}, ${col[1]}, ${col[2]}, ${alpha})`;
            ctx.fillRect(x - size/2, y - size/2, size, size);
        }

        // --- Region glow (when active) ---
        for (const [regName, info] of Object.entries(data.regions)) {
            const act = regionActivity[regName] || 0;
            if (act < 0.02) continue;

            const x = padX + info.x * bw;
            const y = padY + info.y * bh;
            const spread = Math.max(10, info.spread * bw * 2);

            const grad = ctx.createRadialGradient(x, y, 0, x, y, spread);
            // Optic lobes get warm yellow, central gets cool blue-white
            if (regName === 'optic' || regName === 'visual_projection') {
                grad.addColorStop(0, `rgba(255, 240, 150, ${act * 0.3})`);
                grad.addColorStop(0.5, `rgba(200, 180, 80, ${act * 0.1})`);
            } else {
                grad.addColorStop(0, `rgba(180, 200, 255, ${act * 0.25})`);
                grad.addColorStop(0.5, `rgba(100, 120, 200, ${act * 0.08})`);
            }
            grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(x, y, spread, 0, Math.PI * 2);
            ctx.fill();
        }

        // --- Key populations (PAM, PPL1, MBON, KC) ---
        for (const [name, pts] of Object.entries(data.pop_points)) {
            const act = popActivity[name] || 0;
            const col = POP_COLORS[name] || [200, 200, 200];
            const alpha = 0.05 + act * 0.9;
            const size = 1 + act * 2;

            ctx.fillStyle = `rgba(${col[0]}, ${col[1]}, ${col[2]}, ${alpha})`;
            for (const [px, py] of pts) {
                const x = padX + px * bw;
                const y = padY + py * bh;
                ctx.fillRect(x - size/2, y - size/2, size, size);
            }

            // Label when active
            if (act > 0.05 && pts.length > 0) {
                const cx = padX + pts.reduce((s, p) => s + p[0], 0) / pts.length * bw;
                const cy = padY + pts.reduce((s, p) => s + p[1], 0) / pts.length * bh;
                ctx.fillStyle = `rgba(${col[0]}, ${col[1]}, ${col[2]}, ${act})`;
                ctx.font = '8px monospace';
                ctx.textAlign = 'center';
                ctx.fillText(`${name} (${pts.length})`, cx, cy - 8);
            }
        }

        // Header
        ctx.fillStyle = 'rgba(100, 140, 170, 0.6)';
        ctx.font = '8px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`${data.total.toLocaleString()} neurons — FlyWire v783`, 4, 10);

        ctx.setTransform(1, 0, 0, 1, 0, 0);
    }

    return { init, update };
})();
