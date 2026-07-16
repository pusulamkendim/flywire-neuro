/**
 * Anatomy View v2 — FlyWire dorsal brain visualization.
 *
 * The static layer is built from real FlyWire soma positions and cached.
 * Runtime population values are aggregate activity (not per-neuron spikes yet),
 * so active populations are shown as anatomically located population glows.
 */

const BrainVis = (() => {
    let canvas = null;
    let ctx = null;
    let data = null;
    let staticLayer = null;
    let staticDirty = true;
    let lastFrameAt = 0;
    let lastTickAt = performance.now();

    // soma_x/soma_y bounds in neuron_annotations.tsv are about 2.22:1.
    // Keeping this ratio prevents the brain from stretching with the panel.
    const ANATOMY_ASPECT = 2.22;
    const MAX_DPR = 2;
    const RENDER_INTERVAL_MS = 1000 / 30;

    const NT_COLORS = [
        [116, 210, 165],  // 0: ACh
        [93, 153, 235],   // 1: GABA
        [220, 104, 178],  // 2: Glutamate
        [250, 213, 75],   // 3: Dopamine
        [178, 116, 232],  // 4: Serotonin
        [235, 158, 74],   // 5: Octopamine
        [103, 122, 145],  // 6: Unknown
    ];

    const POP_COLORS = {
        PAM:  [91, 244, 131],
        PPL1: [255, 91, 91],
        MBON: [65, 225, 180],
        KC:   [255, 224, 102],
    };

    // Different populations have very different sizes, so a single raw-count
    // divisor would make PPL1 invisible and KC permanently saturated.
    const POP_SCALES = { PAM: 18, PPL1: 4, MBON: 20, KC: 120 };

    const POP_READERS = {
        PAM: pop => pop.pam || 0,
        PPL1: pop => pop.ppl1 || 0,
        MBON: pop => (pop.mbon_approach || 0)
            + (pop.mbon_avoidance || 0)
            + (pop.mbon_suppress || 0),
        KC: pop => pop.kc || 0,
    };

    // Match the motor resolver's entry thresholds. A signal that can change
    // behavior must also be clearly visible even when its raw rate is 0.01.
    const DN_VISUAL_THRESHOLDS = {
        escape: 0.08,
        forward: 0.01,
        backward: 0.02,
        turn_L: 0.02,
        turn_R: 0.02,
        groom: 0.02,
        feed: 0.05,
    };

    const DN_REGION_MAP = {
        escape:  ['optic', 'visual_projection', 'central'],
        forward: ['central', 'motor', 'descending'],
        backward:['central', 'motor'],
        turn_L:  ['central', 'descending'],
        turn_R:  ['central', 'descending'],
        groom:   ['sensory', 'central'],
        feed:    ['sensory', 'central', 'motor'],
    };

    const popActivity = {};
    const popTargets = {};
    const regionActivity = {};
    const regionTargets = {};

    function init() {
        canvas = document.getElementById('brain-canvas');
        if (!canvas) return;

        ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
        staticLayer = document.createElement('canvas');

        const observer = new ResizeObserver(() => {
            staticDirty = true;
        });
        observer.observe(canvas);

        fetch('/static/assets/brain_map.json')
            .then(r => r.json())
            .then(d => {
                data = d;
                staticDirty = true;
                console.log(`Anatomy View: ${d.total.toLocaleString()} neurons, ${d.sampled} soma points rendered`);
            })
            .catch(err => console.error('Brain anatomy map could not be loaded:', err));

        requestAnimationFrame(_tick);
    }

    function update(frame) {
        if (!data) return;
        const dn = frame.dn || {};
        const pop = frame.pop || {};

        for (const name of Object.keys(POP_COLORS)) {
            const raw = Math.max(0, POP_READERS[name](pop));
            const scale = POP_SCALES[name] || 30;
            // Log response preserves weak activity without letting large groups clip.
            popTargets[name] = Math.min(1, Math.log1p(raw) / Math.log1p(scale));
        }

        for (const [dnKey, regions] of Object.entries(DN_REGION_MAP)) {
            const raw = Math.max(0, Number(dn[dnKey]) || 0);
            const threshold = DN_VISUAL_THRESHOLDS[dnKey] || 0.05;
            const value = Math.min(1, Math.sqrt(raw / threshold));
            for (const region of regions) {
                regionTargets[region] = Math.max(regionTargets[region] || 0, value);
            }
        }
    }

    function _tick(now) {
        const dt = Math.min(100, Math.max(0, now - lastTickAt));
        lastTickAt = now;

        // Quick attack, slower biological-looking fluorescence decay.
        for (const name of Object.keys(POP_COLORS)) {
            const current = popActivity[name] || 0;
            const target = popTargets[name] || 0;
            const tau = target > current ? 90 : 360;
            popActivity[name] = current + (target - current) * (1 - Math.exp(-dt / tau));
            popTargets[name] = target * Math.exp(-dt / 520);
        }

        const regionNames = new Set([
            ...Object.keys(regionActivity),
            ...Object.keys(regionTargets),
        ]);
        for (const name of regionNames) {
            const current = regionActivity[name] || 0;
            const target = regionTargets[name] || 0;
            const tau = target > current ? 100 : 480;
            regionActivity[name] = current + (target - current) * (1 - Math.exp(-dt / tau));
            regionTargets[name] = target * Math.exp(-dt / 650);
        }

        if (data && (staticDirty || now - lastFrameAt >= RENDER_INTERVAL_MS)) {
            _render(now);
            lastFrameAt = now;
        }
        requestAnimationFrame(_tick);
    }

    function _ensureCanvasSize() {
        const rect = canvas.getBoundingClientRect();
        const width = Math.max(1, Math.round(rect.width));
        const height = Math.max(1, Math.round(rect.height));
        const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
        const pixelWidth = Math.round(width * dpr);
        const pixelHeight = Math.round(height * dpr);

        if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
            canvas.width = pixelWidth;
            canvas.height = pixelHeight;
            staticLayer.width = pixelWidth;
            staticLayer.height = pixelHeight;
            staticDirty = true;
        }

        return { width, height, dpr };
    }

    function _brainLayout(width, height) {
        const outerX = Math.max(16, width * 0.025);
        const outerTop = 24;
        const outerBottom = 23;
        const availableWidth = width - outerX * 2;
        const availableHeight = height - outerTop - outerBottom;

        let brainWidth = availableWidth;
        let brainHeight = brainWidth / ANATOMY_ASPECT;
        if (brainHeight > availableHeight) {
            brainHeight = availableHeight;
            brainWidth = brainHeight * ANATOMY_ASPECT;
        }

        return {
            x: (width - brainWidth) / 2,
            y: outerTop + (availableHeight - brainHeight) / 2,
            width: brainWidth,
            height: brainHeight,
        };
    }

    function _rebuildStaticLayer(size, layout) {
        const base = staticLayer.getContext('2d');
        const { width, height, dpr } = size;
        base.setTransform(dpr, 0, 0, dpr, 0, 0);
        base.clearRect(0, 0, width, height);

        const background = base.createRadialGradient(
            width * 0.5, height * 0.48, 10,
            width * 0.5, height * 0.5, Math.max(width, height) * 0.65
        );
        background.addColorStop(0, '#07131b');
        background.addColorStop(0.48, '#03090f');
        background.addColorStop(1, '#010305');
        base.fillStyle = background;
        base.fillRect(0, 0, width, height);

        // Real soma-density silhouette. It gives the point cloud a coherent
        // anatomical volume without inventing a synthetic brain outline.
        const density = document.createElement('canvas');
        density.width = 256;
        density.height = Math.round(256 / ANATOMY_ASPECT);
        const densityCtx = density.getContext('2d');
        densityCtx.fillStyle = 'rgba(94, 183, 220, 0.075)';
        for (const [px, py] of data.points) {
            densityCtx.fillRect(px * density.width, py * density.height, 1.6, 1.6);
        }

        base.save();
        base.globalCompositeOperation = 'screen';
        base.globalAlpha = 0.75;
        base.filter = 'blur(10px)';
        base.drawImage(density, layout.x, layout.y, layout.width, layout.height);
        base.filter = 'blur(3px)';
        base.globalAlpha = 0.34;
        base.drawImage(density, layout.x, layout.y, layout.width, layout.height);
        base.restore();

        // Quiet midline: orientation aid, not an anatomical boundary.
        base.save();
        base.strokeStyle = 'rgba(104, 174, 205, 0.09)';
        base.lineWidth = 1;
        base.setLineDash([2, 5]);
        const midX = layout.x + layout.width * 0.515;
        base.beginPath();
        base.moveTo(midX, layout.y + layout.height * 0.12);
        base.lineTo(midX, layout.y + layout.height * 0.88);
        base.stroke();
        base.restore();

        // Batch soma circles by neurotransmitter for cleaner, faster rendering.
        for (let nt = 0; nt < NT_COLORS.length; nt++) {
            const col = NT_COLORS[nt];
            base.beginPath();
            for (const [px, py, pointNt] of data.points) {
                if (pointNt !== nt) continue;
                const x = layout.x + px * layout.width;
                const y = layout.y + py * layout.height;
                const radius = nt === 6 ? 0.45 : 0.62;
                base.moveTo(x + radius, y);
                base.arc(x, y, radius, 0, Math.PI * 2);
            }
            base.fillStyle = `rgba(${col[0]}, ${col[1]}, ${col[2]}, ${nt === 6 ? 0.16 : 0.30})`;
            base.fill();
        }

        _drawAnatomyLabels(base, layout);
        _drawHeaderAndLegend(base, size);
        staticDirty = false;
    }

    function _drawAnatomyLabels(target, layout) {
        const labels = [
            ['OPTIC LOBE', 0.13, 0.58],
            ['CENTRAL BRAIN', 0.515, 0.31],
            ['OPTIC LOBE', 0.89, 0.58],
            ['ANTENNAL / SEZ', 0.515, 0.79],
        ];

        target.save();
        target.font = '600 8px "JetBrains Mono", monospace';
        target.textAlign = 'center';
        target.textBaseline = 'middle';
        for (const [label, px, py] of labels) {
            const x = layout.x + px * layout.width;
            const y = layout.y + py * layout.height;
            const metrics = target.measureText(label);
            target.fillStyle = 'rgba(1, 6, 10, 0.62)';
            target.fillRect(x - metrics.width / 2 - 4, y - 6, metrics.width + 8, 12);
            target.fillStyle = 'rgba(132, 180, 199, 0.43)';
            target.fillText(label, x, y);
        }
        target.restore();
    }

    function _drawHeaderAndLegend(target, size) {
        const { width } = size;
        target.save();
        target.font = '600 8px "JetBrains Mono", monospace';
        target.textBaseline = 'middle';
        target.textAlign = 'left';
        target.fillStyle = 'rgba(131, 180, 201, 0.72)';
        target.fillText(
            `${data.total.toLocaleString()} NEURONS  ·  ${data.sampled.toLocaleString()} SOMA SAMPLE  ·  DORSAL`,
            9, 10
        );

        const legend = [
            ['ACh', 0], ['GABA', 1], ['GLUT', 2],
            ['DA', 3], ['5-HT', 4], ['OA', 5],
        ];
        let x = Math.max(9, width - 242);
        for (const [label, nt] of legend) {
            const col = NT_COLORS[nt];
            target.beginPath();
            target.arc(x, 10, 2, 0, Math.PI * 2);
            target.fillStyle = `rgb(${col[0]}, ${col[1]}, ${col[2]})`;
            target.fill();
            target.fillStyle = 'rgba(151, 172, 190, 0.68)';
            target.fillText(label, x + 5, 10);
            x += label.length * 5 + 17;
        }
        target.restore();
    }

    function _render(now) {
        if (!ctx || !data) return;
        const size = _ensureCanvasSize();
        const layout = _brainLayout(size.width, size.height);

        if (staticDirty) _rebuildStaticLayer(size, layout);

        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.drawImage(staticLayer, 0, 0);
        ctx.setTransform(size.dpr, 0, 0, size.dpr, 0, 0);

        const peak = Math.max(
            0,
            ...Object.values(regionActivity),
            ...Object.values(popActivity),
        );
        if (peak > 0.035) {
            ctx.fillStyle = `rgba(0, 3, 8, ${Math.min(0.22, peak * 0.18)})`;
            ctx.fillRect(0, 0, size.width, size.height);
        }

        _drawRegionActivity(ctx, layout, now);
        _drawPopulationActivity(ctx, layout);
        _drawActivitySummary(ctx, size);
        _drawVignette(ctx, size);
    }

    function _drawRegionActivity(target, layout, now) {
        target.save();
        target.globalCompositeOperation = 'screen';

        for (const [regionName, activity] of Object.entries(regionActivity)) {
            if (activity < 0.012) continue;
            const info = data.regions[regionName];
            if (!info) continue;

            // The optic superclass contains both hemispheres; show both instead
            // of placing one misleading glow at their combined centroid.
            const centers = regionName === 'optic'
                ? [[0.14, 0.57], [0.89, 0.57]]
                : [[info.x, info.y]];

            for (const [px, py] of centers) {
                const x = layout.x + px * layout.width;
                const y = layout.y + py * layout.height;
                const radius = Math.max(16, Math.min(92, info.spread * layout.width * 0.32));
                const warm = regionName === 'optic' || regionName === 'visual_projection';
                const rgb = warm ? [255, 218, 112] : [92, 174, 245];
                const pulse = 0.92 + 0.08 * Math.sin(now * 0.006 + px * 9);
                const gradient = target.createRadialGradient(x, y, 0, x, y, radius);
                gradient.addColorStop(0, `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${activity * 0.68})`);
                gradient.addColorStop(0.24, `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${activity * 0.34})`);
                gradient.addColorStop(0.62, `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${activity * 0.12})`);
                gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
                target.fillStyle = gradient;
                target.beginPath();
                target.arc(x, y, radius, 0, Math.PI * 2);
                target.fill();

                target.strokeStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${0.2 + activity * 0.7})`;
                target.lineWidth = 0.8 + activity * 1.8;
                target.shadowColor = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${activity})`;
                target.shadowBlur = 5 + activity * 13;
                target.beginPath();
                target.arc(x, y, radius * (0.42 + 0.05 * pulse), 0, Math.PI * 2);
                target.stroke();
                target.shadowBlur = 0;
            }
        }
        target.restore();
    }

    function _drawPopulationActivity(target, layout) {
        target.save();
        target.globalCompositeOperation = 'screen';

        for (const [name, points] of Object.entries(data.pop_points)) {
            const activity = popActivity[name] || 0;
            if (activity < 0.008 || points.length === 0) continue;
            const col = POP_COLORS[name] || [210, 220, 230];

            const centerX = points.reduce((sum, point) => sum + point[0], 0) / points.length;
            const centerY = points.reduce((sum, point) => sum + point[1], 0) / points.length;
            const cx = layout.x + centerX * layout.width;
            const cy = layout.y + centerY * layout.height;
            const haloRadius = 14 + 32 * activity;
            const halo = target.createRadialGradient(cx, cy, 0, cx, cy, haloRadius);
            halo.addColorStop(0, `rgba(${col[0]}, ${col[1]}, ${col[2]}, ${activity * 0.58})`);
            halo.addColorStop(0.4, `rgba(${col[0]}, ${col[1]}, ${col[2]}, ${activity * 0.18})`);
            halo.addColorStop(1, 'rgba(0, 0, 0, 0)');
            target.fillStyle = halo;
            target.beginPath();
            target.arc(cx, cy, haloRadius, 0, Math.PI * 2);
            target.fill();

            const radius = 0.9 + activity * 2.05;
            target.beginPath();
            for (const [px, py] of points) {
                const x = layout.x + px * layout.width;
                const y = layout.y + py * layout.height;
                target.moveTo(x + radius, y);
                target.arc(x, y, radius, 0, Math.PI * 2);
            }
            target.shadowColor = `rgba(${col[0]}, ${col[1]}, ${col[2]}, ${activity})`;
            target.shadowBlur = 5 + activity * 11;
            target.fillStyle = `rgba(${col[0]}, ${col[1]}, ${col[2]}, ${0.2 + activity * 0.8})`;
            target.fill();
            target.shadowBlur = 0;

            if (activity > 0.12) {
                target.font = '700 8px "JetBrains Mono", monospace';
                target.textAlign = 'center';
                target.fillStyle = `rgba(${col[0]}, ${col[1]}, ${col[2]}, ${0.35 + activity * 0.65})`;
                target.fillText(`${name} · ${points.length}`, cx, cy - haloRadius - 4);
            }
        }
        target.restore();
    }

    function _drawActivitySummary(target, size) {
        const active = Object.entries(regionActivity)
            .filter(([, value]) => value >= 0.08)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3);
        if (active.length === 0) return;

        target.save();
        target.font = '700 8px "JetBrains Mono", monospace';
        target.textBaseline = 'middle';
        target.textAlign = 'left';
        let x = 9;
        const y = size.height - 10;
        target.fillStyle = 'rgba(151, 172, 190, 0.58)';
        target.fillText('ACTIVE', x, y);
        x += 39;

        for (const [name, value] of active) {
            const label = `${name.replaceAll('_', ' ').toUpperCase()} ${Math.round(value * 100)}%`;
            const width = target.measureText(label).width + 11;
            target.fillStyle = 'rgba(12, 37, 52, 0.86)';
            target.strokeStyle = 'rgba(92, 174, 245, 0.58)';
            target.lineWidth = 1;
            target.beginPath();
            target.roundRect(x, y - 6, width, 12, 3);
            target.fill();
            target.stroke();
            target.fillStyle = 'rgba(186, 229, 250, 0.92)';
            target.fillText(label, x + 5, y);
            x += width + 5;
            if (x > size.width - 100) break;
        }
        target.restore();
    }

    function _drawVignette(target, size) {
        const gradient = target.createRadialGradient(
            size.width / 2, size.height / 2, Math.min(size.width, size.height) * 0.25,
            size.width / 2, size.height / 2, Math.max(size.width, size.height) * 0.65
        );
        gradient.addColorStop(0, 'rgba(0, 0, 0, 0)');
        gradient.addColorStop(1, 'rgba(0, 2, 6, 0.34)');
        target.fillStyle = gradient;
        target.fillRect(0, 0, size.width, size.height);
    }

    return { init, update };
})();
