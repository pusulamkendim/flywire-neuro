/**
 * Dashboard: bar gauges, info panel, NT time-series chart.
 */

const Dashboard = (() => {
    // Bar definitions
    const DN_BARS = [
        { key: 'dn_escape',   label: 'GF Escape', color: 'var(--accent-escape)' },
        { key: 'dn_forward',  label: 'P9 Forward', color: 'var(--accent-forward)' },
        { key: 'dn_backward', label: 'MDN Back',   color: '#9b59b6' },
        { key: 'dn_turn_L',   label: 'Turn L',     color: '#3498db' },
        { key: 'dn_turn_R',   label: 'Turn R',     color: '#e67e22' },
        { key: 'dn_groom',    label: 'Groom',      color: 'var(--accent-groom)' },
        { key: 'dn_feed',     label: 'Feed',        color: 'var(--accent-feed)' },
    ];

    const DRIVE_BARS = [
        { key: 0, label: 'Left',  color: '#3498db' },
        { key: 1, label: 'Right', color: '#e67e22' },
    ];

    const NT_BARS = [
        { key: 'pam',           label: 'PAM',       color: 'var(--accent-pam)' },
        { key: 'ppl1',          label: 'PPL1',      color: 'var(--accent-ppl1)' },
        { key: 'mbon_approach', label: 'MBON App',  color: '#27ae60' },
        { key: 'mbon_avoidance',label: 'MBON Avd',  color: '#c0392b' },
        { key: 'mbon_suppress', label: 'MBON Sup',  color: '#2980b9' },
        { key: 'serotonin',    label: '5-HT',       color: 'var(--accent-serotonin)' },
        { key: 'octopamine',   label: 'OA',         color: 'var(--accent-octopamine)' },
        { key: 'gaba',         label: 'GABA',       color: 'var(--accent-gaba)' },
        { key: 'ach',          label: 'ACh',        color: 'var(--accent-ach)' },
        { key: 'glut',         label: 'Glut',       color: 'var(--accent-glut)' },
    ];

    // NT chart state
    const CHART_HISTORY = 100;
    const chartData = {};
    let chartCtx = null;
    const chartKeys = ['pam', 'ppl1', 'gaba', 'ach', 'serotonin'];
    const chartColors = {
        pam: '#2ecc71', ppl1: '#e74c3c', gaba: '#3498db',
        ach: '#27ae60', serotonin: '#9b59b6',
    };

    // NT max values for normalization (auto-scales)
    let ntMaxValues = {};

    function init() {
        _buildBars('dn-bars', DN_BARS);
        _buildBars('drive-bars', DRIVE_BARS);
        _buildBars('nt-bars', NT_BARS);

        // Init chart data
        chartKeys.forEach(k => { chartData[k] = new Array(CHART_HISTORY).fill(0); });
        chartCtx = document.getElementById('nt-chart').getContext('2d');
    }

    function _buildBars(containerId, defs) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';
        defs.forEach(d => {
            const key = typeof d.key === 'number' ? `drive_${d.key}` : d.key;
            container.innerHTML += `
                <div class="bar-row">
                    <span class="bar-label">${d.label}</span>
                    <div class="bar-track">
                        <div class="bar-fill" id="bar-${key}" style="width:0%;background:${d.color}"></div>
                    </div>
                    <span class="bar-value" id="val-${key}">0</span>
                </div>`;
        });
    }

    function update(frame) {
        if (!frame) return;

        // DN bars (0-1 range)
        DN_BARS.forEach(d => {
            const v = frame[d.key] || 0;
            const pct = Math.min(100, v * 100);
            document.getElementById(`bar-${d.key}`).style.width = pct + '%';
            document.getElementById(`val-${d.key}`).textContent = v.toFixed(3);
        });

        // Drive bars (-0.5 to 1.5, normalize to 0-100)
        DRIVE_BARS.forEach(d => {
            const v = frame.drive[d.key] || 0;
            const pct = Math.min(100, Math.max(0, (v + 0.5) / 2.0 * 100));
            const key = `drive_${d.key}`;
            document.getElementById(`bar-${key}`).style.width = pct + '%';
            document.getElementById(`val-${key}`).textContent = v.toFixed(3);
        });

        // NT bars (auto-scaling)
        NT_BARS.forEach(d => {
            const v = frame[d.key] || 0;
            // Track max for normalization
            ntMaxValues[d.key] = Math.max(ntMaxValues[d.key] || 1, v, 1);
            const pct = Math.min(100, (v / ntMaxValues[d.key]) * 100);
            document.getElementById(`bar-${d.key}`).style.width = pct + '%';
            document.getElementById(`val-${d.key}`).textContent = v;
        });

        // Info panel
        document.getElementById('info-time').textContent = frame.t_ms.toFixed(1) + ' ms';
        document.getElementById('info-phase').textContent = frame.phase;
        document.getElementById('info-behavior').textContent = frame.behavior_mode;
        document.getElementById('info-flight').textContent = frame.flight_state;
        document.getElementById('info-alt').textContent = frame.alt_mm.toFixed(1) + ' mm';
        document.getElementById('info-wing').textContent = frame.wing_freq.toFixed(0) + ' Hz';
        document.getElementById('info-spikes').textContent = frame.total_spikes;

        // Viewport overlay
        document.getElementById('time-display').textContent = `t = ${frame.t_ms.toFixed(1)} ms`;
        document.getElementById('pos-display').textContent =
            `pos = [${frame.pos[0].toFixed(1)}, ${frame.pos[1].toFixed(1)}, ${frame.pos[2].toFixed(1)}]`;

        // Behavior badge
        const badge = document.getElementById('behavior-badge');
        badge.textContent = frame.behavior_mode.toUpperCase();
        const badgeColors = {
            walking: 'var(--accent-forward)', escape: 'var(--accent-escape)',
            flight: 'var(--accent-flight)', grooming: 'var(--accent-groom)',
            feeding: 'var(--accent-feed)',
        };
        badge.style.borderColor = badgeColors[frame.behavior_mode] || 'var(--border)';
        badge.style.color = badgeColors[frame.behavior_mode] || 'var(--text-primary)';

        // NT chart
        _updateChart(frame);

        // Progress needle
        _updateProgress(frame);
    }

    function _updateChart(frame) {
        if (!chartCtx) return;

        // Push new values
        chartKeys.forEach(k => {
            chartData[k].push(frame[k] || 0);
            if (chartData[k].length > CHART_HISTORY) chartData[k].shift();
        });

        const canvas = chartCtx.canvas;
        const w = canvas.width = canvas.clientWidth;
        const h = canvas.height = canvas.clientHeight;
        chartCtx.clearRect(0, 0, w, h);

        // Find max across all series
        let maxVal = 1;
        chartKeys.forEach(k => {
            const m = Math.max(...chartData[k]);
            if (m > maxVal) maxVal = m;
        });

        // Draw lines
        chartKeys.forEach(k => {
            chartCtx.strokeStyle = chartColors[k];
            chartCtx.lineWidth = 1.5;
            chartCtx.beginPath();
            const data = chartData[k];
            for (let i = 0; i < data.length; i++) {
                const x = (i / (CHART_HISTORY - 1)) * w;
                const y = h - (data[i] / maxVal) * (h - 4) - 2;
                if (i === 0) chartCtx.moveTo(x, y);
                else chartCtx.lineTo(x, y);
            }
            chartCtx.stroke();
        });

        // Legend (top-right)
        let ly = 12;
        chartKeys.forEach(k => {
            chartCtx.fillStyle = chartColors[k];
            chartCtx.font = '9px monospace';
            chartCtx.fillText(k.toUpperCase(), w - 65, ly);
            ly += 12;
        });
    }

    function _updateProgress(frame) {
        const needle = document.getElementById('progress-needle');
        const bar = document.getElementById('phase-bar');
        if (!needle || !bar || !Controls.currentScenario) return;

        const scenario = Controls.currentScenario;
        const totalMs = scenario.duration_s * 1000;
        const pct = Math.min(100, (frame.t_ms / totalMs) * 100);
        needle.style.left = pct + '%';
    }

    function resetChart() {
        chartKeys.forEach(k => { chartData[k] = new Array(CHART_HISTORY).fill(0); });
        ntMaxValues = {};
    }

    return { init, update, resetChart };
})();
