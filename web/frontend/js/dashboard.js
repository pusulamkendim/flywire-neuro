/**
 * Dashboard: bar gauges for DN outputs, dopamine, neuromodulators, bulk NT.
 */

const Dashboard = (() => {
    const DN_BARS = [
        { key: 'escape',  label: 'GF Escape/Flight', color: '#e74c3c' },
        { key: 'forward', label: 'P9 Forward',        color: '#2ecc71' },
        { key: 'backward',label: 'MDN Backward',      color: '#9b59b6' },
        { key: 'turn_L',  label: 'DNa01/02 Turn L',   color: '#3498db' },
        { key: 'turn_R',  label: 'DNa01/02 Turn R',   color: '#e67e22' },
        { key: 'groom',   label: 'aDN1 Groom',        color: '#00bcd4' },
        { key: 'feed',    label: 'MN9 Feed',          color: '#ff9800' },
    ];

    const DA_BARS = [
        { key: 'pam',           label: 'PAM Reward',   color: '#2ecc71' },
        { key: 'ppl1',          label: 'PPL1 Punish',  color: '#e74c3c' },
        { key: 'mbon_approach', label: 'MBON App',     color: '#27ae60' },
        { key: 'mbon_avoidance',label: 'MBON Avd',     color: '#c0392b' },
        { key: 'mbon_suppress', label: 'MBON Sup',     color: '#2980b9' },
    ];

    const NM_BARS = [
        { key: 'serotonin',  label: '5-HT',  color: '#9b59b6' },
        { key: 'octopamine', label: 'OA',     color: '#e67e22' },
    ];

    const NT_BARS = [
        { key: 'gaba', label: 'GABA',  color: '#3498db' },
        { key: 'ach',  label: 'ACh',   color: '#27ae60' },
        { key: 'glut', label: 'Glut',  color: '#e84393' },
    ];

    // Chart
    const CHART_HISTORY = 100;
    const chartData = {};
    let chartCtx = null;
    const chartKeys = ['pam', 'ppl1', 'gaba', 'ach', 'serotonin', 'escape', 'forward'];
    const chartColors = {
        pam: '#2ecc71', ppl1: '#e74c3c', gaba: '#3498db',
        ach: '#27ae60', serotonin: '#9b59b6', escape: '#e74c3c', forward: '#2ecc71',
    };

    let ntMaxValues = {};

    function init() {
        _buildBars('dn-bars', DN_BARS);
        _buildBars('da-bars', DA_BARS);
        _buildBars('nm-bars', NM_BARS);
        _buildBars('nt-bars', NT_BARS);

        chartKeys.forEach(k => { chartData[k] = new Array(CHART_HISTORY).fill(0); });
        chartCtx = document.getElementById('nt-chart').getContext('2d');
    }

    function _buildBars(containerId, defs) {
        const container = document.getElementById(containerId);
        if (!container) return;
        container.innerHTML = '';
        defs.forEach(d => {
            container.innerHTML += `
                <div class="bar-row">
                    <span class="bar-label">${d.label}</span>
                    <div class="bar-track">
                        <div class="bar-fill" id="bar-${d.key}" style="width:0%;background:${d.color}"></div>
                    </div>
                    <span class="bar-value" id="val-${d.key}">0</span>
                </div>`;
        });
    }

    function update(frame) {
        if (!frame) return;

        // DN bars (0-1 range from brain data)
        DN_BARS.forEach(d => {
            const v = frame.dn ? (frame.dn[d.key] || 0) : (frame['dn_' + d.key] || frame[d.key] || 0);
            const pct = Math.min(100, v * 100);
            const el = document.getElementById(`bar-${d.key}`);
            const vel = document.getElementById(`val-${d.key}`);
            if (el) el.style.width = pct + '%';
            if (vel) vel.textContent = v.toFixed(3);
        });

        // Population bars (auto-scaling)
        [...DA_BARS, ...NM_BARS, ...NT_BARS].forEach(d => {
            const v = frame.pop ? (frame.pop[d.key] || 0) : (frame[d.key] || 0);
            ntMaxValues[d.key] = Math.max(ntMaxValues[d.key] || 1, v, 1);
            const pct = Math.min(100, (v / ntMaxValues[d.key]) * 100);
            const el = document.getElementById(`bar-${d.key}`);
            const vel = document.getElementById(`val-${d.key}`);
            if (el) el.style.width = pct + '%';
            if (vel) vel.textContent = v;
        });

        // Info
        if (frame.t_ms !== undefined) {
            const el = document.getElementById('info-time');
            if (el) el.textContent = frame.t_ms.toFixed(1) + ' ms';
        }
        if (frame.behavior_mode) {
            const el = document.getElementById('info-behavior');
            if (el) el.textContent = frame.behavior_mode;
        }
        if (frame.total_spikes !== undefined) {
            const el = document.getElementById('info-spikes');
            if (el) el.textContent = frame.total_spikes;
        }
        if (frame.brain_steps !== undefined) {
            const el = document.getElementById('info-steps');
            if (el) el.textContent = frame.brain_steps;
        }

        // Flight state (derived from GF rate)
        const gf = frame.dn ? (frame.dn.escape || 0) : (frame.dn_escape || 0);
        const flightEl = document.getElementById('info-flight');
        if (flightEl) {
            if (gf > 0.3) flightEl.textContent = 'FLYING';
            else if (gf > 0.06) flightEl.textContent = 'TAKEOFF';
            else flightEl.textContent = 'GROUNDED';
            flightEl.style.color = gf > 0.06 ? '#e74c3c' : 'var(--text-primary)';
        }

        // Drive L/R (derived from forward ± turn)
        const fwd = frame.dn ? (frame.dn.forward || 0) : (frame.dn_forward || 0);
        const turnL = frame.dn ? (frame.dn.turn_L || 0) : (frame.dn_turn_L || 0);
        const turnR = frame.dn ? (frame.dn.turn_R || 0) : (frame.dn_turn_R || 0);
        const bkw = frame.dn ? (frame.dn.backward || 0) : (frame.dn_backward || 0);
        const turn = turnL - turnR;
        const driveL = Math.max(0, fwd * (1 + turn) - bkw);
        const driveR = Math.max(0, fwd * (1 - turn) - bkw);
        const driveEl = document.getElementById('info-drive');
        if (driveEl) driveEl.textContent = `${driveL.toFixed(2)} / ${driveR.toFixed(2)}`;

        // Badge
        const badge = document.getElementById('behavior-badge');
        if (badge && frame.behavior_mode) {
            badge.textContent = frame.behavior_mode.toUpperCase();
        }

        // Time display
        if (frame.t_ms !== undefined) {
            const td = document.getElementById('time-display');
            if (td) td.textContent = `t = ${frame.t_ms.toFixed(1)} ms`;
        }

        _updateChart(frame);
    }

    function _updateChart(frame) {
        if (!chartCtx) return;

        chartKeys.forEach(k => {
            let v = 0;
            if (frame.dn && frame.dn[k] !== undefined) v = frame.dn[k] * 100;
            else if (frame.pop && frame.pop[k] !== undefined) v = frame.pop[k];
            else if (frame[k] !== undefined) v = frame[k];
            chartData[k].push(v);
            if (chartData[k].length > CHART_HISTORY) chartData[k].shift();
        });

        const canvas = chartCtx.canvas;
        const w = canvas.width = canvas.clientWidth;
        const h = canvas.height = canvas.clientHeight;
        chartCtx.clearRect(0, 0, w, h);

        let maxVal = 1;
        chartKeys.forEach(k => {
            const m = Math.max(...chartData[k]);
            if (m > maxVal) maxVal = m;
        });

        chartKeys.forEach(k => {
            chartCtx.strokeStyle = chartColors[k] || '#888';
            chartCtx.lineWidth = 1.5;
            chartCtx.beginPath();
            for (let i = 0; i < chartData[k].length; i++) {
                const x = (i / (CHART_HISTORY - 1)) * w;
                const y = h - (chartData[k][i] / maxVal) * (h - 4) - 2;
                if (i === 0) chartCtx.moveTo(x, y);
                else chartCtx.lineTo(x, y);
            }
            chartCtx.stroke();
        });

        let ly = 12;
        chartKeys.forEach(k => {
            chartCtx.fillStyle = chartColors[k] || '#888';
            chartCtx.font = '9px monospace';
            chartCtx.fillText(k.toUpperCase(), w - 65, ly);
            ly += 12;
        });
    }

    function resetChart() {
        chartKeys.forEach(k => { chartData[k] = new Array(CHART_HISTORY).fill(0); });
        ntMaxValues = {};
    }

    return { init, update, resetChart };
})();
