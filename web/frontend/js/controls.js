/**
 * Controls: scenario picker, start/stop, phase timeline.
 */

const Controls = (() => {
    let scenarios = [];
    let selectedScenario = 'escape';
    let currentScenario = null;  // full scenario object from backend

    const PHASE_COLORS = {
        hunger: '#f39c12', looming: '#e74c3c', free: '#3498db',
        search: '#f39c12', sugar: '#2ecc71', feeding: '#ff9800',
        walking: '#27ae60', touch: '#00bcd4',
    };

    async function init() {
        try {
            const res = await fetch('/api/scenarios');
            scenarios = await res.json();
        } catch (e) {
            // Fallback
            scenarios = [
                { name: 'escape', label: 'Hunger → Escape', description: '' },
                { name: 'foraging', label: 'Sugar Foraging', description: '' },
                { name: 'grooming', label: 'Touch → Grooming', description: '' },
            ];
        }

        const container = document.getElementById('scenario-buttons');
        container.innerHTML = '';
        scenarios.forEach(s => {
            const btn = document.createElement('button');
            btn.textContent = s.label;
            btn.title = s.description;
            btn.dataset.scenario = s.name;
            if (s.name === selectedScenario) btn.classList.add('active');
            btn.addEventListener('click', () => {
                selectedScenario = s.name;
                container.querySelectorAll('button').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
            container.appendChild(btn);
        });
    }

    function getSelected() {
        return selectedScenario;
    }

    function buildPhaseBar(scenarioData) {
        currentScenario = scenarioData;
        const bar = document.getElementById('phase-bar');
        bar.innerHTML = '';

        if (!scenarioData || !scenarioData.phases) return;

        const totalMs = scenarioData.duration_s * 1000;
        scenarioData.phases.forEach(p => {
            const width = ((p.end_ms - p.start_ms) / totalMs) * 100;
            const seg = document.createElement('div');
            seg.className = 'phase-segment';
            seg.style.width = width + '%';
            seg.style.background = PHASE_COLORS[p.name] || '#555';
            seg.textContent = p.name;
            bar.appendChild(seg);
        });
    }

    return {
        init,
        getSelected,
        buildPhaseBar,
        get currentScenario() { return currentScenario; },
    };
})();
