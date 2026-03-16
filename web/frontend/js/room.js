/**
 * Three.js room with NeuroMechFly — per-mesh joint animation.
 *
 * Key architecture:
 *   - glTF meshes: vertices in LOCAL frame (mm), all at origin
 *   - flyModel (parent): rotation -90°X + scale → converts MuJoCo→Three.js
 *   - Per-mesh: position = geom_xpos, quaternion = geom_xmat (MuJoCo frame)
 *   - Parent transform handles coordinate conversion automatically
 */

const Room = (() => {
    let scene, camera, renderer, controls, clock;
    let flyGroup, flyModel, flyGlow;
    let trailPoints = [], trailLine;

    let geomMeshMap = {};    // name → THREE.Mesh
    let geomNames = [];      // from fly_pose.json

    // Brain-driven animation: replay cached walk frames based on DN rates
    let walkCache = null;     // { geomNames, frames } from walk cache
    let walkFrameIdx = 0;
    let flyHeading = 0;       // accumulated heading angle

    const BEHAVIOR_COLORS = {
        walking:  new THREE.Color(0x2ecc71),
        escape:   new THREE.Color(0xe74c3c),
        flight:   new THREE.Color(0x9b59b6),
        grooming: new THREE.Color(0x00bcd4),
        feeding:  new THREE.Color(0xff9800),
    };

    function init(canvas) {
        clock = new THREE.Clock();
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x87ceeb);
        scene.fog = new THREE.Fog(0x87ceeb, 60, 120);

        camera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.1, 500);
        camera.position.set(6, 4, 10);

        renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        renderer.setSize(canvas.clientWidth, canvas.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.shadowMap.enabled = true;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.2;

        controls = new THREE.OrbitControls(camera, canvas);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.target.set(0, 1, 0);
        controls.minDistance = 2;
        controls.maxDistance = 80;

        scene.add(new THREE.AmbientLight(0x8899aa, 1.0));
        const sun = new THREE.DirectionalLight(0xfffae6, 1.8);
        sun.position.set(40, 60, -50);
        sun.castShadow = true;
        sun.shadow.mapSize.set(2048, 2048);
        const sc = sun.shadow.camera;
        sc.near = 1; sc.far = 150; sc.left = -40; sc.right = 40; sc.top = 40; sc.bottom = -40;
        scene.add(sun);
        // Hemisphere light (sky blue + ground brown)
        scene.add(new THREE.HemisphereLight(0x87ceeb, 0xd4b876, 0.4));

        buildRoom();
        buildFlyGroup();
        buildTrail();
        loadFly();

        window.addEventListener('resize', () => {
            camera.aspect = canvas.clientWidth / canvas.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(canvas.clientWidth, canvas.clientHeight);
        });
        animate();
    }

    function buildRoom() {
        // === SANDY GROUND ===
        const sandTex = _makeCheckerTexture(0xd4b876, 0xc9a85c, 64);
        const ground = new THREE.Mesh(
            new THREE.PlaneGeometry(200, 200),
            new THREE.MeshStandardMaterial({ map: sandTex, roughness: 0.95, metalness: 0 })
        );
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        scene.add(ground);

        // === SKY (gradient dome) ===
        const skyGeo = new THREE.SphereGeometry(90, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2);
        const skyMat = new THREE.MeshBasicMaterial({
            color: 0x87ceeb, side: THREE.BackSide,
        });
        const sky = new THREE.Mesh(skyGeo, skyMat);
        scene.add(sky);

        // === SUN ===
        const sunGeo = new THREE.SphereGeometry(3, 16, 16);
        const sunMat = new THREE.MeshBasicMaterial({ color: 0xffee88, emissive: 0xffdd44 });
        const sun = new THREE.Mesh(sunGeo, sunMat);
        sun.position.set(40, 60, -50);
        scene.add(sun);
        // Sun glow
        const sunGlow = new THREE.PointLight(0xffeecc, 0.8, 200);
        sunGlow.position.copy(sun.position);
        scene.add(sunGlow);

        // === TREES / BUSHES (background greenery) ===
        const treeMat = new THREE.MeshStandardMaterial({ color: 0x2d5a1e, roughness: 0.8 });
        const trunkMat = new THREE.MeshStandardMaterial({ color: 0x4a3220, roughness: 0.9 });

        // Trees at various distances
        const treePositions = [
            [-30, -40], [-15, -45], [10, -50], [35, -42], [50, -48],
            [-45, -35], [55, -38], [-25, -55], [25, -55], [0, -60],
        ];
        treePositions.forEach(([x, z]) => {
            const height = 8 + Math.random() * 6;
            const radius = 3 + Math.random() * 3;
            // Trunk
            const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.5, height * 0.5, 6), trunkMat);
            trunk.position.set(x, height * 0.25, z);
            trunk.castShadow = true;
            scene.add(trunk);
            // Canopy
            const canopy = new THREE.Mesh(new THREE.SphereGeometry(radius, 8, 6), treeMat);
            canopy.position.set(x, height * 0.6, z);
            canopy.castShadow = true;
            scene.add(canopy);
        });

        // Small bushes closer
        const bushMat = new THREE.MeshStandardMaterial({ color: 0x3a7a28, roughness: 0.85 });
        [[-12, -15], [15, -18], [-20, -10], [22, -12], [8, -20], [-8, -22]].forEach(([x, z]) => {
            const bush = new THREE.Mesh(new THREE.SphereGeometry(1.5 + Math.random(), 6, 5), bushMat);
            bush.position.set(x, 0.8, z);
            bush.castShadow = true;
            scene.add(bush);
        });

        // === GRASS PATCHES ===
        const grassMat = new THREE.MeshStandardMaterial({ color: 0x5a8a3a, roughness: 0.9 });
        for (let i = 0; i < 15; i++) {
            const gx = (Math.random() - 0.5) * 60;
            const gz = (Math.random() - 0.5) * 60;
            const patch = new THREE.Mesh(
                new THREE.CircleGeometry(1 + Math.random() * 2, 8),
                grassMat
            );
            patch.rotation.x = -Math.PI / 2;
            patch.position.set(gx, 0.01, gz);
            scene.add(patch);
        }

        // === SUGAR WATER PUDDLE (stimulus: sugar) ===
        const sugarMat = new THREE.MeshStandardMaterial({
            color: 0x88ccff, transparent: true, opacity: 0.6,
            roughness: 0.1, metalness: 0.3,
        });
        const puddle = new THREE.Mesh(new THREE.CircleGeometry(3, 16), sugarMat);
        puddle.rotation.x = -Math.PI / 2;
        puddle.position.set(8, 0.02, 5);
        scene.add(puddle);
        // Sugar crystals around puddle
        const crystalMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.3, metalness: 0.1 });
        for (let i = 0; i < 8; i++) {
            const angle = (i / 8) * Math.PI * 2;
            const cr = new THREE.Mesh(new THREE.OctahedronGeometry(0.15, 0), crystalMat);
            cr.position.set(8 + Math.cos(angle) * 2.5, 0.15, 5 + Math.sin(angle) * 2.5);
            cr.rotation.set(Math.random(), Math.random(), 0);
            scene.add(cr);
        }
        // Label
        _addLabel('SUGAR WATER', 8, 1.5, 5, 0x88ccff);

        // === ROTTEN FRUIT (stimulus: or56a - geosmin) ===
        const rotMat = new THREE.MeshStandardMaterial({ color: 0x5a3a1a, roughness: 0.9 });
        const rotFruit = new THREE.Mesh(new THREE.SphereGeometry(0.8, 8, 6), rotMat);
        rotFruit.position.set(-8, 0.5, 6);
        rotFruit.scale.set(1.2, 0.7, 1);
        scene.add(rotFruit);
        // Mold spots
        const moldMat = new THREE.MeshStandardMaterial({ color: 0x2a4a2a, roughness: 1 });
        for (let i = 0; i < 4; i++) {
            const spot = new THREE.Mesh(new THREE.SphereGeometry(0.15, 4, 4), moldMat);
            spot.position.set(-8 + (Math.random()-0.5)*0.8, 0.7, 6 + (Math.random()-0.5)*0.6);
            scene.add(spot);
        }
        _addLabel('ROTTEN FRUIT', -8, 1.5, 6, 0x27ae60);

        // === DARK APPROACHING OBJECT (stimulus: lc4 - looming) ===
        const threatMat = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.5 });
        const threat = new THREE.Mesh(new THREE.SphereGeometry(2, 12, 8), threatMat);
        threat.position.set(0, 3, -10);
        threat.castShadow = true;
        scene.add(threat);
        _addLabel('THREAT', 0, 6, -10, 0xe74c3c);

        // === OBSTACLES / ROCKS (physical barriers) ===
        const rockMat = new THREE.MeshStandardMaterial({ color: 0x888880, roughness: 0.85 });
        [[6, -5, 1.2], [-5, -7, 0.9], [12, 2, 0.7], [-10, 3, 1.0]].forEach(([x, z, s]) => {
            const rock = new THREE.Mesh(
                new THREE.DodecahedronGeometry(s, 1),
                rockMat
            );
            rock.position.set(x, s * 0.4, z);
            rock.castShadow = true;
            scene.add(rock);
        });

        // === BITTER LEAF (stimulus: bitter) ===
        const bitterMat = new THREE.MeshStandardMaterial({ color: 0x1a4a1a, roughness: 0.8 });
        const leaf = new THREE.Mesh(new THREE.PlaneGeometry(2, 3), bitterMat);
        leaf.rotation.x = -Math.PI / 2 + 0.1;
        leaf.position.set(-4, 0.05, -3);
        scene.add(leaf);
        _addLabel('BITTER LEAF', -4, 1.2, -3, 0xc0392b);

        // === DUST CLOUD AREA (stimulus: jo - touch) ===
        const dustMat = new THREE.MeshStandardMaterial({
            color: 0xccbb88, transparent: true, opacity: 0.15, roughness: 1,
        });
        const dust = new THREE.Mesh(new THREE.SphereGeometry(3, 8, 6), dustMat);
        dust.position.set(4, 1.5, -5);
        scene.add(dust);
        _addLabel('DUST', 4, 3.5, -5, 0x00bcd4);
    }

    function _makeCheckerTexture(c1, c2, size) {
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = size;
        const ctx = canvas.getContext('2d');
        const half = size / 2;
        const color1 = '#' + c1.toString(16).padStart(6, '0');
        const color2 = '#' + c2.toString(16).padStart(6, '0');
        ctx.fillStyle = color1; ctx.fillRect(0, 0, size, size);
        ctx.fillStyle = color2;
        ctx.fillRect(0, 0, half, half);
        ctx.fillRect(half, half, half, half);
        const tex = new THREE.CanvasTexture(canvas);
        tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
        tex.repeat.set(40, 40);
        return tex;
    }

    function _addLabel(text, x, y, z, color) {
        // Simple sprite label
        const canvas = document.createElement('canvas');
        canvas.width = 256; canvas.height = 64;
        const ctx = canvas.getContext('2d');
        ctx.font = 'bold 24px monospace';
        ctx.fillStyle = '#' + color.toString(16).padStart(6, '0');
        ctx.textAlign = 'center';
        ctx.fillText(text, 128, 40);
        const tex = new THREE.CanvasTexture(canvas);
        const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: 0.7 });
        const sprite = new THREE.Sprite(mat);
        sprite.position.set(x, y, z);
        sprite.scale.set(4, 1, 1);
        scene.add(sprite);
    }

    function buildFlyGroup() {
        flyGroup = new THREE.Group();
        flyGroup.position.set(0, 0.0, 0);  // ground level
        flyGlow = new THREE.PointLight(0x2ecc71, 0.6, 5);
        flyGlow.position.set(0, 0.3, 0);
        flyGroup.add(flyGlow);
        scene.add(flyGroup);
    }

    function loadFly() {
        const loader = new THREE.GLTFLoader();
        loader.load('/static/assets/neuromechfly.glb', (gltf) => {
            flyModel = gltf.scene;

            // flyModel is the coordinate converter:
            // MuJoCo Z-up → Three.js Y-up, and mm → scene units
            flyModel.rotation.set(-Math.PI / 2, 0, 0);
            flyModel.scale.setScalar(1.0);

            // Index meshes by name
            flyModel.traverse((child) => {
                if (child.isMesh) {
                    child.castShadow = true;
                    if (child.material) {
                        child.material.roughness = 0.6;
                        child.material.metalness = 0.05;
                    }
                    geomMeshMap[child.name] = child;
                }
            });

            flyGroup.add(flyModel);

            // Debug: log ALL node names and types
            const allNames = [];
            flyModel.traverse((c) => {
                allNames.push(`${c.type}:${c.name}`);
            });
            console.log('glTF nodes:', allNames.length, allNames.slice(0, 15));
            console.log('Meshes found:', Object.keys(geomMeshMap).length, Object.keys(geomMeshMap).slice(0, 10));

            // Load initial pose and apply
            fetch('/static/assets/fly_pose.json')
                .then(r => r.json())
                .then(data => {
                    geomNames = data.geom_names;
                    applyPose(data.initial_pose);
                    poseLoaded = true;

                    // Feet are at Z ≈ -1.2mm in MuJoCo (relative to thorax).
                    // After -90° X rotation: Z → Y, so feet at Y ≈ -1.2.
                    // Lift by 1.2 so feet touch flyGroup origin (Y=0 local = table surface).
                    flyModel.position.y = 1.2;
                });
        });
    }

    /**
     * Apply pose to all meshes.
     * pose = flat [px, py, pz, qx, qy, qz, qw] × n_geoms
     * Coordinates are in MuJoCo frame (mm, relative to thorax).
     * flyModel's rotation+scale converts to Three.js automatically.
     */
    function applyPose(pose) {
        for (let i = 0; i < geomNames.length; i++) {
            const mesh = geomMeshMap[geomNames[i]];
            if (!mesh) continue;
            const off = i * 7;
            // Position in MuJoCo mm (parent handles coord swap)
            mesh.position.set(pose[off], pose[off+1], pose[off+2]);
            // Quaternion from MuJoCo xmat (scipy format: x,y,z,w)
            mesh.quaternion.set(pose[off+3], pose[off+4], pose[off+5], pose[off+6]);
        }
    }

    function buildTrail() {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(200 * 3), 3));
        geo.setDrawRange(0, 0);
        trailLine = new THREE.Line(geo, new THREE.LineBasicMaterial({
            color: 0x2ecc71, transparent: true, opacity: 0.4 }));
        scene.add(trailLine);
    }

    // === Public: Walk animation ===
    let currentModel = 'neuromechfly';  // track which glTF is loaded

    function walkInit(data) {
        flyGlow.color.copy(BEHAVIOR_COLORS.walking);
        trailLine.material.color.copy(BEHAVIOR_COLORS.walking);

        // Check if geom names suggest flybody model (has 'thorax' not 'Thorax')
        const names = data ? data.geom_names || [] : [];
        const isFlybody = names.includes('thorax');  // flybody uses lowercase

        if (isFlybody && currentModel !== 'flybody') {
            _loadModel('flybody', '/static/assets/flybody.glb', '/static/assets/flybody_pose.json');
        } else if (!isFlybody && currentModel !== 'neuromechfly') {
            _loadModel('neuromechfly', '/static/assets/neuromechfly.glb', '/static/assets/fly_pose.json');
        }
    }

    function _loadModel(name, glbUrl, poseUrl) {
        currentModel = name;
        // Remove old model
        if (flyModel) {
            flyGroup.remove(flyModel);
            flyModel = null;
        }
        geomMeshMap = {};
        geomNames = [];

        const loader = new THREE.GLTFLoader();
        loader.load(glbUrl, (gltf) => {
            flyModel = gltf.scene;
            flyModel.rotation.set(-Math.PI / 2, 0, 0);
            flyModel.scale.setScalar(name === 'flybody' ? 10.0 : 1.0);
            flyModel.traverse((child) => {
                if (child.isMesh) {
                    child.castShadow = true;
                    if (child.material) {
                        child.material.roughness = 0.6;
                        child.material.metalness = 0.05;
                    }
                    geomMeshMap[child.name] = child;
                }
            });

            // trimesh wraps each mesh in a named Group node — also index those
            flyModel.traverse((node) => {
                if (node.name && !geomMeshMap[node.name]) {
                    geomMeshMap[node.name] = node;
                }
            });

            flyGroup.add(flyModel);
            console.log('Mesh map:', Object.keys(geomMeshMap).length, 'keys. Sample:', Object.keys(geomMeshMap).slice(0, 8));

            fetch(poseUrl).then(r => r.json()).then(poseData => {
                geomNames = poseData.geom_names;
                console.log('Pose names sample:', geomNames.slice(0, 5));
                // Check how many match
                let matched = 0;
                for (const n of geomNames) { if (geomMeshMap[n]) matched++; }
                console.log(`Matched: ${matched}/${geomNames.length}`);

                applyPose(poseData.initial_pose);
                flyModel.updateMatrixWorld(true);
                const box = new THREE.Box3().setFromObject(flyModel);
                flyModel.position.y -= box.min.y;
                flyGroup.position.set(0, 0, 0);  // reset to ground
                console.log(`${name} ready. Box Y: [${box.min.y.toFixed(2)}, ${box.max.y.toFixed(2)}]`);
            });
        });
    }

    function walkUpdate(data) {
        if (!flyGroup || !geomNames.length) return;

        // Apply per-mesh pose (position + quaternion in MuJoCo frame)
        if (data.poses) applyPose(data.poses);

        // Fly global position → scene units
        const fp = data.fly_pos;
        const s = currentModel === 'flybody' ? 3.0 : 0.3;
        const tx = fp[0] * s;
        const ty = 0;  // ground level
        const tz = -fp[1] * s;
        flyGroup.position.x += (tx - flyGroup.position.x) * 0.3;
        flyGroup.position.y += (ty - flyGroup.position.y) * 0.3;
        flyGroup.position.z += (tz - flyGroup.position.z) * 0.3;

        _updateTrail();
        controls.target.lerp(flyGroup.position.clone(), 0.03);
    }

    function walkEnd() {}

    // === Brain-driven animation ===
    // Load walk cache once, then replay frames based on DN rates
    function loadWalkCache() {
        if (walkCache) return;
        fetch('/api/walk_cache')
            .then(r => r.json())
            .then(cache => {
                if (cache.frames && cache.frames.length > 0) {
                    walkCache = cache;
                    // Also store walk geom names for pose application
                    if (cache.geom_names) geomNames = cache.geom_names;
                    console.log('Walk cache loaded:', cache.frames.length, 'frames');
                }
            })
            .catch(() => console.log('No walk cache available'));
    }

    function brainDrive(frame) {
        if (!flyGroup || !frame || !frame.dn) return;

        const dn = frame.dn;
        const fwd = dn.forward || 0;
        const esc = dn.escape || 0;
        const grm = dn.groom || 0;
        const turnL = dn.turn_L || 0;
        const turnR = dn.turn_R || 0;
        const behavior = frame.behavior_mode || 'idle';

        // Behavior color
        const c = BEHAVIOR_COLORS[behavior] || BEHAVIOR_COLORS.walking;
        flyGlow.color.copy(c);
        flyGlow.intensity = 0.4 + Math.max(fwd, esc, grm) * 1.5;
        trailLine.material.color.copy(c);

        // Walk: advance through cached frames + move forward
        if (walkCache && walkCache.frames && fwd > 0.01) {
            const speed = fwd * 2;  // frames to advance per brain frame
            walkFrameIdx = (walkFrameIdx + Math.ceil(speed)) % walkCache.frames.length;
            const wf = walkCache.frames[walkFrameIdx];

            // Apply joint poses from walk cache
            if (wf.poses && geomNames.length > 0) {
                applyPose(wf.poses);
            }

            // Move fly forward in its heading direction
            const moveSpeed = fwd * 0.15;
            flyGroup.position.x += Math.sin(flyHeading) * moveSpeed;
            flyGroup.position.z += Math.cos(flyHeading) * moveSpeed;
        }

        // Turn
        const turn = (turnL - turnR) * 0.05;
        flyHeading += turn;
        flyGroup.rotation.y = flyHeading;

        // Escape: jump up
        if (esc > 0.06) {
            flyGroup.position.y = Math.min(5, flyGroup.position.y + esc * 0.3);
        } else {
            flyGroup.position.y = Math.max(0, flyGroup.position.y - 0.1);
        }

        _updateTrail();
        controls.target.lerp(flyGroup.position.clone(), 0.03);
    }

    // === Public: NT simulation fly update ===
    function updateFly(frame) {
        if (!flyGroup || !frame) return;
        const s = 0.3;
        flyGroup.position.x += (frame.pos[0]*s - flyGroup.position.x) * 0.3;
        flyGroup.position.y += (frame.pos[2]*s - flyGroup.position.y) * 0.3;
        flyGroup.position.z += (frame.pos[1]*s - flyGroup.position.z) * 0.3;
        if (frame.drive[0] !== 0 || frame.drive[1] !== 0)
            flyGroup.rotation.y += (frame.drive[0] - frame.drive[1]) * 0.025;
        const c = BEHAVIOR_COLORS[frame.behavior_mode] || BEHAVIOR_COLORS.walking;
        flyGlow.color.copy(c);
        trailLine.material.color.copy(c);
        _updateTrail();
        controls.target.lerp(flyGroup.position.clone(), 0.03);
    }

    function resetTrail() {
        trailPoints = [];
        if (trailLine) trailLine.geometry.setDrawRange(0, 0);
    }

    function _updateTrail() {
        trailPoints.push(flyGroup.position.clone());
        if (trailPoints.length > 200) trailPoints.shift();
        const pa = trailLine.geometry.getAttribute('position');
        for (let i = 0; i < trailPoints.length; i++)
            pa.setXYZ(i, trailPoints[i].x, trailPoints[i].y, trailPoints[i].z);
        pa.needsUpdate = true;
        trailLine.geometry.setDrawRange(0, trailPoints.length);
    }

    function animate() {
        requestAnimationFrame(animate);
        clock.getDelta();
        controls.update();
        renderer.render(scene, camera);
    }

    return { init, updateFly, resetTrail, walkInit, walkUpdate, walkEnd, loadWalkCache, brainDrive };
})();
