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
    let poseLoaded = false;

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
        scene.background = new THREE.Color(0x0a0a12);
        scene.fog = new THREE.Fog(0x0a0a12, 80, 200);

        camera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.1, 500);
        camera.position.set(4, 6, 6);

        renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        renderer.setSize(canvas.clientWidth, canvas.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.shadowMap.enabled = true;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.2;

        controls = new THREE.OrbitControls(camera, canvas);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.target.set(0, 5, 0);
        controls.minDistance = 2;
        controls.maxDistance = 80;

        scene.add(new THREE.AmbientLight(0x556688, 0.8));
        const sun = new THREE.DirectionalLight(0xffeedd, 1.4);
        sun.position.set(20, 30, 15);
        sun.castShadow = true;
        sun.shadow.mapSize.set(1024, 1024);
        scene.add(sun);

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
        const floor = new THREE.Mesh(new THREE.PlaneGeometry(120, 120),
            new THREE.MeshStandardMaterial({ color: 0x1a1a2e, roughness: 0.9 }));
        floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true;
        scene.add(floor);
        scene.add(new THREE.GridHelper(120, 120, 0x222244, 0x181830));

        const tMat = new THREE.MeshStandardMaterial({ color: 0x2a1f14, roughness: 0.7 });
        const table = new THREE.Mesh(new THREE.BoxGeometry(50, 0.5, 30), tMat);
        table.position.set(0, 4, 0); table.castShadow = true; table.receiveShadow = true;
        scene.add(table);
        [[-24,2,-14],[24,2,-14],[-24,2,14],[24,2,14]].forEach(p => {
            const l = new THREE.Mesh(new THREE.BoxGeometry(0.8, 4, 0.8), tMat);
            l.position.set(...p); l.castShadow = true; scene.add(l);
        });

        const wall = new THREE.Mesh(new THREE.PlaneGeometry(120, 30),
            new THREE.MeshStandardMaterial({ color: 0x141428 }));
        wall.position.set(0, 15, -60); scene.add(wall);
        const win = new THREE.Mesh(new THREE.PlaneGeometry(20, 12),
            new THREE.MeshStandardMaterial({ color: 0x88aacc, emissive: 0x334466,
                emissiveIntensity: 0.3, transparent: true, opacity: 0.3 }));
        win.position.set(0, 14, -59.9); scene.add(win);
    }

    function buildFlyGroup() {
        flyGroup = new THREE.Group();
        flyGroup.position.set(0, 4.2, 0);  // table surface
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
    function walkInit() {
        flyGlow.color.copy(BEHAVIOR_COLORS.walking);
        trailLine.material.color.copy(BEHAVIOR_COLORS.walking);
    }

    function walkUpdate(data) {
        if (!flyGroup || !geomNames.length) return;

        // Apply per-mesh pose (position + quaternion in MuJoCo frame)
        if (data.poses) applyPose(data.poses);

        // Fly global position (mm → scene units), Y fixed to table
        const fp = data.fly_pos;
        const s = 0.3;
        const tx = fp[0] * s;
        const tz = -fp[1] * s;
        flyGroup.position.x += (tx - flyGroup.position.x) * 0.3;
        flyGroup.position.y = 4.25;  // always on table surface
        flyGroup.position.z += (tz - flyGroup.position.z) * 0.3;

        _updateTrail();
        controls.target.lerp(flyGroup.position.clone(), 0.03);
    }

    function walkEnd() {}

    // === Public: NT simulation fly update ===
    function updateFly(frame) {
        if (!flyGroup || !frame) return;
        const s = 0.3;
        flyGroup.position.x += (frame.pos[0]*s - flyGroup.position.x) * 0.3;
        flyGroup.position.y += (4.2 + frame.pos[2]*s - flyGroup.position.y) * 0.3;
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

    return { init, updateFly, resetTrail, walkInit, walkUpdate, walkEnd };
})();
