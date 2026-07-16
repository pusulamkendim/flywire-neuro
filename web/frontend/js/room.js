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
    let worldGroup = null;
    let worldConfig = null;
    let worldGround = null;
    let worldDebugVisible = false;
    let trailPoints = [], trailLine;

    let geomMeshMap = {};    // name → THREE.Mesh
    let geomNames = [];      // from fly_pose.json
    let activeRenderMode = 'standard';
    const FLIGHT_RENDER_MODES = ['flight_path', 'flight_data_preview', 'flight_explore'];
    const WALKING_DATA_RENDER_MODES = ['walking_data_preview', 'walking_data_explore'];
    const PERSISTENT_RENDER_MODES = ['digital_life', 'proboscis_overlay'];
    const DIRECT_POSITION_RENDER_MODES = [
        'flight_data_preview', 'flight_explore',
        'walking_data_preview', 'walking_data_explore',
        ...PERSISTENT_RENDER_MODES,
    ];
    let lastFlightState = null;
    let preserveLandedOrientation = false;

    // Feeding keeps the NeuromechFly body and overlays only the more detailed
    // flybody mouth parts. This avoids swapping the whole animal mid-scene.
    const LABRUM_NAMES = ['labrum_left_lower', 'labrum_right_lower'];
    const MAIN_MOUTH_MAP = { rostrum: 'Rostrum', haustellum: 'Haustellum' };
    const FEED_TRANSLATION_SCALE = 5.5;  // flybody units → restrained NMF motion
    const FEED_ROTATION_SCALE = 0.68;
    const LABRUM_GEOMETRY_SCALE = 6.0;
    let proboscisModel = null;
    let proboscisMeshMap = {};
    let feedGeomIndices = {};
    let feedMouthReference = null;
    let mainMouthReference = null;

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
        // Millimetre world: keep the near microhabitat crisp while letting the
        // full arena dissolve into atmospheric perspective instead of exposing
        // a hard ground edge.
        scene.fog = new THREE.Fog(0x9bcfe5, 320, 1050);

        camera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.1, 2400);
        camera.position.set(13, 7, 22);

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
        controls.maxDistance = 650;

        scene.add(new THREE.AmbientLight(0x8899aa, 1.0));
        const sun = new THREE.DirectionalLight(0xfffae6, 1.8);
        sun.position.set(40, 60, -50);
        sun.castShadow = true;
        sun.shadow.mapSize.set(2048, 2048);
        const sc = sun.shadow.camera;
        sc.near = 1; sc.far = 900; sc.left = -260; sc.right = 260; sc.top = 260; sc.bottom = -260;
        scene.add(sun);
        // Hemisphere light (sky blue + ground brown)
        scene.add(new THREE.HemisphereLight(0x87ceeb, 0xd4b876, 0.4));

        buildRoom();
        buildFlyGroup();
        buildTrail();
        loadFly();
        loadProboscisOverlay();

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
        worldGround = new THREE.Mesh(
            new THREE.PlaneGeometry(160, 100),
            new THREE.MeshStandardMaterial({ map: sandTex, roughness: 0.95, metalness: 0 })
        );
        worldGround.rotation.x = -Math.PI / 2;
        worldGround.receiveShadow = true;
        scene.add(worldGround);

        // === SKY (gradient dome) ===
        const skyGeo = new THREE.SphereGeometry(1200, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2);
        const skyMat = new THREE.MeshBasicMaterial({
            color: 0x87ceeb, side: THREE.BackSide,
        });
        const sky = new THREE.Mesh(skyGeo, skyMat);
        scene.add(sky);

        _loadWorld('microhabitat_v1');
    }

    async function _loadWorld(worldId) {
        try {
            const response = await fetch(`/api/world/${encodeURIComponent(worldId)}`);
            if (!response.ok) throw new Error(`world request failed: ${response.status}`);
            worldConfig = await response.json();
            _buildConfiguredWorld(worldConfig);
            const scaleEl = document.getElementById('world-scale-display');
            if (scaleEl) scaleEl.textContent = `WORLD ${worldConfig.id} · 1 unit = 1 mm`;
        } catch (error) {
            console.error('World v1 could not be loaded:', error);
        }
    }

    function _buildConfiguredWorld(config) {
        if (worldGroup) scene.remove(worldGroup);
        worldGroup = new THREE.Group();
        worldGroup.name = config.id;
        scene.add(worldGroup);

        const { width_mm: width, depth_mm: depth, grid_step_mm: step } = config.arena;
        worldGround.geometry.dispose();
        worldGround.geometry = new THREE.PlaneGeometry(width, depth);
        worldGround.material.map.repeat.set(width / 10, depth / 10);

        const grid = _makeRectangularGrid(width, depth, step);
        grid.position.y = 0.025;
        worldGroup.add(grid);

        (config.scenery || []).forEach(item => {
            const object = _createScenery(item);
            if (!object) return;
            object.name = item.id;
            object.position.fromArray(item.position_mm);
            object.userData.visualOnly = true;
            worldGroup.add(object);
        });

        config.entities.forEach(entity => {
            const mesh = _createWorldEntity(entity);
            if (!mesh) return;
            mesh.name = entity.id;
            mesh.position.fromArray(entity.position_mm);
            mesh.castShadow = entity.type !== 'dust_patch';
            mesh.receiveShadow = true;
            worldGroup.add(mesh);
            _addSensoryZones(entity);
        });
        _addMillimetreRuler(width, depth);
        toggleWorldDebug(worldDebugVisible);
    }

    function _makeRectangularGrid(width, depth, step) {
        const points = [];
        for (let x = -width / 2; x <= width / 2 + 0.001; x += step) {
            points.push(new THREE.Vector3(x, 0, -depth / 2), new THREE.Vector3(x, 0, depth / 2));
        }
        for (let z = -depth / 2; z <= depth / 2 + 0.001; z += step) {
            points.push(new THREE.Vector3(-width / 2, 0, z), new THREE.Vector3(width / 2, 0, z));
        }
        return new THREE.LineSegments(
            new THREE.BufferGeometry().setFromPoints(points),
            new THREE.LineBasicMaterial({ color: 0x9f936f, transparent: true, opacity: 0.11 })
        );
    }

    function _createScenery(item) {
        const color = new THREE.Color(item.color || '#57734b');
        const material = new THREE.MeshStandardMaterial({ color, roughness: 1, flatShading: true });
        if (item.type === 'hill' || item.type === 'bush') {
            const detail = item.type === 'hill' ? 2 : 1;
            const mound = new THREE.Mesh(new THREE.DodecahedronGeometry(1, detail), material);
            mound.scale.fromArray(item.size_mm);
            mound.castShadow = false;
            mound.receiveShadow = true;
            return mound;
        }
        if (item.type === 'tree') {
            const group = new THREE.Group();
            const height = item.height_mm;
            const crownRadius = item.crown_radius_mm;
            const trunk = new THREE.Mesh(
                new THREE.CylinderGeometry(height * 0.025, height * 0.04, height * 0.58, 7),
                new THREE.MeshStandardMaterial({ color: 0x5d3f2b, roughness: 1, flatShading: true })
            );
            trunk.position.y = height * 0.29;
            const crown = new THREE.Mesh(new THREE.DodecahedronGeometry(1, 1), material);
            crown.scale.set(crownRadius, height * 0.31, crownRadius * 0.88);
            crown.position.y = height * 0.69;
            trunk.castShadow = crown.castShadow = true;
            trunk.receiveShadow = crown.receiveShadow = true;
            group.add(trunk, crown);
            return group;
        }
        return null;
    }

    function _createWorldEntity(entity) {
        const g = entity.geometry || {};
        const materials = {
            sugar_droplet: new THREE.MeshStandardMaterial({ color: 0x8bd8ff, transparent: true, opacity: 0.72, roughness: 0.15 }),
            fruit_chunk: new THREE.MeshStandardMaterial({ color: 0xb66a32, roughness: 0.82 }),
            mold_patch: new THREE.MeshStandardMaterial({ color: 0x315b36, roughness: 1 }),
            bitter_leaf: new THREE.MeshStandardMaterial({ color: 0x315f28, roughness: 0.88 }),
            dust_patch: new THREE.MeshStandardMaterial({ color: 0xcbbd91, transparent: true, opacity: 0.16, roughness: 1 }),
            pebble: new THREE.MeshStandardMaterial({ color: 0x77776e, roughness: 0.92 }),
            grass_blade: new THREE.MeshStandardMaterial({ color: 0x4f7d35, roughness: 0.9 }),
        };
        let geometry;
        if (entity.type === 'sugar_droplet') {
            geometry = new THREE.CylinderGeometry(g.radius_mm, g.radius_mm, g.height_mm, 24);
        } else if (entity.type === 'fruit_chunk') {
            geometry = new THREE.SphereGeometry(0.5, 14, 10);
        } else if (entity.type === 'mold_patch') {
            geometry = new THREE.CircleGeometry(g.radius_mm, 18);
        } else if (entity.type === 'bitter_leaf') {
            geometry = new THREE.BoxGeometry(...g.size_mm);
        } else if (entity.type === 'dust_patch') {
            geometry = new THREE.SphereGeometry(g.radius_mm, 12, 8);
        } else if (entity.type === 'pebble') {
            geometry = new THREE.DodecahedronGeometry(g.radius_mm, 1);
        } else if (entity.type === 'grass_blade') {
            geometry = new THREE.CylinderGeometry(g.radius_mm * 0.35, g.radius_mm, g.height_mm, 7);
        } else {
            return null;
        }
        const mesh = new THREE.Mesh(geometry, materials[entity.type]);
        if (entity.type === 'fruit_chunk') mesh.scale.fromArray(g.size_mm);
        if (entity.type === 'mold_patch') mesh.rotation.x = -Math.PI / 2;
        return mesh;
    }

    function _addSensoryZones(entity) {
        const colors = {
            taste_sugar: 0xffa534, taste_bitter: 0xd44a4a, odor_food: 0x71d47d,
            odor_geosmin: 0x5fa66a, antennal_touch: 0x49c9e8,
        };
        for (const [channel, params] of Object.entries(entity.sensory || {})) {
            const radius = params.range_mm || params.contact_radius_mm;
            if (!radius) continue;
            const zone = new THREE.Mesh(
                new THREE.RingGeometry(Math.max(0.05, radius - 0.12), radius, 48),
                new THREE.MeshBasicMaterial({ color: colors[channel] || 0xffffff, transparent: true, opacity: 0.42, side: THREE.DoubleSide })
            );
            zone.rotation.x = -Math.PI / 2;
            zone.position.set(entity.position_mm[0], 0.06, entity.position_mm[2]);
            zone.userData.sensorZone = true;
            zone.userData.channel = channel;
            worldGroup.add(zone);
        }
    }

    function _addMillimetreRuler(width, depth) {
        const x = -width / 2 + 5;
        const z = depth / 2 - 5;
        const points = [new THREE.Vector3(x, 0.08, z), new THREE.Vector3(x + 10, 0.08, z)];
        const ruler = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(points),
            new THREE.LineBasicMaterial({ color: 0xffffff })
        );
        ruler.userData.worldRuler = true;
        worldGroup.add(ruler);
        for (const dx of [0, 10]) {
            const tick = new THREE.Line(
                new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(x + dx, 0.08, z - 0.6),
                    new THREE.Vector3(x + dx, 0.08, z + 0.6),
                ]),
                new THREE.LineBasicMaterial({ color: 0xffffff })
            );
            tick.userData.worldRuler = true;
            worldGroup.add(tick);
        }
    }

    function toggleWorldDebug(force) {
        worldDebugVisible = typeof force === 'boolean' ? force : !worldDebugVisible;
        if (worldGroup) {
            worldGroup.traverse(node => {
                if (node.userData.sensorZone) node.visible = worldDebugVisible;
            });
        }
        const button = document.getElementById('world-debug-toggle');
        if (button) button.textContent = worldDebugVisible ? 'Hide sensor ranges' : 'Show sensor ranges';
        return worldDebugVisible;
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
            _attachProboscisOverlay();

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

    function loadProboscisOverlay() {
        const loader = new THREE.GLTFLoader();
        loader.load('/static/assets/proboscis_flybody.glb', (gltf) => {
            proboscisModel = gltf.scene;
            // The overlay lives directly in NeuromechFly coordinates. Only the
            // two missing labrum tips are shown; base mouth meshes stay native.
            proboscisModel.position.set(0, 0, 0);
            proboscisModel.scale.setScalar(1.0);
            proboscisModel.visible = activeRenderMode === 'proboscis_overlay';
            proboscisMeshMap = {};

            proboscisModel.traverse((node) => {
                if (node.name) proboscisMeshMap[node.name] = node;
                if (node.isMesh) {
                    node.castShadow = true;
                    node.visible = LABRUM_NAMES.includes(node.name);
                    node.scale.setScalar(LABRUM_GEOMETRY_SCALE);
                }
            });

            _attachProboscisOverlay();
            _matchLabrumMaterial();
        });
    }

    function _attachProboscisOverlay() {
        if (!proboscisModel || !flyModel || currentModel !== 'neuromechfly') return;
        if (proboscisModel.parent !== flyModel) flyModel.add(proboscisModel);
        _matchLabrumMaterial();
    }

    function _matchLabrumMaterial() {
        const source = geomMeshMap.Haustellum || geomMeshMap.Rostrum;
        if (!source || !source.material || !proboscisModel) return;
        for (const name of LABRUM_NAMES) {
            const node = proboscisMeshMap[name];
            if (!node) continue;
            node.traverse((child) => {
                if (!child.isMesh) return;
                child.material = source.material.clone();
                child.material.roughness = 0.7;
                child.material.metalness = 0.02;
            });
        }
    }

    function _setProboscisOverlayVisible(visible) {
        _attachProboscisOverlay();
        if (proboscisModel) proboscisModel.visible = visible;
    }

    function _readFeedTransform(pose, name) {
        const index = feedGeomIndices[name];
        if (index === undefined) return null;
        const off = index * 7;
        return {
            position: new THREE.Vector3(pose[off], pose[off + 1], pose[off + 2]),
            quaternion: new THREE.Quaternion(
                pose[off + 3], pose[off + 4], pose[off + 5], pose[off + 6]),
        };
    }

    function _captureMainMouthReference() {
        mainMouthReference = {};
        for (const mainName of Object.values(MAIN_MOUTH_MAP)) {
            const mesh = geomMeshMap[mainName];
            if (!mesh) continue;
            mainMouthReference[mainName] = {
                position: mesh.position.clone(),
                quaternion: mesh.quaternion.clone(),
            };
        }
    }

    function _restoreMainMouth() {
        if (!mainMouthReference) return;
        for (const [name, transform] of Object.entries(mainMouthReference)) {
            const mesh = geomMeshMap[name];
            if (!mesh) continue;
            mesh.position.copy(transform.position);
            mesh.quaternion.copy(transform.quaternion);
        }
    }

    function _applyMainMouthDelta(pose, feedName, mainName) {
        const mesh = geomMeshMap[mainName];
        const reference = feedMouthReference && feedMouthReference[feedName];
        const mainReference = mainMouthReference && mainMouthReference[mainName];
        const current = _readFeedTransform(pose, feedName);
        if (!mesh || !reference || !mainReference || !current) return;

        const translation = current.position.clone()
            .sub(reference.position)
            .multiplyScalar(FEED_TRANSLATION_SCALE);
        mesh.position.copy(mainReference.position).add(translation);

        const delta = current.quaternion.clone()
            .multiply(reference.quaternion.clone().invert());
        const restrainedDelta = new THREE.Quaternion().identity()
            .slerp(delta, FEED_ROTATION_SCALE);
        mesh.quaternion.copy(restrainedDelta.multiply(mainReference.quaternion));
    }

    function _applyLabrum(pose, name) {
        const labrum = proboscisMeshMap[name];
        const labrumFeed = _readFeedTransform(pose, name);
        const haustellumFeed = _readFeedTransform(pose, 'haustellum');
        const mainHaustellum = geomMeshMap.Haustellum;
        if (!labrum || !labrumFeed || !haustellumFeed || !mainHaustellum) return;

        // Express the labrum relative to flybody's haustellum, then attach that
        // local offset to the animated native NeuromechFly haustellum.
        const invHaustellum = haustellumFeed.quaternion.clone().invert();
        const localOffset = labrumFeed.position.clone()
            .sub(haustellumFeed.position)
            .applyQuaternion(invHaustellum)
            .multiplyScalar(FEED_TRANSLATION_SCALE);
        labrum.position.copy(localOffset)
            .applyQuaternion(mainHaustellum.quaternion)
            .add(mainHaustellum.position);

        const relativeRotation = invHaustellum.multiply(labrumFeed.quaternion);
        labrum.quaternion.copy(mainHaustellum.quaternion)
            .multiply(relativeRotation);
    }

    function _applyProboscisPose(pose) {
        if (!proboscisModel || !pose) return;

        if (!feedMouthReference) {
            feedMouthReference = {};
            for (const feedName of Object.keys(MAIN_MOUTH_MAP)) {
                feedMouthReference[feedName] = _readFeedTransform(pose, feedName);
            }
        }

        for (const [feedName, mainName] of Object.entries(MAIN_MOUTH_MAP)) {
            _applyMainMouthDelta(pose, feedName, mainName);
        }
        for (const name of LABRUM_NAMES) {
            _applyLabrum(pose, name);
        }
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

        const names = data ? data.geom_names || [] : [];
        const previousRenderMode = activeRenderMode;
        activeRenderMode = data && data.render_mode || 'standard';
        const persistentTransition = PERSISTENT_RENDER_MODES.includes(previousRenderMode)
            && PERSISTENT_RENDER_MODES.includes(activeRenderMode);

        // A completed landing keeps its final world quaternion on walk_end.
        // Reset it only when a subsequent non-flight behavior actually starts.
        if (preserveLandedOrientation && !FLIGHT_RENDER_MODES.includes(activeRenderMode)
                && !persistentTransition) {
            flyGroup.rotation.order = 'XYZ';
            flyGroup.rotation.set(0, 0, 0);
            preserveLandedOrientation = false;
        }
        if (FLIGHT_RENDER_MODES.includes(previousRenderMode)
                && !FLIGHT_RENDER_MODES.includes(activeRenderMode)
                && !preserveLandedOrientation && !persistentTransition) {
            flyGroup.rotation.order = 'XYZ';
            flyGroup.rotation.set(0, 0, 0);
        }
        if (WALKING_DATA_RENDER_MODES.includes(previousRenderMode)
                && !WALKING_DATA_RENDER_MODES.includes(activeRenderMode)
                && !persistentTransition) {
            flyGroup.rotation.order = 'XYZ';
            flyGroup.rotation.set(0, 0, 0);
        }

        if (FLIGHT_RENDER_MODES.includes(activeRenderMode)) {
            lastFlightState = null;
            preserveLandedOrientation = false;
            flyGlow.color.copy(BEHAVIOR_COLORS.flight);
            trailLine.material.color.copy(BEHAVIOR_COLORS.flight);
        }

        if (activeRenderMode === 'proboscis_overlay') {
            feedGeomIndices = {};
            names.forEach((name, index) => { feedGeomIndices[name] = index; });
            feedMouthReference = null;
            _captureMainMouthReference();
            _setProboscisOverlayVisible(true);
            return;
        }

        if (previousRenderMode === 'proboscis_overlay') {
            _restoreMainMouth();
            feedMouthReference = null;
            mainMouthReference = null;
        }
        _setProboscisOverlayVisible(false);

        // Check if geom names suggest flybody model (has 'thorax' not 'Thorax')
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
            _attachProboscisOverlay();
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
        if (data.flight_state) lastFlightState = data.flight_state;
        if (!flyGroup || !geomNames.length) return;

        const behaviorColor = BEHAVIOR_COLORS[data.behavior_mode]
            || (['TAKEOFF', 'FLYING', 'LANDING'].includes(data.flight_state)
                ? BEHAVIOR_COLORS.flight : BEHAVIOR_COLORS.walking);
        flyGlow.color.copy(behaviorColor);
        trailLine.material.color.copy(behaviorColor);

        // Feed frames contain flybody transforms, but only their mouth geoms
        // are applied. All other behaviors continue to animate the base model.
        if (activeRenderMode === 'proboscis_overlay') {
            _applyProboscisPose(data.poses);
        } else if (data.poses) {
            applyPose(data.poses);
        }

        // Fly global position → scene units
        const fp = data.fly_pos;
        const s = 1.0;  // MuJoCo cache positions and World v1 both use millimetres.
        const tx = fp[0] * s;
        const airborne = FLIGHT_RENDER_MODES.includes(activeRenderMode)
            || ['TAKEOFF', 'FLYING', 'LANDING'].includes(data.flight_state);
        const ty = airborne ? fp[2] * s : 0;
        const tz = -fp[1] * s;
        if (DIRECT_POSITION_RENDER_MODES.includes(activeRenderMode)) {
            // Preview frames are already resampled at 120 Hz. Applying their
            // positions directly preserves the measured short saccade path.
            flyGroup.position.set(tx, ty, tz);
        } else {
            flyGroup.position.x += (tx - flyGroup.position.x) * 0.3;
            flyGroup.position.y += (ty - flyGroup.position.y) * 0.3;
            flyGroup.position.z += (tz - flyGroup.position.z) * 0.3;
        }
        const posEl = document.getElementById('pos-display');
        if (posEl) {
            posEl.textContent = `pos = [${fp[0].toFixed(1)}, ${fp[1].toFixed(1)}, ${fp[2].toFixed(1)}] mm`;
        }

        if ((['flight_data_preview', 'flight_explore', 'digital_life'].includes(activeRenderMode))
                && data.body_quat) {
            // Cache quaternion is already converted from FlyBody Z-up into
            // the Three.js Y-up world and expressed relative to frame zero.
            flyGroup.quaternion.set(...data.body_quat);
        } else if (activeRenderMode === 'flight_path') {
            // NeuromechFly faces local +X. Yaw follows the circle tangent and
            // local-X roll gives a readable inward bank through the turn.
            flyGroup.rotation.order = 'YXZ';
            flyGroup.rotation.set(
                data.bank_rad || 0,
                data.heading_rad || 0,
                0,
            );
        } else if ((WALKING_DATA_RENDER_MODES.includes(activeRenderMode)
                || PERSISTENT_RENDER_MODES.includes(activeRenderMode))
                && Number.isFinite(data.body_heading_rad)) {
            flyGroup.rotation.order = 'YXZ';
            flyGroup.rotation.set(0, data.body_heading_rad, 0);
        }

        _updateTrail();
        controls.target.lerp(flyGroup.position.clone(), 0.03);
    }

    function walkEnd(data) {
        if (activeRenderMode === 'proboscis_overlay') {
            _restoreMainMouth();
            _setProboscisOverlayVisible(false);
        }
        feedMouthReference = null;
        mainMouthReference = null;
        const landedFlight = FLIGHT_RENDER_MODES.includes(activeRenderMode)
            && lastFlightState === 'GROUNDED';
        const preserveRequested = Boolean(data && data.preserve_world_orientation);
        preserveLandedOrientation = landedFlight || preserveRequested;
        if (!preserveLandedOrientation) flyGroup.rotation.set(0, 0, 0);
        activeRenderMode = 'standard';
    }

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
        const s = 1.0;
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

    return { init, updateFly, resetTrail, walkInit, walkUpdate, walkEnd, loadWalkCache, brainDrive, toggleWorldDebug };
})();
