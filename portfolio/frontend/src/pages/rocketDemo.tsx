import { useEffect, useRef, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { TrajectoryPlayer, SimData } from '../rocket/trajectory_player';
import { LivePlots } from '../rocket/live_plots';
import {
  BOOSTER,
  STAGE0,
  ENGINE_TRANSFORMS,
  ENGINE_GIMBAL_OFFSET_Z,
} from '../rocket/assembly.config';

const BASE = import.meta.env.BASE_URL;

const API_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? 'https://rocket-sim-api.onrender.com';

export default function RocketDemo() {
  const containerRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  const trajectoryPlayerRef  = useRef<TrajectoryPlayer | null>(null);
  const resetCatchRef        = useRef<(() => void) | null>(null);
  const runDemoRef           = useRef<(() => void) | null>(null);
  const catchEnabledRef         = useRef(false);
  const setTowerVisibilityRef   = useRef<((v: boolean) => void) | null>(null);
  const fadeTowerOutRef         = useRef<(() => void) | null>(null);
  const isFirstSetpointRef      = useRef(true);
  const pendingSetpointRef      = useRef<[number, number, number] | null>(null);
  const lastSetpointClickRef    = useRef(0);
  const lastLiveSetpointRef     = useRef<[number, number, number]>([0, 0, 48]);
  const savedFrameRef           = useRef<import('../rocket/trajectory_player').SimFrame | null>(null);
  const livePlotsRef            = useRef<LivePlots | null>(null);
  const [simStatus, setSimStatus] = useState<'idle' | 'loading' | 'playing' | 'error'>('idle');
  const [simError,  setSimError]  = useState('');
  const [spX, setSpX] = useState('0');
  const [spY, setSpY] = useState('0');
  const [spZ, setSpZ] = useState('48');

  const runDemo = useCallback((): void => {
    const tp    = trajectoryPlayerRef.current;
    const reset = resetCatchRef.current;
    if (!tp || !reset) return;
    setSimError('');
    setSimStatus('loading');
    // Disconnect any live sim before playing the canned demo
    tp.disconnect();
    catchEnabledRef.current = true;
    setTowerVisibilityRef.current?.(true);
    fetch(`${BASE}landing.json`)
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() as Promise<SimData>; })
      .then((data) => { reset(); tp.reset(); tp.playFromData(data); setSimStatus('playing'); })
      .catch((e) => { setSimStatus('error'); setSimError(String(e)); });
  }, []);

  const isDefaultSetpoint = (x: number, y: number, z: number) =>
    Math.abs(x) < 0.01 && Math.abs(y) < 0.01 && Math.abs(z - 48) < 0.01;

  const runSetpoint = useCallback(() => {
    const tp    = trajectoryPlayerRef.current;
    const reset = resetCatchRef.current;
    if (!tp || !reset) return;
    const x = parseFloat(spX) || 0;
    const y = parseFloat(spY) || 0;
    const z = parseFloat(spZ) || 50;

    lastSetpointClickRef.current = Date.now();
    catchEnabledRef.current = false;

    // If already in a live WS session, just hot-swap the setpoint
    if (tp.isConnected) {
      reset();
      tp.sendSetpoint(x, y, z);
      lastLiveSetpointRef.current = [x, y, z];
      return;
    }

    setSimError('');
    setSimStatus('loading');

    const saved = savedFrameRef.current;
    savedFrameRef.current = null;

    if (saved) {
      // Resume: booster stays where it is — no fade, no position reset
      reset();
      tp.resetKeepPosition();
      const wsBase = API_URL.replace(/^https/, 'wss').replace(/^http/, 'ws');
      lastLiveSetpointRef.current = [x, y, z];
      const [qw, qx, qy, qz] = saved.quat;
      const [px, py, pz] = saved.pos;
      const params = `x=${x}&y=${y}&z=${z}&ix=${px}&iy=${py}&iz=${pz}&iqw=${qw}&iqx=${qx}&iqy=${qy}&iqz=${qz}`;
      tp.connect(`${wsBase}/ws/simulate?${params}`);
      setSimStatus('playing');
      return;
    }

    fadeTowerOutRef.current?.();
    reset();
    tp.reset();

    // Default setpoint → play from pre-baked static JSON, no backend needed
    if (isDefaultSetpoint(x, y, z)) {
      fetch(`${BASE}landing.json`)
        .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() as Promise<SimData>; })
        .then((data) => { tp.playFromData(data); setSimStatus('playing'); })
        .catch((e) => { setSimStatus('error'); setSimError(String(e)); });
      return;
    }

    // Custom setpoint → live WS
    const wsBase = API_URL.replace(/^https/, 'wss').replace(/^http/, 'ws');
    lastLiveSetpointRef.current = [x, y, z];
    if (isFirstSetpointRef.current) {
      isFirstSetpointRef.current = false;
      pendingSetpointRef.current = [x, y, z];
      tp.connect(`${wsBase}/ws/simulate?x=0&y=0&z=20`);
    } else {
      tp.connect(`${wsBase}/ws/simulate?x=${x}&y=${y}&z=${z}`);
    }
    setSimStatus('playing');
  }, [spX, spY, spZ]);

  useEffect(() => { runDemoRef.current = runDemo; }, [runDemo]);

  // Pad-clear handoff + 1-min idle disconnect
  useEffect(() => {
    const checkId = setInterval(() => {
      const tp = trajectoryPlayerRef.current;

      // Once booster clears the pad (z ≥ 15 m), send the user's actual setpoint
      if (pendingSetpointRef.current && tp) {
        const pos = tp.latestPos;
        if (pos && pos[2] >= 15) {
          const [px, py, pz] = pendingSetpointRef.current;
          pendingSetpointRef.current = null;
          tp.sendSetpoint(px, py, pz);
          lastLiveSetpointRef.current = [px, py, pz];
        }
      }

      // 1 min since last "Go to Setpoint" click → close the connection
      if (
        tp?.isConnected &&
        lastSetpointClickRef.current > 0 &&
        Date.now() - lastSetpointClickRef.current > 60_000
      ) {
        lastSetpointClickRef.current = 0;
        savedFrameRef.current = tp.latestFrame;   // remember where the booster ended up
        tp.disconnect();
        setSimStatus('idle');
      }
    }, 500);

    return () => clearInterval(checkId);
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // ─────────────────────────────────────────────────────────────────────────
    // Everything below is verbatim from rendering/src/main.ts.
    // Only changes:
    //   • document.body.appendChild  →  container.appendChild
    //   • "/models/..."              →  `${BASE}models/...`
    //   • buildControlPanel removed  (portfolio doesn't need sliders)
    //   • keydown listener removed   (no ws reconnect in portfolio)
    //   • trajectory player connects via playFromData(demo.json) instead of WS
    // ─────────────────────────────────────────────────────────────────────────

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d0e10);
    scene.fog = new THREE.FogExp2(0x0d0e10, 0.0008);

    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1, 2000);
    camera.position.set(0, 60, 120);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.9;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    container.appendChild(renderer.domElement);      // ← only change from main.ts

    const pmrem = new THREE.PMREMGenerator(renderer);
    pmrem.compileEquirectangularShader();
    const envTexture = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environment = envTexture;
    pmrem.dispose();

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 2;
    controls.maxDistance = 1000;
    controls.maxPolarAngle = Math.PI;

    function makeGrid(size: number, divisions: number, color: number, opacity: number, y = 0): THREE.GridHelper {
      const grid = new THREE.GridHelper(size, divisions, color, color);
      (grid.material as THREE.LineBasicMaterial).opacity = opacity;
      (grid.material as THREE.LineBasicMaterial).transparent = true;
      (grid.material as THREE.LineBasicMaterial).depthWrite = false;
      grid.position.y = y;
      return grid;
    }

    scene.add(new THREE.AmbientLight(0x8090a0, 0.02));

    const sun = new THREE.DirectionalLight(0xffe8c0, 6.0);
    sun.position.set(300, 220, 120);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.near = 10;
    sun.shadow.camera.far = 100000;
    sun.shadow.camera.left = -50000;
    sun.shadow.camera.right = 50000;
    sun.shadow.camera.top = 50000;
    sun.shadow.camera.bottom = -50000;
    sun.shadow.bias = -0.001;
    scene.add(sun);
    scene.add(sun.target);

    const rim = new THREE.DirectionalLight(0x6090c0, 0.6);
    rim.position.set(-120, 80, -100);
    scene.add(rim);

    const bounce = new THREE.HemisphereLight(0x5080b0, 0x1a1c1e, 0.5);
    scene.add(bounce);

    const loader = new GLTFLoader();

    const CATCH_CLOSE_DIST = 50;
    const CATCH_LAND_DIST  = 2;

    let boosterModel: THREE.Object3D | null = null;

    const gimbalPivots:   (THREE.Group | null)[]  = [null, null, null];
    const gimbalBaseQuat: THREE.Quaternion[]       = [new THREE.Quaternion(), new THREE.Quaternion(), new THREE.Quaternion()];
    let trajectoryPlayer: TrajectoryPlayer | null  = null;
    const livePlots = new LivePlots();
      livePlotsRef.current = livePlots;

    const thrustArrows: THREE.ArrowHelper[] = [];
    const THRUST_REF_N = 1_000_000;
    let simSetpoint: [number, number, number] | null = null;
    let catchPhase1Done = false;
    let catchPhase2Done = false;

    type ModuleAnim = { startTime: number; startVal: number; endVal: number; duration: number; onUpdate: (v: number) => void; onDone?: () => void };
    const moduleAnims: ModuleAnim[] = [];

    function tickModuleAnims() {
      const now = performance.now();
      for (let i = moduleAnims.length - 1; i >= 0; i--) {
        const a = moduleAnims[i];
        const t = Math.min(1, (now - a.startTime) / (a.duration * 1000));
        a.onUpdate(a.startVal + (a.endVal - a.startVal) * t);
        if (t >= 1) { a.onDone?.(); moduleAnims.splice(i, 1); }
      }
    }

    let leftChopstick:  THREE.Object3D | null = null;
    let rightChopstick: THREE.Object3D | null = null;
    let chopConnector:  THREE.Object3D | null = null;
    const leftChopstickBaseQ  = new THREE.Quaternion();
    const rightChopstickBaseQ = new THREE.Quaternion();
    let leftChopAngleDeg   = 0;
    let rightChopAngleDeg  = 0;

    const CHOP_AMP_PER_DEGPS = 0.12;
    const CHOP_AMP_MAX       = 6.0;
    const CHOP_GAMMA         = 0.7;
    const CHOP_OMEGA         = 5.0;
    const CHOP_ROT_SPEED_REF = 10;
    type ChopOsc = { active: boolean; startTime: number; baseAngle: number; amp: number };
    const leftOsc:  ChopOsc = { active: false, startTime: 0, baseAngle: 0, amp: CHOP_AMP_PER_DEGPS * CHOP_ROT_SPEED_REF };
    const rightOsc: ChopOsc = { active: false, startTime: 0, baseAngle: 0, amp: CHOP_AMP_PER_DEGPS * CHOP_ROT_SPEED_REF };

    const _chopQa = new THREE.Quaternion();
    const _chopQb = new THREE.Quaternion();

    function oscOffset(osc: ChopOsc, sign: number): number {
      if (!osc.active) return 0;
      const t = (performance.now() - osc.startTime) / 1000;
      const env = Math.exp(-CHOP_GAMMA * t);
      if (env < 0.005) { osc.active = false; return 0; }
      return sign * osc.amp * env * Math.sin(CHOP_OMEGA * t);
    }

    function tickChopstick(obj: THREE.Object3D | null, baseQ: THREE.Quaternion, sliderDeg: number, oscY: ChopOsc, oscX: ChopOsc, signX: number, minDeg: number, maxDeg: number) {
      if (!obj || !(oscY.active || oscX.active)) return;
      const degY = Math.min(maxDeg, Math.max(minDeg, sliderDeg + oscOffset(oscY, 1)));
      const degX = oscOffset(oscX, signX);
      _chopQa.setFromAxisAngle(CHOP_LOCAL_Y, degY * (Math.PI / 180));
      _chopQb.setFromAxisAngle(CHOP_LOCAL_X, degX * (Math.PI / 180));
      obj.quaternion.copy(baseQ).multiply(_chopQa).multiply(_chopQb);
    }

    const CHOP_LOCAL_Y = new THREE.Vector3(0, 1, 0);
    const CHOP_LOCAL_X = new THREE.Vector3(1, 0, 0);

    const leftOscX:  ChopOsc = { active: false, startTime: 0, baseAngle: 0, amp: 0.8 };
    const rightOscX: ChopOsc = { active: false, startTime: 0, baseAngle: 0, amp: 0.8 };

    const CHOP_TRANS_AMP_PER_MPS = 0.5;
    const CHOP_TRANS_AMP_MAX     = 4.0;
    const CHOP_TRANS_SPEED_REF   = 5;
    const CHOP_TRANS_OMEGA       = 3.5;
    type TransOsc = { active: boolean; startTime: number; basePos: number; amp: number };
    const connTransOsc: TransOsc = { active: false, startTime: 0, basePos: 0, amp: CHOP_TRANS_AMP_PER_MPS * CHOP_TRANS_SPEED_REF };

    function tickTransOscillation(osc: TransOsc, obj: THREE.Object3D | null) {
      if (!osc.active || !obj) return;
      const t = (performance.now() - osc.startTime) / 1000;
      const offset = -osc.amp * Math.exp(-CHOP_GAMMA * t) * Math.sin(CHOP_TRANS_OMEGA * t);
      obj.position.y = osc.basePos + offset;
      if (Math.exp(-CHOP_GAMMA * t) < 0.005) { osc.active = false; obj.position.y = osc.basePos; }
    }

    function triggerLeftWobble(rotSpeed = CHOP_ROT_SPEED_REF) {
      leftOsc.amp = Math.min(CHOP_AMP_MAX, CHOP_AMP_PER_DEGPS * rotSpeed);
      leftOsc.baseAngle = leftChopAngleDeg; leftOsc.startTime = performance.now(); leftOsc.active = true;
    }
    function triggerLeftStopWobble(rotSpeed = CHOP_ROT_SPEED_REF) {
      leftOsc.amp = Math.min(CHOP_AMP_MAX, CHOP_AMP_PER_DEGPS * rotSpeed);
      leftOsc.baseAngle = leftChopAngleDeg; leftOsc.startTime = performance.now(); leftOsc.active = true;
    }
    function triggerRightWobble(rotSpeed = CHOP_ROT_SPEED_REF) {
      rightOsc.amp = Math.min(CHOP_AMP_MAX, CHOP_AMP_PER_DEGPS * rotSpeed);
      rightOsc.baseAngle = rightChopAngleDeg; rightOsc.startTime = performance.now(); rightOsc.active = true;
    }
    function triggerRightStopWobble(rotSpeed = CHOP_ROT_SPEED_REF) {
      rightOsc.amp = Math.min(CHOP_AMP_MAX, CHOP_AMP_PER_DEGPS * rotSpeed);
      rightOsc.baseAngle = rightChopAngleDeg; rightOsc.startTime = performance.now(); rightOsc.active = true;
    }
    function triggerTransWobble(transSpeed = CHOP_TRANS_SPEED_REF) {
      if (!chopConnector) return;
      const now  = performance.now();
      const xAmp = Math.min(CHOP_AMP_MAX, CHOP_AMP_PER_DEGPS * transSpeed * 0.5);
      connTransOsc.amp      = Math.min(CHOP_TRANS_AMP_MAX, CHOP_TRANS_AMP_PER_MPS * transSpeed);
      connTransOsc.basePos  = chopConnector.position.y;
      connTransOsc.startTime = now; connTransOsc.active = true;
      leftOscX.amp  = xAmp; leftOscX.startTime  = now; leftOscX.active  = true;
      rightOscX.amp = xAmp; rightOscX.startTime = now; rightOscX.active = true;
    }

    // ── Stage 0 ───────────────────────────────────────────────────────────────
    loader.load(`${BASE}models/stage0.glb`, (gltf) => {
      const model = gltf.scene;
      model.scale.setScalar(1000);
      model.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(model);
      model.position.y -= box.min.y;
      model.position.x = STAGE0.position.x;
      model.position.z = STAGE0.position.z;
      model.rotation.y = STAGE0.rotationY;
      scene.add(model);

      const chopstickMat  = new THREE.MeshStandardMaterial({ color: 0x1a1c1f, roughness: 0.3, metalness: 0.85, envMapIntensity: 1.8 });
      const concreteMat   = new THREE.MeshStandardMaterial({ color: 0x52575c, roughness: 0.75, metalness: 0.05, envMapIntensity: 0.8 });
      const launchPadMat2 = new THREE.MeshStandardMaterial({ color: 0x52575c, roughness: 0.92, metalness: 0.0, envMapIntensity: 0.5 });

      const towerObjects: THREE.Mesh[] = [];

      model.traverse((child) => {
        const mesh = child as THREE.Mesh;
        if (!mesh.isMesh) return;
        mesh.castShadow = true;
        if (["left_chopstick", "right_chopstick", "QDBoosterClaw2001", "Chopstick_TowerConnector002"].includes(mesh.name)) {
          if (mesh.name === "left_chopstick")  { leftChopstick = mesh;  leftChopstickBaseQ.copy(mesh.quaternion); }
          if (mesh.name === "right_chopstick") { rightChopstick = mesh; rightChopstickBaseQ.copy(mesh.quaternion); }
          mesh.material = chopstickMat;
          towerObjects.push(mesh);
        } else if (["Tower_base", "top-section", "mid-section"].includes(mesh.name)) {
          mesh.material = concreteMat;
          towerObjects.push(mesh);
        } else if (mesh.name === "launch_pad") {
          mesh.material = launchPadMat2;
        }
      });

      // Collect unique materials
      const towerMats = Array.from(new Set(towerObjects.map((m) => m.material as THREE.MeshStandardMaterial)));

      // Immediate show — used when switching back to demo mode
      setTowerVisibilityRef.current = (visible: boolean) => {
        for (const mat of towerMats) {
          mat.transparent = false; mat.depthWrite = true; mat.opacity = 1; mat.needsUpdate = true;
        }
        for (const obj of towerObjects) obj.visible = visible;
      };

      // Smooth fade-out via its own rAF loop — no dependency on moduleAnims
      let fadeRafId = 0;
      fadeTowerOutRef.current = () => {
        cancelAnimationFrame(fadeRafId);
        for (const obj of towerObjects) obj.visible = true;
        for (const mat of towerMats) {
          mat.transparent = true; mat.depthWrite = false; mat.opacity = 1; mat.needsUpdate = true;
        }
        const DURATION = 700;
        const t0 = performance.now();
        const tick = () => {
          const p = Math.min(1, (performance.now() - t0) / DURATION);
          const opacity = 1 - p;
          for (const mat of towerMats) mat.opacity = opacity;
          if (p < 1) {
            fadeRafId = requestAnimationFrame(tick);
          } else {
            for (const obj of towerObjects) obj.visible = false;
            for (const mat of towerMats) {
              mat.transparent = false; mat.depthWrite = true; mat.opacity = 1; mat.needsUpdate = true;
            }
          }
        };
        fadeRafId = requestAnimationFrame(tick);
      };

      const connNode = model.getObjectByName("Chopstick_TowerConnector002");
      if (connNode) { chopConnector = connNode; }
    }, undefined, (err) => console.error("Stage0 load error:", err));

    // ── Booster ───────────────────────────────────────────────────────────────
    loader.load(`${BASE}models/booster_asm.glb`, (gltf) => {
      const model = gltf.scene;
      model.rotation.z = BOOSTER.rotationZ;

      const steel = new THREE.MeshStandardMaterial({
        color: 0x6a7075, roughness: 0.45, metalness: 0.75, envMapIntensity: 1.2,
      });
      const engineBlack = new THREE.MeshStandardMaterial({
        color: 0x080808, roughness: 0.95, metalness: 0.1, envMapIntensity: 0.1,
      });

      model.updateMatrixWorld(true);
      const modelBox     = new THREE.Box3().setFromObject(model);
      const modelMinY    = modelBox.min.y;
      const modelHeight  = modelBox.max.y - modelBox.min.y;
      const engineThreshold = modelMinY + modelHeight * 0.25;

      const meshBox = new THREE.Box3();
      model.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          meshBox.setFromObject(child);
          const centerY = (meshBox.min.y + meshBox.max.y) / 2;
          (child as THREE.Mesh).material = centerY < engineThreshold ? engineBlack : steel;
          (child as THREE.Mesh).castShadow = true;
        }
      });

      scene.add(model);

      const box    = new THREE.Box3().setFromObject(model);
      const size   = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());

      model.position.set(BOOSTER.position.x, BOOSTER.position.y, BOOSTER.position.z);
      model.updateMatrixWorld(true);

      const modelRotInv = new THREE.Quaternion();
      model.getWorldQuaternion(modelRotInv);
      modelRotInv.invert();
      const oldModelWorldPos = new THREE.Vector3(-center.x, -box.min.y, -center.z);

      const toModelLocal = (wx: number, wy: number, wz: number): THREE.Vector3 =>
        new THREE.Vector3(wx - oldModelWorldPos.x, wy - oldModelWorldPos.y, wz - oldModelWorldPos.z)
          .applyQuaternion(modelRotInv);

      const toModelLocalQuat = (rx: number, ry: number, rz: number): THREE.Quaternion =>
        modelRotInv.clone().multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz)));

      const radius = size.length() / 2;
      const fov    = (camera.fov * Math.PI) / 180;
      const dist   = (radius / Math.sin(fov / 2)) * 1.5;
      camera.position.set(dist * 0.6, dist * 0.5, dist * 0.8);
      camera.lookAt(0, size.y / 2, 0);
      controls.target.set(0, size.y / 2, 0);

      camera.near = dist * 0.001;
      camera.far  = dist * 20;
      camera.updateProjectionMatrix();
      scene.fog = new THREE.FogExp2(0x0d0e10, 0.6 / (dist * 8));
      controls.maxDistance = dist * 8;
      controls.update();

      const gridSize = radius * 30;
      const fine     = Math.round(gridSize / radius) * 10;
      scene.add(makeGrid(gridSize, fine, 0x555555, 0.5, 0));
      scene.add(makeGrid(gridSize, Math.round(fine / 10), 0x777777, 0.85, 0));

      const groundDisc = new THREE.Mesh(
        new THREE.PlaneGeometry(radius * 4, radius * 4),
        new THREE.MeshStandardMaterial({ color: 0x1a1b1e, roughness: 0.9, metalness: 0.1 }),
      );
      groundDisc.rotation.x = -Math.PI / 2;
      groundDisc.position.y = -0.5;
      groundDisc.receiveShadow = true;
      scene.add(groundDisc);

      const s = size.y * 1.5;
      sun.shadow.camera.left   = -s;
      sun.shadow.camera.right  = s;
      sun.shadow.camera.top    = s * 2;
      sun.shadow.camera.bottom = -s;
      sun.shadow.camera.far    = size.y * 10;
      sun.target.position.set(model.position.x, size.y / 2, model.position.z);
      sun.target.updateMatrixWorld();
      sun.shadow.camera.updateProjectionMatrix();

      // ── Grid fins ──────────────────────────────────────────────────────────
      const finMat = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 1.0, metalness: 0.0 });
      const FIN_TRANSFORMS: [number, number, number, number, number, number][] = [
        [    0, 65860.3,  4520, -Math.PI / 2, Math.PI,            0],
        [-4520, 65860.0,     0, -Math.PI / 2, Math.PI, Math.PI / 2],
        [ 4520, 65860.0,     0, -Math.PI / 2, Math.PI,-Math.PI / 2],
      ];
      FIN_TRANSFORMS.forEach(([px, py, pz, rx, ry, rz], i) => {
        loader.load(`${BASE}models/grid_fin_ass_asm.glb`, (gltfFin) => {
          const fin = gltfFin.scene;
          fin.traverse((child) => {
            if ((child as THREE.Mesh).isMesh) {
              (child as THREE.Mesh).material = finMat;
              (child as THREE.Mesh).castShadow = true;
            }
          });
          fin.position.copy(toModelLocal(px, py, pz));
          fin.quaternion.copy(toModelLocalQuat(rx, ry, rz));
          model.add(fin);
        }, undefined, (err) => console.error(`Failed to load grid fin ${i + 1}:`, err));
      });

      // ── Engines ────────────────────────────────────────────────────────────
      const engMat = new THREE.MeshStandardMaterial({ color: 0x7a3010, roughness: 0.8, metalness: 0.2, envMapIntensity: 0.6 });
      ENGINE_TRANSFORMS.forEach(([px, py, pz, rx, ry, rz], i) => {
        loader.load(`${BASE}models/engine.glb`, (gltfEng) => {
          const eng = gltfEng.scene;
          eng.traverse((child) => {
            if ((child as THREE.Mesh).isMesh) {
              (child as THREE.Mesh).material = engMat;
              (child as THREE.Mesh).castShadow = true;
              (child as THREE.Mesh).receiveShadow = true;
            }
          });
          eng.position.z = ENGINE_GIMBAL_OFFSET_Z;

          const pivot    = new THREE.Group();
          const thrustLen = size.z * 4.0;
          if (trajectoryPlayer) trajectoryPlayer.thrustArrowBaseLen = thrustLen;
          const arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), thrustLen, 0xff0000, thrustLen * 0.08, thrustLen * 0.04);
          thrustArrows.push(arrow);
          pivot.add(arrow);
          pivot.add(eng);

          pivot.position.copy(toModelLocal(px, py, pz));
          pivot.quaternion.copy(toModelLocalQuat(rx, ry, rz));
          model.add(pivot);

          gimbalPivots[i]   = pivot;
          gimbalBaseQuat[i].copy(pivot.quaternion);
        }, undefined, (err) => console.error(`Failed to load engine ${i + 1}:`, err));
      });

      boosterModel = model;

      // ── Trajectory player ──────────────────────────────────────────────────
      const _chopQtmp  = new THREE.Quaternion();
      const _chopLocalY = new THREE.Vector3(0, 1, 0);

      function openChopsticks() {
        leftChopAngleDeg  = 30;
        rightChopAngleDeg = -30;
        _chopQtmp.setFromAxisAngle(_chopLocalY,  30 * (Math.PI / 180));
        leftChopstick?.quaternion.copy(leftChopstickBaseQ).multiply(_chopQtmp);
        _chopQtmp.setFromAxisAngle(_chopLocalY, -30 * (Math.PI / 180));
        rightChopstick?.quaternion.copy(rightChopstickBaseQ).multiply(_chopQtmp);
      }

      function resetCatchSequence() {
        catchPhase1Done = false;
        catchPhase2Done = false;
        moduleAnims.length = 0;
        if (boosterModel && boosterModel.parent !== scene) scene.attach(boosterModel);
        if (trajectoryPlayer) trajectoryPlayer.frozen = false;
        for (const arrow of thrustArrows) arrow.visible = true;
        openChopsticks();
      }

      trajectoryPlayer = new TrajectoryPlayer(model, gimbalPivots, gimbalBaseQuat, scene);
      trajectoryPlayer.onMeta  = (total, sp) => { livePlots.setMeta(total, sp); simSetpoint = sp; };
      trajectoryPlayer.onFrame = (t, pos, eng, omega, u_cart) => livePlots.addFrame(t, pos, eng, omega, u_cart);
      trajectoryPlayer.thrustArrows = thrustArrows;
      trajectoryPlayer.thrustRefN   = THRUST_REF_N;
      trajectoryPlayerRef.current = trajectoryPlayer;
      resetCatchRef.current       = resetCatchSequence;

      // ── Catch sequence checker @ 5 Hz ──────────────────────────────────────
      const catchIntervalId = setInterval(() => {
        if (!catchEnabledRef.current) return;
        if (!trajectoryPlayer || !simSetpoint) return;
        const pos = trajectoryPlayer.latestPos;
        if (!pos) return;

        const dx = pos[0] - simSetpoint[0];
        const dy = pos[1] - simSetpoint[1];
        const dz = pos[2] - simSetpoint[2];
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (!catchPhase1Done && dist < CATCH_CLOSE_DIST) {
          catchPhase1Done = true;
          const now      = performance.now();
          const duration = 6.0;
          const startLeft  = leftChopAngleDeg;
          const startRight = rightChopAngleDeg;
          const localChopYv = new THREE.Vector3(0, 1, 0);
          const qTmp        = new THREE.Quaternion();
          triggerLeftWobble();
          triggerRightWobble();
          moduleAnims.push({
            startTime: now, startVal: startLeft, endVal: -4, duration,
            onUpdate: (v) => {
              leftChopAngleDeg = v;
              if (leftChopstick && !leftOsc.active && !leftOscX.active) {
                qTmp.setFromAxisAngle(localChopYv, v * (Math.PI / 180));
                leftChopstick.quaternion.copy(leftChopstickBaseQ).multiply(qTmp);
              }
            },
            onDone: () => triggerLeftStopWobble(),
          });
          moduleAnims.push({
            startTime: now, startVal: startRight, endVal: 7, duration,
            onUpdate: (v) => {
              rightChopAngleDeg = v;
              if (rightChopstick && !rightOsc.active && !rightOscX.active) {
                qTmp.setFromAxisAngle(localChopYv, v * (Math.PI / 180));
                rightChopstick.quaternion.copy(rightChopstickBaseQ).multiply(qTmp);
              }
            },
            onDone: () => triggerRightStopWobble(),
          });
        }

        if (!catchPhase2Done && Math.abs(dz) < CATCH_LAND_DIST) {
          catchPhase2Done = true;
          for (const arrow of thrustArrows) arrow.visible = false;
          if (trajectoryPlayer) trajectoryPlayer.frozen = true;
          if (boosterModel && chopConnector) chopConnector.attach(boosterModel);
          triggerTransWobble();
        }
      }, 200);

      // Store IDs for cleanup
      (renderer as any).__catchIntervalId    = catchIntervalId;
      (renderer as any).__trajectoryPlayer   = trajectoryPlayer;
      (renderer as any).__livePlots          = livePlots;
      (renderer as any).__resetCatchSequence = resetCatchSequence;

      openChopsticks();

    }, undefined, (err) => console.error("Failed to load booster:", err));

    // ── FPS counter ───────────────────────────────────────────────────────────
    const fpsEl = document.createElement("div");
    fpsEl.style.cssText = `position:fixed;bottom:16px;left:16px;z-index:1000;font:13px/1 monospace;color:#8ab4d4;background:rgba(10,12,16,0.75);border:1px solid rgba(255,255,255,0.08);border-radius:6px;padding:5px 10px;pointer-events:none;`;
    document.body.appendChild(fpsEl);
    let fpsFrames = 0, fpsLast = performance.now();

    // ── Animate ───────────────────────────────────────────────────────────────
    let rafId = 0;
    function animate() {
      rafId = requestAnimationFrame(animate);
      controls.update();
      tickModuleAnims();
      trajectoryPlayer?.tick();
      livePlots.render();
      tickChopstick(leftChopstick,  leftChopstickBaseQ,  leftChopAngleDeg,  leftOsc,  leftOscX,   1, -8,  60);
      tickChopstick(rightChopstick, rightChopstickBaseQ, rightChopAngleDeg, rightOsc, rightOscX, -1, -60,  8);
      tickTransOscillation(connTransOsc, chopConnector);
      renderer.render(scene, camera);
      fpsFrames++;
      const now = performance.now();
      if (now - fpsLast >= 500) {
        fpsEl.textContent = `${Math.round(fpsFrames * 1000 / (now - fpsLast))} FPS`;
        fpsFrames = 0; fpsLast = now;
      }
    }
    animate();

    // ── Resize ────────────────────────────────────────────────────────────────
    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener("resize", onResize);

    // ── Space → replay demo ───────────────────────────────────────────────────
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !e.repeat) {
        e.preventDefault();
        runDemoRef.current?.();
      }
    };
    window.addEventListener("keydown", onKeyDown);

    // ── Cleanup ───────────────────────────────────────────────────────────────
    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("resize", onResize);
      window.removeEventListener("keydown", onKeyDown);
      clearInterval((renderer as any).__catchIntervalId);
      ((renderer as any).__trajectoryPlayer as TrajectoryPlayer | undefined)?.dispose();
      livePlotsRef.current?.dispose();
      livePlotsRef.current = null;
      if (fpsEl.parentNode) fpsEl.parentNode.removeChild(fpsEl);
      renderer.dispose();
      envTexture.dispose();
      if (renderer.domElement.parentNode) renderer.domElement.parentNode.removeChild(renderer.domElement);
    };
  }, []);

  const isLoading = simStatus === 'loading';

  const inputStyle: React.CSSProperties = {
    width: '100%', minWidth: 0, background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.12)',
    borderRadius: '4px', color: '#cdd6e0', fontFamily: 'monospace', fontSize: '12px',
    padding: '4px 4px', textAlign: 'center', boxSizing: 'border-box',
  };

  const btnBase: React.CSSProperties = {
    width: '100%', padding: '7px 0', borderRadius: '5px', border: 'none', cursor: 'pointer',
    fontFamily: 'monospace', fontSize: '11px', letterSpacing: '0.08em', textTransform: 'uppercase',
    transition: 'opacity 0.2s',
  };

  // ── Overlay ─────────────────────────────────────────────────────────────────
  return (
    <div style={{ position: 'fixed', inset: 0, background: '#0d0e10' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />

      {/* Top bar */}
      <div style={{ position: 'absolute', top: 0, left: 0, right: 0, padding: '24px 32px', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', pointerEvents: 'none' }}>
        <button
          onClick={() => navigate('/downloadsPage')}
          style={{ pointerEvents: 'auto', background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
          className="text-xs font-light tracking-widest text-white/60 hover:text-[#9F8E6D] transition-colors duration-300 uppercase"
        >
          ← Back
        </button>
        <div style={{ textAlign: 'center', position: 'absolute', left: '50%', transform: 'translateX(-50%)' }}>
          <div className="text-sm font-light tracking-widest text-white/80 uppercase">Starship Booster Landing</div>
          <div className="text-xs font-light tracking-widest text-white/40 uppercase mt-1">LQR Full-State Feedback Control</div>
        </div>
      </div>

      {/* Control panel — bottom right */}
      <div style={{
        position: 'absolute', bottom: '16px', right: '16px', zIndex: 1000,
        background: 'rgba(10,12,16,0.88)', border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: '10px', padding: '14px 16px', width: '220px', boxSizing: 'border-box',
        fontFamily: 'monospace', color: '#cdd6e0',
      }}>
        <div style={{ fontSize: '10px', letterSpacing: '0.1em', color: '#8ab4d4', textTransform: 'uppercase', marginBottom: '12px' }}>
          Simulation Control
        </div>

        {/* Option 1 — Demo */}
        <button
          onClick={runDemo}
          disabled={isLoading}
          style={{ ...btnBase, background: 'rgba(138,180,212,0.12)', color: '#8ab4d4', opacity: isLoading ? 0.5 : 1 }}
        >
          ▶ Simulate Landing
        </button>
        <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.25)', textAlign: 'center', marginTop: '4px', marginBottom: '14px' }}>
          or press Space
        </div>

        {/* Divider */}
        <div style={{ borderTop: '1px solid rgba(255,255,255,0.07)', marginBottom: '12px' }} />

        {/* Option 2 — Custom setpoint */}
        <div style={{ fontSize: '10px', letterSpacing: '0.08em', color: 'rgba(255,255,255,0.4)', textTransform: 'uppercase', marginBottom: '8px' }}>
          Custom Setpoint (m)
        </div>
        <div style={{ display: 'flex', gap: '6px', marginBottom: '8px', width: '100%', boxSizing: 'border-box' }}>
          {([['X', spX, setSpX], ['Y', spY, setSpY], ['Z', spZ, setSpZ]] as [string, string, (v: string) => void][]).map(([label, val, setter]) => (
            <label key={label} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '3px', flex: '1 1 0', minWidth: 0 }}>
              <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.35)' }}>{label}</span>
              <input
                type="number"
                value={val}
                onChange={(e) => setter(e.target.value)}
                style={inputStyle}
              />
            </label>
          ))}
        </div>
        <button
          onClick={runSetpoint}
          disabled={isLoading}
          style={{ ...btnBase, background: 'rgba(159,142,109,0.15)', color: '#9F8E6D', opacity: isLoading ? 0.5 : 1 }}
        >
          {isLoading ? 'Running…' : '▶ Go to Setpoint'}
        </button>

        {/* Status / error */}
        {simError && (
          <div style={{ marginTop: '8px', fontSize: '10px', color: '#e06c75', wordBreak: 'break-word' }}>
            {simError}
          </div>
        )}
      </div>
    </div>
  );
}
