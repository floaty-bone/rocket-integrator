import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { startTrajectoryAnimation } from "./trajectory.js";
import { TrajectoryPlayer } from "./trajectory_player.js";
import {
  BOOSTER,
  STAGE0,
  FIN_TRANSFORMS,
  ENGINE_TRANSFORMS,
  ENGINE_LABEL_ORDER,
  ENGINE_GIMBAL_OFFSET_Z,
  GIMBAL_LIMIT_DEG,
} from "./assembly.config.js";

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
document.body.appendChild(renderer.domElement);

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

function makeMergedGrid(size: number, fineDiv: number, coarseDiv: number, y = 0): THREE.LineSegments {
  const positions: number[] = [];
  const colors: number[] = [];
  const half = size / 2;

  const addLine = (x1: number, z1: number, x2: number, z2: number, r: number, g: number, b: number, a: number) => {
    positions.push(x1, 0, z1, x2, 0, z2);
    colors.push(r, g, b, a, r, g, b, a);
  };

  for (let i = 0; i <= fineDiv; i++) {
    const t = (i / fineDiv) * size - half;
    const isCoarse = i % (fineDiv / coarseDiv) === 0;
    const [r, g, b, a] = isCoarse ? [0.47, 0.47, 0.47, 0.85] : [0.33, 0.33, 0.33, 0.5];
    addLine(t, -half, t, half, r, g, b, a);
    addLine(-half, t, half, t, r, g, b, a);
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 4));
  const mat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false });
  const mesh = new THREE.LineSegments(geo, mat);
  mesh.position.y = y;
  return mesh;
}

// const ground = new THREE.Mesh(
//   new THREE.PlaneGeometry(600000, 600000),
//   new THREE.MeshStandardMaterial({ color: 0x1a1b1e, roughness: 0.9, metalness: 0.1 })
// );
// ground.rotation.x = -Math.PI / 2;
// ground.position.y = -0.5;
// scene.add(ground);


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

let stage0Model: THREE.Object3D | null = null;
let boosterModel: THREE.Object3D | null = null;

const gimbalPivots: (THREE.Group | null)[]     = [null, null, null];
const gimbalBaseQuat: THREE.Quaternion[]        = [new THREE.Quaternion(), new THREE.Quaternion(), new THREE.Quaternion()];
let trajectoryPlayer: TrajectoryPlayer | null   = null;

let leftChopstick: THREE.Object3D | null = null;
let rightChopstick: THREE.Object3D | null = null;
let chopConnector: THREE.Object3D | null = null;
const leftChopstickBaseQ = new THREE.Quaternion();
const rightChopstickBaseQ = new THREE.Quaternion();
let chopConnectorBaseY = 0;

// chopstick current angles (degrees) tracked so the oscillation knows the rest position
let leftChopAngleDeg = 0;
let rightChopAngleDeg = 0;

// damped oscillation state per chopstick
const CHOP_AMP_PER_DEGPS = 0.12;  // degrees of wobble per deg/s of rotation speed
const CHOP_AMP_MAX = 6.0;          // hard cap (degrees)
const CHOP_GAMMA = 0.7;            // damping coefficient
const CHOP_OMEGA = 5.0;            // angular frequency rad/s
const CHOP_ROT_SPEED_REF = 10;     // deg/s treated as "default" speed for manual wobble buttons
type ChopOsc = { active: boolean; startTime: number; baseAngle: number; amp: number };
const leftOsc: ChopOsc = { active: false, startTime: 0, baseAngle: 0, amp: CHOP_AMP_PER_DEGPS * CHOP_ROT_SPEED_REF };
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

function tickChopstick(
  obj: THREE.Object3D | null, baseQ: THREE.Quaternion,
  sliderDeg: number, oscY: ChopOsc, oscX: ChopOsc, signX: number,
  minDeg: number, maxDeg: number
) {
  if (!obj || !(oscY.active || oscX.active)) return;
  const degY = Math.min(maxDeg, Math.max(minDeg, sliderDeg + oscOffset(oscY, 1)));
  const degX = oscOffset(oscX, signX);
  _chopQa.setFromAxisAngle(CHOP_LOCAL_Y, degY * (Math.PI / 180));
  _chopQb.setFromAxisAngle(CHOP_LOCAL_X, degX * (Math.PI / 180));
  obj.quaternion.copy(baseQ).multiply(_chopQa).multiply(_chopQb);
}

const CHOP_LOCAL_Y = new THREE.Vector3(0, 1, 0);
const CHOP_LOCAL_X = new THREE.Vector3(1, 0, 0);

const leftOscX: ChopOsc = { active: false, startTime: 0, baseAngle: 0, amp: 0.8 };
const rightOscX: ChopOsc = { active: false, startTime: 0, baseAngle: 0, amp: 0.8 };

// translational wobble for the connector node
const CHOP_TRANS_AMP_PER_MPS = 0.2;  // metres of wobble per m/s of translation speed
const CHOP_TRANS_AMP_MAX = 4.0;
const CHOP_TRANS_SPEED_REF = 5;       // m/s treated as "default" for manual wobble buttons
const CHOP_TRANS_OMEGA = 3.5;
type TransOsc = { active: boolean; startTime: number; basePos: number; amp: number };
const connTransOsc: TransOsc = { active: false, startTime: 0, basePos: 0, amp: CHOP_TRANS_AMP_PER_MPS * CHOP_TRANS_SPEED_REF };

function tickTransOscillation(osc: TransOsc, obj: THREE.Object3D | null) {
  if (!osc.active || !obj) return;
  const t = (performance.now() - osc.startTime) / 1000;
  const offset = -osc.amp * Math.exp(-CHOP_GAMMA * t) * Math.sin(CHOP_TRANS_OMEGA * t);
  obj.position.y = osc.basePos + offset;
  if (Math.exp(-CHOP_GAMMA * t) < 0.005) {
    osc.active = false;
    obj.position.y = osc.basePos;
  }
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
  const now = performance.now();
  const xAmp = Math.min(CHOP_AMP_MAX, CHOP_AMP_PER_DEGPS * transSpeed * 0.5);
  connTransOsc.amp = Math.min(CHOP_TRANS_AMP_MAX, CHOP_TRANS_AMP_PER_MPS * transSpeed);
  connTransOsc.basePos = chopConnector.position.y;
  connTransOsc.startTime = now;
  connTransOsc.active = true;
  leftOscX.amp = xAmp; leftOscX.startTime = now; leftOscX.active = true;
  rightOscX.amp = xAmp; rightOscX.startTime = now; rightOscX.active = true;
}

// stage 0 launch structure
loader.load("/models/stage0.glb", (gltf) => {
  const model = gltf.scene;
  stage0Model = model;
  model.scale.setScalar(1000);
  model.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(model);
  model.position.y -= box.min.y;
  model.position.x = STAGE0.position.x;
  model.position.z = STAGE0.position.z;
  model.rotation.y = STAGE0.rotationY;
  scene.add(model);

  const chopstickMat = new THREE.MeshStandardMaterial({ color: 0x1a1c1f, roughness: 0.3, metalness: 0.85, envMapIntensity: 1.8 });
  const concreteMat = new THREE.MeshStandardMaterial({ color: 0x52575c, roughness: 0.75, metalness: 0.05, envMapIntensity: 0.8 });
  const launchPadMat2 = new THREE.MeshStandardMaterial({ color: 0x52575c, roughness: 0.92, metalness: 0.0, envMapIntensity: 0.5 });

  const meshBox = new THREE.Box3();
  model.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (!mesh.isMesh) return;
    console.log(mesh);
    mesh.castShadow = true;
    if (["left_chopstick", "right_chopstick", "QDBoosterClaw2001", "Chopstick_TowerConnector002"].includes(mesh.name)) {
      if (mesh.name === "left_chopstick") { leftChopstick = mesh; leftChopstickBaseQ.copy(mesh.quaternion); }
      if (mesh.name === "right_chopstick") { rightChopstick = mesh; rightChopstickBaseQ.copy(mesh.quaternion); }
      mesh.material = chopstickMat;
    } else if (["Tower_base", "top-section", "mid-section"].includes(mesh.name)) {
      mesh.material = concreteMat;
    } else if (mesh.name === "launch_pad") {
      mesh.material = launchPadMat2;
    }
  });

  const connNode = model.getObjectByName("Chopstick_TowerConnector002");
  if (connNode) {
    chopConnector = connNode;
    chopConnectorBaseY = connNode.position.y;
  }
}, undefined, (err) => console.error("Stage0 load error:", err));

loader.load(
  "/models/booster_asm.glb",
  (gltf) => {
    const model = gltf.scene;
    model.rotation.z = BOOSTER.rotationZ;

    const steel = new THREE.MeshStandardMaterial({
      color: 0x6a7075, roughness: 0.45, metalness: 0.75, envMapIntensity: 1.2,
    });
    const engineBlack = new THREE.MeshStandardMaterial({
      color: 0x080808, roughness: 0.95, metalness: 0.1, envMapIntensity: 0.1,
    });

    model.updateMatrixWorld(true);
    const modelBox = new THREE.Box3().setFromObject(model);
    const modelMinY = modelBox.min.y;
    const modelHeight = modelBox.max.y - modelBox.min.y;
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

    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());

    model.position.set(BOOSTER.position.x, BOOSTER.position.y, BOOSTER.position.z);
    model.updateMatrixWorld(true);

    // ENGINE/FIN_TRANSFORMS were authored as world positions when the model sat at
    // (-center.x, -box.min.y, -center.z) with rotZ=π/2. Converting to model-local:
    //   local = rotInv * (worldPos - oldModelWorldPos)
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
    const fov = (camera.fov * Math.PI) / 180;
    const dist = (radius / Math.sin(fov / 2)) * 1.5;
    camera.position.set(dist * 0.6, dist * 0.5, dist * 0.8);
    camera.lookAt(0, size.y / 2, 0);
    controls.target.set(0, size.y / 2, 0);

    camera.near = dist * 0.001;
    camera.far = dist * 20;
    camera.updateProjectionMatrix();
    scene.fog = new THREE.FogExp2(0x0d0e10, 0.6 / (dist * 8));
    controls.maxDistance = dist * 8;

    controls.update();

    const gridSize = radius * 30;
    const fine = Math.round(gridSize / radius) * 10;
    scene.add(makeGrid(gridSize, fine, 0x555555, 0.5, 0));
    scene.add(makeGrid(gridSize, Math.round(fine / 10), 0x777777, 0.85, 0));

    const groundDisc = new THREE.Mesh(
      new THREE.PlaneGeometry(radius * 4, radius * 4),
      new THREE.MeshStandardMaterial({ color: 0x1a1b1e, roughness: 0.9, metalness: 0.1 })
    );
    groundDisc.rotation.x = -Math.PI / 2;
    groundDisc.position.y = -0.5;
    groundDisc.receiveShadow = true;
    scene.add(groundDisc);

    const s = size.y * 1.5;
    sun.shadow.camera.left = -s;
    sun.shadow.camera.right = s;
    sun.shadow.camera.top = s * 2;
    sun.shadow.camera.bottom = -s;
    sun.shadow.camera.far = size.y * 10;
    sun.target.position.set(model.position.x, size.y / 2, model.position.z);
    sun.target.updateMatrixWorld();
    sun.shadow.camera.updateProjectionMatrix();

    const comOffset = center.clone().sub(model.position);

    trajectoryPlayer = new TrajectoryPlayer(model, gimbalPivots, gimbalBaseQuat);

    window.addEventListener("keydown", (e) => {
      if (e.key === "t" || e.key === "T") {
        model.position.set(-center.x, -box.min.y, -center.z);
        model.rotation.z = BOOSTER.rotationZ;
        startTrajectoryAnimation(scene, model, comOffset, size.y);
      }
      if (e.key === "p" || e.key === "P") {
        trajectoryPlayer?.connect();
      }
    });

    const addFrame = (target: THREE.Object3D, axisLength: number) => {
      target.add(new THREE.AxesHelper(axisLength));
      const makeLabel = (text: string, color: string): THREE.Sprite => {
        const canvas = document.createElement("canvas");
        canvas.width = 128; canvas.height = 64;
        const ctx = canvas.getContext("2d")!;
        ctx.fillStyle = color;
        ctx.font = "bold 48px monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(text, 64, 32);
        const mat = new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthTest: false, transparent: true });
        const sprite = new THREE.Sprite(mat);
        sprite.scale.setScalar(axisLength * 0.4);
        return sprite;
      };
      const x = makeLabel("X", "#ff4444"); x.position.set(axisLength * 1.15, 0, 0); target.add(x);
      const y = makeLabel("Y", "#44ff44"); y.position.set(0, axisLength * 1.15, 0); target.add(y);
      const z = makeLabel("Z", "#4488ff"); z.position.set(0, 0, axisLength * 1.15); target.add(z);
    };

    // booster reference frame
    const boosterFrame = new THREE.Group();
    boosterFrame.rotation.x = BOOSTER.frameRotationX;
    addFrame(boosterFrame, size.z * 0.9);
    model.add(boosterFrame);

    // --- grid fins ---
    const finMat = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 1.0, metalness: 0.0 });
    const fins: (THREE.Object3D | null)[] = [null, null, null];
    const finBaseQuat: THREE.Quaternion[] = [new THREE.Quaternion(), new THREE.Quaternion(), new THREE.Quaternion()];

    const attachFin = (fin: THREE.Object3D, px: number, py: number, pz: number, rx: number, ry: number, rz: number, idx: number) => {
      fin.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          (child as THREE.Mesh).material = finMat;
          (child as THREE.Mesh).castShadow = true;
        }
      });
      fin.position.copy(toModelLocal(px, py, pz));
      fin.quaternion.copy(toModelLocalQuat(rx, ry, rz));
      model.add(fin);
      fins[idx] = fin;
      finBaseQuat[idx].copy(fin.quaternion);
    };

    const FIN_TRANSFORMS: [number, number, number, number, number, number][] = [
      [0, 65860.3, 4520, -Math.PI / 2, Math.PI, 0],
      [-4520, 65860.0, 0, -Math.PI / 2, Math.PI, Math.PI / 2],
      [4520, 65860.0, 0, -Math.PI / 2, Math.PI, -Math.PI / 2],
    ];

    FIN_TRANSFORMS.forEach(([px, py, pz, rx, ry, rz], i) => {
      loader.load(
        "/models/grid_fin_ass_asm.glb",
        (gltfFin) => attachFin(gltfFin.scene, px, py, pz, rx, ry, rz, i),
        undefined,
        (err) => console.error(`Failed to load grid fin ${i + 1}:`, err)
      );
    });

    // --- engines ---

    const engMat = new THREE.MeshStandardMaterial({ color: 0x7a3010, roughness: 0.8, metalness: 0.2, envMapIntensity: 0.6 });

    ENGINE_TRANSFORMS.forEach(([px, py, pz, rx, ry, rz], i) => {
      loader.load(
        "/models/engine.glb",
        (gltfEng) => {
          const eng = gltfEng.scene;
          eng.traverse((child) => {
            if ((child as THREE.Mesh).isMesh) {
              (child as THREE.Mesh).material = engMat;
              (child as THREE.Mesh).castShadow = true;
              (child as THREE.Mesh).receiveShadow = true;
            }
          });
          eng.position.z = ENGINE_GIMBAL_OFFSET_Z;

          const pivot = new THREE.Group();
          addFrame(pivot, size.z * 0.5);
          pivot.add(eng);

          pivot.position.copy(toModelLocal(px, py, pz));
          pivot.quaternion.copy(toModelLocalQuat(rx, ry, rz));
          model.add(pivot);

          gimbalPivots[i] = pivot;
          gimbalBaseQuat[i].copy(pivot.quaternion);
        },
        undefined,
        (err) => console.error(`Failed to load engine ${i + 1}:`, err)
      );
    });

    boosterModel = model;
    buildControlPanel(model, fins, finBaseQuat, gimbalPivots, gimbalBaseQuat);

    console.log(`Booster loaded — size: ${size.x.toFixed(1)} x ${size.y.toFixed(1)} x ${size.z.toFixed(1)}`);
  },
  (xhr) => console.log(`Loading: ${((xhr.loaded / xhr.total) * 100).toFixed(0)}%`),
  (err) => console.error("Failed to load booster:", err)
);


function buildControlPanel(
  _booster: THREE.Object3D,
  fins: (THREE.Object3D | null)[],
  finBaseQuat: THREE.Quaternion[],
  gimbalPivots: (THREE.Group | null)[],
  gimbalBaseQuat: THREE.Quaternion[]
) {
  const DEG = Math.PI / 180;

  const wrapper = document.createElement("div");
  wrapper.style.cssText = `
    position:fixed; top:20px; right:20px; z-index:1000;
    font:13px/1.6 monospace; color:#cdd6e0;
  `;

  const titleBar = document.createElement("div");
  titleBar.style.cssText = `
    display:flex; align-items:center; justify-content:space-between;
    background:rgba(10,12,16,0.95);     border:1px solid rgba(255,255,255,0.08); border-radius:10px 10px 0 0;
    padding:8px 14px; cursor:pointer; user-select:none;
  `;
  const titleText = document.createElement("span");
  titleText.textContent = "Controls";
  titleText.style.cssText = "font-weight:bold; color:#8ab4d4; font-size:11px; letter-spacing:.08em; text-transform:uppercase;";
  const chevron = document.createElement("span");
  chevron.textContent = "▲";
  chevron.style.cssText = "font-size:10px; color:#556070; transition:transform 0.2s;";
  titleBar.appendChild(titleText);
  titleBar.appendChild(chevron);

  const panel = document.createElement("div");
  panel.style.cssText = `
    background:rgba(10,12,16,0.88);     border:1px solid rgba(255,255,255,0.08); border-top:none; border-radius:0 0 10px 10px;
    padding:14px 22px; display:flex; flex-direction:column; gap:6px; min-width:380px;
  `;

  let collapsed = false;
  titleBar.addEventListener("click", () => {
    collapsed = !collapsed;
    panel.style.display = collapsed ? "none" : "flex";
    chevron.style.transform = collapsed ? "rotate(180deg)" : "";
  });

  wrapper.appendChild(titleBar);
  wrapper.appendChild(panel);

  const makeHeader = (text: string) => {
    const h = document.createElement("div");
    h.textContent = text;
    h.style.cssText = "font-weight:bold; color:#8ab4d4; font-size:11px; letter-spacing:.08em; text-transform:uppercase; margin-bottom:2px;";
    return h;
  };

  const makeRow = (label: string, min: number, max: number, initial: number, unit: string, onChange: (v: number) => void, onRelease?: () => void) => {
    const row = document.createElement("div");
    row.style.cssText = "display:flex; align-items:center; gap:10px;";

    const lbl = document.createElement("span");
    lbl.textContent = label;
    lbl.style.cssText = "width:56px; font-size:12px; color:#8a9bb0; flex-shrink:0;";

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = String(min); slider.max = String(max); slider.step = "0.5";
    slider.value = String(initial);
    slider.style.cssText = "flex:1; accent-color:#8ab4d4; cursor:pointer;";

    const num = document.createElement("input");
    num.type = "number";
    num.min = String(min); num.max = String(max); num.step = "0.5";
    num.value = String(initial);
    num.style.cssText = `
      width:58px; flex-shrink:0; background:#0d1117; border:1px solid #2a3a4a;
      border-radius:4px; color:#e0e8f0; font:12px monospace; padding:2px 5px;
      text-align:right; -moz-appearance:textfield;
    `;

    const apply = (v: number) => {
      const c = Math.min(max, Math.max(min, v));
      slider.value = String(c); num.value = String(c);
      onChange(c);
    };

    slider.addEventListener("input", () => apply(parseFloat(slider.value)));
    if (onRelease) slider.addEventListener("pointerup", onRelease);
    num.addEventListener("change", () => { apply(parseFloat(num.value)); onRelease?.(); });
    num.addEventListener("keydown", (e) => { if (e.key === "Enter") { apply(parseFloat(num.value)); onRelease?.(); } });

    const valLbl = document.createElement("span");
    valLbl.textContent = unit;
    valLbl.style.cssText = "font-size:11px; color:#556070; flex-shrink:0;";

    row.appendChild(lbl); row.appendChild(slider); row.appendChild(num); row.appendChild(valLbl);
    return row;
  };

  // chopstick rotation around local Y
  panel.appendChild(makeHeader("Tower — Chopsticks (local Y)"));
  const localChopY = new THREE.Vector3(0, 1, 0);
  const qc = new THREE.Quaternion();

  const makeWobbleBtn = (label: string, triggerOsc: ChopOsc, getAngle: () => number) => {
    const btn = document.createElement("button");
    btn.textContent = `Wobble ${label}`;
    btn.style.cssText = `
      padding:4px 10px; font:11px monospace; cursor:pointer;
      background:#0d1117; color:#8ab4d4; border:1px solid #2a3a4a;
      border-radius:4px; align-self:flex-start;
    `;
    btn.addEventListener("click", () => {
      triggerOsc.baseAngle = getAngle();
      triggerOsc.startTime = performance.now();
      triggerOsc.active = true;
    });
    return btn;
  };

  panel.appendChild(makeRow("Left", -8, 60, 0, "°", (deg) => {
    leftChopAngleDeg = deg;
    if (!leftChopstick || leftOsc.active) return;
    qc.setFromAxisAngle(localChopY, deg * DEG);
    leftChopstick.quaternion.copy(leftChopstickBaseQ).multiply(qc);
  }, triggerLeftStopWobble));
  panel.appendChild(makeWobbleBtn("Left", leftOsc, () => leftChopAngleDeg));

  panel.appendChild(makeRow("Right", -60, 8, 0, "°", (deg) => {
    rightChopAngleDeg = deg;
    if (!rightChopstick || rightOsc.active) return;
    qc.setFromAxisAngle(localChopY, deg * DEG);
    rightChopstick.quaternion.copy(rightChopstickBaseQ).multiply(qc);
  }, triggerRightStopWobble));
  panel.appendChild(makeWobbleBtn("Right", rightOsc, () => rightChopAngleDeg));

  panel.appendChild(makeRow("Height", -50, 50, 0, "m", (v) => {
    if (!chopConnector) return;
    chopConnector.position.y = chopConnectorBaseY + v;
  }, triggerTransWobble));

  const wobbleXBtn = document.createElement("button");
  wobbleXBtn.textContent = "Wobble (local X)";
  wobbleXBtn.style.cssText = `
    padding:4px 10px; font:11px monospace; cursor:pointer;
    background:#0d1117; color:#8ab4d4; border:1px solid #2a3a4a;
    border-radius:4px; align-self:flex-start;
  `;
  wobbleXBtn.addEventListener("click", () => {
    const now = performance.now();
    leftOscX.baseAngle = 0; leftOscX.startTime = now; leftOscX.active = true;
    rightOscX.baseAngle = 0; rightOscX.startTime = now; rightOscX.active = true;
    if (chopConnector) {
      connTransOsc.basePos = chopConnector.position.y;
      connTransOsc.startTime = now;
      connTransOsc.active = true;
    }
  });
  panel.appendChild(wobbleXBtn);

  // grid fin pivots
  const localY = new THREE.Vector3(0, 1, 0);
  const q = new THREE.Quaternion();

  panel.appendChild(makeHeader("Grid Fins — Pivot (local Y)"));
  ["Fin 1", "Fin 2", "Fin 3"].forEach((label, i) => {
    panel.appendChild(makeRow(label, -90, 90, 0, "°", (deg) => {
      const fin = fins[i]; if (!fin) return;
      q.setFromAxisAngle(localY, deg * DEG);
      fin.quaternion.copy(finBaseQuat[i]).multiply(q);
    }));
  });

  // engine gimbal sliders — Rx and Ry per engine
  const qx = new THREE.Quaternion();
  const qy = new THREE.Quaternion();
  const localX = new THREE.Vector3(1, 0, 0);
  const localYv = new THREE.Vector3(0, 1, 0);
  const engRx = [0, 0, 0];
  const engRy = [0, 0, 0];

  ENGINE_LABEL_ORDER.forEach(([label, i]) => {
    panel.appendChild(makeHeader(`${label} — Gimbal`));
    panel.appendChild(makeRow("Rx", -GIMBAL_LIMIT_DEG, GIMBAL_LIMIT_DEG, 0, "°", (deg) => {
      const pivot = gimbalPivots[i]; if (!pivot) return;
      engRx[i] = deg;
      qx.setFromAxisAngle(localX, engRx[i] * DEG);
      qy.setFromAxisAngle(localYv, engRy[i] * DEG);
      pivot.quaternion.copy(gimbalBaseQuat[i]).multiply(qx).multiply(qy);
    }));
    panel.appendChild(makeRow("Ry", -GIMBAL_LIMIT_DEG, GIMBAL_LIMIT_DEG, 0, "°", (deg) => {
      const pivot = gimbalPivots[i]; if (!pivot) return;
      engRy[i] = deg;
      qx.setFromAxisAngle(localX, engRx[i] * DEG);
      qy.setFromAxisAngle(localYv, engRy[i] * DEG);
      pivot.quaternion.copy(gimbalBaseQuat[i]).multiply(qx).multiply(qy);
    }));
  });

  document.body.appendChild(wrapper);

  // ── Position Panel ─────────────────────────────────────────────────────────
  const posWrapper = document.createElement("div");
  posWrapper.style.cssText = `
    position:fixed; top:20px; right:440px; z-index:1000;
    font:13px/1.6 monospace; color:#cdd6e0;
  `;

  const posTitleBar = document.createElement("div");
  posTitleBar.style.cssText = `
    display:flex; align-items:center; justify-content:space-between;
    background:rgba(10,12,16,0.95);     border:1px solid rgba(255,255,255,0.08); border-radius:10px 10px 0 0;
    padding:8px 14px; cursor:pointer; user-select:none;
  `;
  const posTitleText = document.createElement("span");
  posTitleText.textContent = "Position";
  posTitleText.style.cssText = "font-weight:bold; color:#8ab4d4; font-size:11px; letter-spacing:.08em; text-transform:uppercase;";
  const posChevron = document.createElement("span");
  posChevron.textContent = "▲";
  posChevron.style.cssText = "font-size:10px; color:#556070; transition:transform 0.2s;";
  posTitleBar.appendChild(posTitleText);
  posTitleBar.appendChild(posChevron);

  const posPanel = document.createElement("div");
  posPanel.style.cssText = `
    background:rgba(10,12,16,0.88);     border:1px solid rgba(255,255,255,0.08); border-top:none; border-radius:0 0 10px 10px;
    padding:14px 22px; display:flex; flex-direction:column; gap:8px; min-width:320px;
  `;

  let posCollapsed = false;
  posTitleBar.addEventListener("click", () => {
    posCollapsed = !posCollapsed;
    posPanel.style.display = posCollapsed ? "none" : "flex";
    posChevron.style.transform = posCollapsed ? "rotate(180deg)" : "";
  });

  posWrapper.appendChild(posTitleBar);
  posWrapper.appendChild(posPanel);

  // target values
  let posTargetLeft = 0;
  let posTargetRight = 0;
  let posTargetHeight = 0;
  let posRotRate = 10;   // deg/s
  let posTransSpeed = 5; // m/s

  const makePosRow = (label: string, min: number, max: number, initial: number, unit: string, onChange: (v: number) => void) => {
    const row = document.createElement("div");
    row.style.cssText = "display:flex; align-items:center; gap:10px;";
    const lbl = document.createElement("span");
    lbl.textContent = label;
    lbl.style.cssText = "width:100px; font-size:12px; color:#8a9bb0; flex-shrink:0;";
    const num = document.createElement("input");
    num.type = "number";
    num.min = String(min); num.max = String(max); num.step = "0.5";
    num.value = String(initial);
    num.style.cssText = `
      flex:1; background:#0d1117; border:1px solid #2a3a4a;
      border-radius:4px; color:#e0e8f0; font:12px monospace; padding:2px 5px;
      text-align:right; -moz-appearance:textfield;
    `;
    const apply = (v: number) => {
      const c = Math.min(max, Math.max(min, v));
      num.value = String(c);
      onChange(c);
    };
    num.addEventListener("change", () => apply(parseFloat(num.value)));
    num.addEventListener("keydown", (e) => { if (e.key === "Enter") apply(parseFloat(num.value)); });
    const valLbl = document.createElement("span");
    valLbl.textContent = unit;
    valLbl.style.cssText = "font-size:11px; color:#556070; flex-shrink:0;";
    row.appendChild(lbl); row.appendChild(num); row.appendChild(valLbl);
    return row;
  };

  const makePosHeader = (text: string) => {
    const h = document.createElement("div");
    h.textContent = text;
    h.style.cssText = "font-weight:bold; color:#8ab4d4; font-size:11px; letter-spacing:.08em; text-transform:uppercase; margin-top:4px;";
    return h;
  };

  posPanel.appendChild(makePosHeader("Target Angles"));
  posPanel.appendChild(makePosRow("Left chopstick", -8, 60, 0, "°", (v) => { posTargetLeft = v; }));
  posPanel.appendChild(makePosRow("Right chopstick", -60, 8, 0, "°", (v) => { posTargetRight = v; }));
  posPanel.appendChild(makePosHeader("Target Height"));
  posPanel.appendChild(makePosRow("Assembly", -50, 50, 0, "m", (v) => { posTargetHeight = v; }));
  posPanel.appendChild(makePosHeader("Move Speed"));
  posPanel.appendChild(makePosRow("Rotation rate", 1, 90, 10, "°/s", (v) => { posRotRate = v; }));
  posPanel.appendChild(makePosRow("Trans speed", 0.1, 50, 5, "m/s", (v) => { posTransSpeed = v; }));

  // animated move state
  type PosAnim = { active: boolean; startTime: number; startVal: number; endVal: number; duration: number; onUpdate: (v: number) => void; onDone: () => void };
  const posAnims: PosAnim[] = [];

  function tickPosAnims() {
    const now = performance.now();
    for (let i = posAnims.length - 1; i >= 0; i--) {
      const a = posAnims[i];
      if (!a.active) { posAnims.splice(i, 1); continue; }
      const t = Math.min(1, (now - a.startTime) / (a.duration * 1000));
      const ease = t;
      a.onUpdate(a.startVal + (a.endVal - a.startVal) * ease);
      if (t >= 1) { a.active = false; a.onDone(); posAnims.splice(i, 1); }
    }
  }

  // patch animate loop to call tickPosAnims
  (window as any).__tickPosAnims = tickPosAnims;

  const execBtn = document.createElement("button");
  execBtn.textContent = "Execute";
  execBtn.style.cssText = `
    margin-top:6px; padding:6px 14px; font:12px monospace; cursor:pointer;
    background:#162030; color:#8ab4d4; border:1px solid #2a3a4a;
    border-radius:6px; align-self:stretch; letter-spacing:.05em;
    transition: background 0.15s;
  `;
  execBtn.addEventListener("mouseenter", () => { execBtn.style.background = "#1e3048"; });
  execBtn.addEventListener("mouseleave", () => { execBtn.style.background = "#162030"; });

  execBtn.addEventListener("click", () => {
    const localChopYv = new THREE.Vector3(0, 1, 0);
    const qTmp = new THREE.Quaternion();

    const startLeft = leftChopAngleDeg;
    const startRight = rightChopAngleDeg;
    const startHeight = chopConnector ? chopConnector.position.y - chopConnectorBaseY : 0;

    const rotDuration = Math.max(
      Math.abs(posTargetLeft - startLeft),
      Math.abs(posTargetRight - startRight)
    ) / posRotRate;

    const transDuration = Math.abs(posTargetHeight - startHeight) / posTransSpeed;

    const hasRotLeft = Math.abs(posTargetLeft - startLeft) > 0.01;
    const hasRotRight = Math.abs(posTargetRight - startRight) > 0.01;
    const hasTrans = Math.abs(posTargetHeight - startHeight) > 0.01 && !!chopConnector;

    if (hasRotLeft) triggerLeftWobble(posRotRate);
    if (hasRotRight) triggerRightWobble(posRotRate);
    if (hasTrans) triggerTransWobble(posTransSpeed);

    if (hasRotLeft) {
      posAnims.push({
        active: true, startTime: performance.now(),
        startVal: startLeft, endVal: posTargetLeft, duration: rotDuration,
        onUpdate: (v) => {
          leftChopAngleDeg = v;
          if (leftChopstick && !leftOsc.active && !leftOscX.active) {
            qTmp.setFromAxisAngle(localChopYv, v * (Math.PI / 180));
            leftChopstick.quaternion.copy(leftChopstickBaseQ).multiply(qTmp);
          }
        },
        onDone: () => { triggerLeftStopWobble(posRotRate); }
      });
    }

    if (hasRotRight) {
      posAnims.push({
        active: true, startTime: performance.now(),
        startVal: startRight, endVal: posTargetRight, duration: rotDuration,
        onUpdate: (v) => {
          rightChopAngleDeg = v;
          if (rightChopstick && !rightOsc.active && !rightOscX.active) {
            qTmp.setFromAxisAngle(localChopYv, v * (Math.PI / 180));
            rightChopstick.quaternion.copy(rightChopstickBaseQ).multiply(qTmp);
          }
        },
        onDone: () => { triggerRightStopWobble(posRotRate); }
      });
    }

    if (hasTrans) {
      posAnims.push({
        active: true, startTime: performance.now(),
        startVal: startHeight, endVal: posTargetHeight, duration: transDuration,
        onUpdate: (v) => {
          if (chopConnector) {
            const newY = chopConnectorBaseY + v;
            chopConnector.position.y = newY;
            connTransOsc.basePos = newY;
          }
        },
        onDone: () => { triggerTransWobble(posTransSpeed); }
      });
    }
  });

  posPanel.appendChild(execBtn);
  document.body.appendChild(posWrapper);
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

const fpsEl = document.createElement("div");
fpsEl.style.cssText = `
  position:fixed; bottom:16px; left:16px; z-index:1000;
  font:13px/1 monospace; color:#8ab4d4;
  background:rgba(10,12,16,0.75);   border:1px solid rgba(255,255,255,0.08); border-radius:6px;
  padding:5px 10px; pointer-events:none;
`;
document.body.appendChild(fpsEl);

let fpsFrames = 0;
let fpsLast = performance.now();

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  if ((window as any).__tickPosAnims) (window as any).__tickPosAnims();
  trajectoryPlayer?.tick();
  tickChopstick(leftChopstick, leftChopstickBaseQ, leftChopAngleDeg, leftOsc, leftOscX, 1, -8, 60);
  tickChopstick(rightChopstick, rightChopstickBaseQ, rightChopAngleDeg, rightOsc, rightOscX, -1, -60, 8);
  tickTransOscillation(connTransOsc, chopConnector);
  renderer.render(scene, camera);

  fpsFrames++;
  const now = performance.now();
  if (now - fpsLast >= 500) {
    fpsEl.textContent = `${Math.round(fpsFrames * 1000 / (now - fpsLast))} FPS`;
    fpsFrames = 0;
    fpsLast = now;
  }
}

animate();
