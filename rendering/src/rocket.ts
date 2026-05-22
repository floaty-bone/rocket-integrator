import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

export interface RocketParts {
  gridFins: {
    FL: THREE.Object3D;
    FR: THREE.Object3D;
    RL: THREE.Object3D;
    RR: THREE.Object3D;
  };
  engines: {
    E1: THREE.Object3D;
    E2: THREE.Object3D;
    E3: THREE.Object3D;
  };
}

export async function loadRocket(scene: THREE.Scene): Promise<RocketParts> {
  const loader = new GLTFLoader();
  const gltf = await loader.loadAsync("/models/booster.glb");

  scene.add(gltf.scene);

  function getPart(name: string): THREE.Object3D {
    const obj = gltf.scene.getObjectByName(name);
    if (!obj) throw new Error(`Part "${name}" not found in GLB. Check Blender object names.`);
    return obj;
  }

  return {
    gridFins: {
      FL: getPart("grid_fin_FL"),
      FR: getPart("grid_fin_FR"),
      RL: getPart("grid_fin_RL"),
      RR: getPart("grid_fin_RR"),
    },
    engines: {
      E1: getPart("engine_center_1"),
      E2: getPart("engine_center_2"),
      E3: getPart("engine_center_3"),
    },
  };
}

// --- test animations ---

// spins all four grid fins 360° over `duration` seconds, then stops
export function testGridFinSpin(parts: RocketParts, duration = 4): () => void {
  const start = performance.now();
  let done = false;

  function tick() {
    if (done) return;
    const t = (performance.now() - start) / 1000;
    const angle = ((t / duration) * Math.PI * 2) % (Math.PI * 2);

    for (const fin of Object.values(parts.gridFins)) {
      fin.rotation.y = angle;
    }

    if (t >= duration) {
      for (const fin of Object.values(parts.gridFins)) fin.rotation.y = 0;
      done = true;
      return;
    }

    requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);
  return () => { done = true; };
}

// sweeps each center engine through a gimbal cone over `duration` seconds, then stops
export function testEngineGimbal(parts: RocketParts, maxAngleDeg = 8, duration = 4): () => void {
  const start = performance.now();
  const max = (maxAngleDeg * Math.PI) / 180;
  let done = false;

  // offset each engine's phase so they don't all move in sync
  const phases = [0, (Math.PI * 2) / 3, (Math.PI * 4) / 3];
  const engines = [parts.engines.E1, parts.engines.E2, parts.engines.E3];

  function tick() {
    if (done) return;
    const t = (performance.now() - start) / 1000;
    const cycle = (t / duration) * Math.PI * 2;

    engines.forEach((engine, i) => {
      engine.rotation.x = Math.sin(cycle + phases[i]) * max;
      engine.rotation.z = Math.cos(cycle + phases[i]) * max;
    });

    if (t >= duration) {
      engines.forEach((e) => { e.rotation.x = 0; e.rotation.z = 0; });
      done = true;
      return;
    }

    requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);
  return () => { done = true; };
}
