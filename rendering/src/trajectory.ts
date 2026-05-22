import * as THREE from "three";

// keyframe path: up then arcing right
const PATH_POINTS = [
  new THREE.Vector3(0,    0,    0),
  new THREE.Vector3(0,    30,   0),
  new THREE.Vector3(0,    70,   0),
  new THREE.Vector3(10,   110,  0),
  new THREE.Vector3(40,   140,  0),
  new THREE.Vector3(90,   155,  0),
  new THREE.Vector3(150,  150,  0),
];

const CURVE = new THREE.CatmullRomCurve3(PATH_POINTS);
const DURATION = 5; // seconds

// live trail that grows as the model moves
function makeTrail(scene: THREE.Scene, maxPoints: number): {
  update: (pos: THREE.Vector3) => void;
} {
  const positions = new Float32Array(maxPoints * 3);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setDrawRange(0, 0);

  const mat = new THREE.LineBasicMaterial({ color: 0x88aacc, linewidth: 1 });
  const line = new THREE.Line(geo, mat);
  scene.add(line);

  let count = 0;

  return {
    update(pos: THREE.Vector3) {
      if (count >= maxPoints) return;
      positions[count * 3 + 0] = pos.x;
      positions[count * 3 + 1] = pos.y;
      positions[count * 3 + 2] = pos.z;
      count++;
      geo.setDrawRange(0, count);
      (geo.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    },
  };
}

export function startTrajectoryAnimation(
  scene: THREE.Scene,
  model: THREE.Object3D,
  comOffset: THREE.Vector3,
  modelHeight: number       // used to scale the path to the model's units
): () => void {
  const trail = makeTrail(scene, 2000);

  // build a scaled reference spline and add it to the scene
  const scale = modelHeight * 3;
  const scaledPoints = PATH_POINTS.map((p) => p.clone().multiplyScalar(scale / 150));
  const scaledCurve = new THREE.CatmullRomCurve3(scaledPoints);

  const refGeo = new THREE.BufferGeometry().setFromPoints(scaledCurve.getPoints(200));
  const refMat = new THREE.LineDashedMaterial({ color: 0x445566, dashSize: modelHeight * 0.3, gapSize: modelHeight * 0.2 });
  const refLine = new THREE.Line(refGeo, refMat);
  refLine.computeLineDistances();
  scene.add(refLine);

  const start = performance.now();
  const origin = model.position.clone();
  let active = true;

  const markerRadius = modelHeight * 0.02;
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(markerRadius, 8, 8),
    new THREE.MeshBasicMaterial({ color: 0xff4444 })
  );
  scene.add(marker);

  function tick() {
    if (!active) return;

    const t = Math.min((performance.now() - start) / 1000 / DURATION, 1);
    const pathPos = scaledCurve.getPoint(t);

    // pure translation — no rotation change
    model.position.set(
      origin.x + pathPos.x,
      origin.y + pathPos.y,
      origin.z + pathPos.z
    );

    const comWorld = model.position.clone().add(comOffset);
    marker.position.copy(comWorld);
    trail.update(comWorld);

    if (t < 1) requestAnimationFrame(tick);
    else active = false;
  }

  requestAnimationFrame(tick);
  return () => { active = false; };
}
