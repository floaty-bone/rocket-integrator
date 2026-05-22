// Booster assembly configuration — locked 2026-05-23
// All positions in mm, angles in radians unless noted.
// Coordinate convention: booster local X = world Y (up), rotation.z = +π/2 applied on load.

export const BOOSTER = {
  rotationZ: Math.PI / 2,          // booster model X → world Y
  frameRotationX: -Math.PI / 2,    // booster reference frame offset
  position: { x: -910, y: 48900, z: 540 },
};

export const STAGE0 = {
  position: { x: -4770, z: 37760 },
  rotationY: 0.3184,
};

// Grid fin world-space transforms (applied before model.attach)
// [px, py, pz, rx, ry, rz]
export const FIN_TRANSFORMS: [number, number, number, number, number, number][] = [
  [    0, 65860.3,  4520, -Math.PI / 2, Math.PI,            0],  // Fin 1
  [-4520, 65860.0,     0, -Math.PI / 2, Math.PI, Math.PI / 2 ],  // Fin 2
  [ 4520, 65860.0,     0, -Math.PI / 2, Math.PI,-Math.PI / 2 ],  // Fin 3
];

// Engine gimbal pivot world-space transforms (applied before model.attach)
// [px, py, pz, rx, ry, rz]
// Label mapping: index 0 → Engine 3, index 1 → Engine 2, index 2 → Engine 1
export const ENGINE_TRANSFORMS: [number, number, number, number, number, number][] = [
  [-1100, 2500,     0,  Math.PI / 2, 0, 0],  // Engine 3 (panel label)
  [  553, 2500,  950.5, Math.PI / 2, 0, 0],  // Engine 2 (panel label)
  [  553, 2500, -930.5, Math.PI / 2, 0, 0],  // Engine 1 (panel label)
];

// Panel label order → ENGINE_TRANSFORMS index
export const ENGINE_LABEL_ORDER: [string, number][] = [
  ["Engine 1", 2],
  ["Engine 2", 1],
  ["Engine 3", 0],
];

// Distance from engine model origin to gimbal pivot point, along engine local Z
export const ENGINE_GIMBAL_OFFSET_Z = 2900;

// Gimbal authority (degrees)
export const GIMBAL_LIMIT_DEG = 15;
