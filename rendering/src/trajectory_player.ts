import * as THREE from "three";

interface SimMeta {
  type: "meta";
  n_frames: number;
  dt: number;
  total_time: number;
}

interface SimFrame {
  type: "frame";
  i: number;
  t: number;
  pos: [number, number, number];    // meters, sim inertial (Z-up)
  quat: [number, number, number, number]; // [w, x, y, z] body→inertial
  engines: [[number, number], [number, number], [number, number]]; // [[α0,β0], [α1,β1], [α2,β2]] radians
}

export class TrajectoryPlayer {
  private readonly booster: THREE.Object3D;
  private readonly gimbalPivots: (THREE.Group | null)[];
  private readonly gimbalBaseQuat: THREE.Quaternion[];
  private readonly initialPos: THREE.Vector3;
  private readonly statusEl: HTMLElement;

  private ws: WebSocket | null = null;
  private meta: SimMeta | null = null;
  private frameA: SimFrame | null = null;
  private frameB: SimFrame | null = null;
  private startWallTime = 0;

  // Coordinate frame change: sim inertial (Z-up) → Three.js world (Y-up)
  // C = Rx(-π/2): maps sim-Z to three-Y and sim-Y to three-(-Z)
  // As quaternion [w, x, y, z]: C = [√2/2, -√2/2, 0, 0]
  // THREE.Quaternion constructor takes (x, y, z, w)
  private readonly _C    = new THREE.Quaternion(-Math.SQRT1_2, 0, 0, Math.SQRT1_2);
  private readonly _Cinv = new THREE.Quaternion( Math.SQRT1_2, 0, 0, Math.SQRT1_2);

  // Reusable scratch objects — never cross method boundaries
  private readonly _qa   = new THREE.Quaternion();
  private readonly _qb   = new THREE.Quaternion();
  private readonly _qtmp = new THREE.Quaternion();
  private readonly _pa   = new THREE.Vector3();
  private readonly _pb   = new THREE.Vector3();
  private readonly _qx   = new THREE.Quaternion();
  private readonly _qy   = new THREE.Quaternion();
  private readonly _axX  = new THREE.Vector3(1, 0, 0);
  private readonly _axY  = new THREE.Vector3(0, 1, 0);

  constructor(
    booster: THREE.Object3D,
    gimbalPivots: (THREE.Group | null)[],
    gimbalBaseQuat: THREE.Quaternion[],
  ) {
    this.booster       = booster;
    this.gimbalPivots  = gimbalPivots;
    this.gimbalBaseQuat = gimbalBaseQuat;
    this.initialPos    = booster.position.clone();
    this.statusEl      = this._makeStatusEl();
  }

  connect(url = "ws://localhost:8765"): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.meta    = null;
    this.frameA  = null;
    this.frameB  = null;

    this._setStatus("Connecting to simulation…");

    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this._setStatus("Connected — waiting for first frame…");
    };

    this.ws.onmessage = (evt: MessageEvent) => {
      this._onMessage(JSON.parse(evt.data) as SimMeta | SimFrame);
    };

    this.ws.onerror = () => {
      this._setStatus("Connection failed — is `python -m rocket.scenarios.sil_hover` running?", 6000);
    };

    this.ws.onclose = () => {
      if (this.meta) {
        this._setStatus(`Simulation complete  (${this.meta.total_time.toFixed(1)} s)`, 4000);
      }
      this.ws = null;
    };
  }

  disconnect(): void {
    this.ws?.close();
    this.ws = null;
    this._restoreBooster();
    this.statusEl.style.display = "none";
  }

  /** Call once per render frame from the main animation loop. */
  tick(): void {
    if (!this.frameA || !this.frameB || !this.meta) return;

    const elapsed = (performance.now() - this.startWallTime) / 1000;
    const dt      = this.meta.dt;
    const alpha   = Math.min(1, Math.max(0, (elapsed - this.frameA.t) / dt));

    this._applyFrame(this.frameA, this.frameB, alpha);
  }

  // ---------------------------------------------------------------------------

  private _onMessage(msg: SimMeta | SimFrame): void {
    if (msg.type === "meta") {
      this.meta          = msg as SimMeta;
      this.startWallTime = performance.now();
      return;
    }

    const frame = msg as SimFrame;
    this.frameA = this.frameB;
    this.frameB = frame;

    if (!this.frameA) {
      this.frameA = frame; // first frame — set both so tick() has something to work with
    }

    if (this.meta) {
      this._setStatus(`Sim  t = ${frame.t.toFixed(2)} s / ${this.meta.total_time.toFixed(1)} s`);
    }
  }

  private _applyFrame(a: SimFrame, b: SimFrame, alpha: number): void {
    const RAD2DEG = 180 / Math.PI;

    // --- position (lerp) ---
    this._simPosToThree(a.pos, this._pa);
    this._simPosToThree(b.pos, this._pb);
    this.booster.position.lerpVectors(this._pa, this._pb, alpha);

    // --- quaternion (slerp) ---
    this._simQuatToThree(a.quat, this._qa);
    this._simQuatToThree(b.quat, this._qb);
    this.booster.quaternion.slerpQuaternions(this._qa, this._qb, alpha);

    // --- engine gimbals (lerp angles) ---
    for (let i = 0; i < 3; i++) {
      const pivot = this.gimbalPivots[i];
      if (!pivot) continue;

      const alphaA = a.engines[i][0] * RAD2DEG;
      const betaA  = a.engines[i][1] * RAD2DEG;
      const alphaB = b.engines[i][0] * RAD2DEG;
      const betaB  = b.engines[i][1] * RAD2DEG;

      const al = (alphaA + (alphaB - alphaA) * alpha) * (Math.PI / 180);
      const be = (betaA  + (betaB  - betaA)  * alpha) * (Math.PI / 180);

      this._qx.setFromAxisAngle(this._axX, al);
      this._qy.setFromAxisAngle(this._axY, be);
      pivot.quaternion.copy(this.gimbalBaseQuat[i]).multiply(this._qx).multiply(this._qy);
    }
  }

  /** sim inertial pos [x,y,z] (m, Z-up) → Three.js world pos (mm, Y-up) */
  private _simPosToThree(pos: [number, number, number], out: THREE.Vector3): void {
    out.set(
      this.initialPos.x + pos[0] * 1000,
      this.initialPos.y + pos[2] * 1000,   // sim Z (up) → three Y (up)
      this.initialPos.z - pos[1] * 1000,   // sim Y → three -Z
    );
  }

  /** sim quat [w,x,y,z] (body→inertial Z-up) → Three.js quat (body→world Y-up) */
  private _simQuatToThree(q: [number, number, number, number], out: THREE.Quaternion): void {
    // THREE.Quaternion constructor order: (x, y, z, w)
    out.set(q[1], q[2], q[3], q[0]);
    // q_three = C * q_sim * C^-1
    this._qtmp.copy(this._C).multiply(out).multiply(this._Cinv);
    out.copy(this._qtmp);
  }

  private _restoreBooster(): void {
    this.booster.position.copy(this.initialPos);
    // Restore the nominal Three.js orientation: rotation.z = π/2
    // = THREE.Quaternion(x=0, y=0, z=√2/2, w=√2/2)
    this.booster.quaternion.set(0, 0, Math.SQRT1_2, Math.SQRT1_2);
    for (let i = 0; i < 3; i++) {
      const p = this.gimbalPivots[i];
      if (p) p.quaternion.copy(this.gimbalBaseQuat[i]);
    }
  }

  private _makeStatusEl(): HTMLElement {
    const el = document.createElement("div");
    el.style.cssText = `
      position:fixed; bottom:16px; right:16px; z-index:1000;
      font:13px/1 monospace; color:#8ab4d4;
      background:rgba(10,12,16,0.85); border:1px solid rgba(255,255,255,0.08);
      border-radius:6px; padding:6px 12px; pointer-events:none; display:none;
    `;
    document.body.appendChild(el);
    return el;
  }

  private _setStatus(text: string, hideAfterMs?: number): void {
    this.statusEl.textContent = text;
    this.statusEl.style.display = "block";
    if (hideAfterMs) setTimeout(() => { this.statusEl.style.display = "none"; }, hideAfterMs);
  }
}
