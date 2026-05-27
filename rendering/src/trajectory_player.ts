import * as THREE from "three";

interface SimMeta {
  type: "meta";
  n_frames: number;
  dt: number;
  total_time: number;
  setpoint: [number, number, number];
}

interface SimFrame {
  type: "frame";
  i: number;
  t: number;
  pos:     [number, number, number];    // meters, sim inertial (Z-up)
  quat:    [number, number, number, number]; // [w, x, y, z] body→inertial
  engines: [[number, number, number], [number, number, number], [number, number, number]]; // [[α,β,T]×3] rad,rad,N
  omega:   [number, number, number];   // body angular rates [ωx, ωy, ωz] rad/s
  u_cart?: [[number, number, number], [number, number, number], [number, number, number]]; // [[Fx,Fy,Fz]×3] N
}

export class TrajectoryPlayer {
  private readonly booster: THREE.Object3D;
  private readonly gimbalPivots: (THREE.Group | null)[];
  private readonly gimbalBaseQuat: THREE.Quaternion[];
  private readonly initialPos: THREE.Vector3;
  private readonly statusEl: HTMLElement;

  public frozen = false;
  public thrustArrows: THREE.ArrowHelper[] = [];
  public thrustArrowBaseLen = 0;   // full-length reference (Three.js units)
  public thrustRefN = 1_000_000;   // thrust (N) that maps to full arrow length

  onMeta?:     (totalTime: number, setpoint: [number, number, number]) => void;
  onFrame?:    (t: number, pos: [number, number, number], engines: [number, number, number][], omega: [number, number, number], u_cart?: [number, number, number][]) => void;
  onComplete?: () => void;

  private ws: WebSocket | null = null;
  private meta: SimMeta | null = null;
  private frameA: SimFrame | null = null;
  private frameB: SimFrame | null = null;
  private startWallTime = 0;

  // Coordinate frame change: sim inertial (Z-up) → Three.js world (Y-up)
  // C = Rx(-π/2): maps sim-Z to three-Y and sim-Y to three-(-Z)
  private readonly _C    = new THREE.Quaternion(-Math.SQRT1_2, 0, 0, Math.SQRT1_2);
  private readonly _Cinv = new THREE.Quaternion( Math.SQRT1_2, 0, 0, Math.SQRT1_2);

  // Reusable scratch objects
  private readonly _qa   = new THREE.Quaternion();
  private readonly _qb   = new THREE.Quaternion();
  private readonly _qtmp = new THREE.Quaternion();
  private readonly _pa   = new THREE.Vector3();
  private readonly _pb   = new THREE.Vector3();
  private readonly _qx   = new THREE.Quaternion();
  private readonly _qy   = new THREE.Quaternion();
  private readonly _axX  = new THREE.Vector3(1, 0, 0);
  private readonly _axY  = new THREE.Vector3(0, 1, 0);

  // CoM trail
  private readonly _scene: THREE.Scene;
  private _trailLine:  THREE.Line | null = null;
  private _trailBuf:   Float32Array | null = null;
  private _trailCount  = 0;
  private static readonly TRAIL_MAX = 4000;

  constructor(
    booster: THREE.Object3D,
    gimbalPivots: (THREE.Group | null)[],
    gimbalBaseQuat: THREE.Quaternion[],
    scene: THREE.Scene,
  ) {
    this.booster        = booster;
    this.gimbalPivots   = gimbalPivots;
    this.gimbalBaseQuat = gimbalBaseQuat;
    this.initialPos     = booster.position.clone();
    this.statusEl       = this._makeStatusEl();
    this._scene         = scene;
  }

  connect(url = "ws://localhost:8765"): void {
    if (this.ws) { this.ws.onopen = null; this.ws.onmessage = null; this.ws.onerror = null; this.ws.onclose = null; this.ws.close(); this.ws = null; }
    this.meta   = null;
    this.frameA = null;
    this.frameB = null;
    this._initTrail();

    this._setStatus("Connecting to simulation…");
    const ws = new WebSocket(url);
    this.ws = ws;

    ws.onopen    = () => { if (this.ws !== ws) return; this._setStatus("Connected — waiting for first frame…"); };
    ws.onmessage = (evt: MessageEvent) => {
      if (this.ws !== ws) return;
      this._onMessage(JSON.parse(evt.data) as SimMeta | SimFrame);
    };
    ws.onerror   = () => {
      if (this.ws !== ws) return;
      this._setStatus("Connection failed — is `python -m rocket.scenarios.sil_hover` running?", 6000);
    };
    ws.onclose   = () => {
      if (this.ws !== ws) return;
      if (this.meta) {
        this._setStatus(`Simulation complete  (${this.meta.total_time.toFixed(1)} s)`, 4000);
        this.onComplete?.();
      }
      this.ws = null;
    };
  }

  disconnect(): void {
    this.ws?.close();
    this.ws = null;
    this._restoreBooster();
    if (this._trailLine) {
      this._scene.remove(this._trailLine);
      this._trailLine.geometry.dispose();
      this._trailLine = null;
    }
    this.statusEl.style.display = "none";
  }

  get latestPos(): [number, number, number] | null {
    return this.frameB?.pos ?? null;
  }

  /** Call once per render frame from the main animation loop. */
  tick(): void {
    if (this.frozen) return;
    if (!this.frameA || !this.frameB || !this.meta) return;
    try {
      const elapsed = (performance.now() - this.startWallTime) / 1000;
      const dt      = this.meta.dt;
      const alpha   = Math.min(1, Math.max(0, (elapsed - this.frameA.t) / dt));
      this._applyFrame(this.frameA, this.frameB, alpha);
      this._appendTrail();
    } catch (e) {
      console.error("TrajectoryPlayer.tick error:", e);
    }
  }

  // ---------------------------------------------------------------------------

  private _onMessage(msg: SimMeta | SimFrame): void {
    if (msg.type === "meta") {
      this.meta          = msg as SimMeta;
      this.startWallTime = performance.now();
      this.onMeta?.(this.meta.total_time, this.meta.setpoint);
      return;
    }

    const frame = msg as SimFrame;
    this.frameA = this.frameB;
    this.frameB = frame;
    if (!this.frameA) this.frameA = frame;

    this.onFrame?.(frame.t, frame.pos, frame.engines, frame.omega ?? [0, 0, 0], frame.u_cart);

    if (this.meta) {
      this._setStatus(`Sim  t = ${frame.t.toFixed(2)} s / ${this.meta.total_time.toFixed(1)} s`);
    }
  }

  private _applyFrame(a: SimFrame, b: SimFrame, alpha: number): void {
    this._simPosToThree(a.pos, this._pa);
    this._simPosToThree(b.pos, this._pb);
    this.booster.position.lerpVectors(this._pa, this._pb, alpha);

    this._simQuatToThree(a.quat, this._qa);
    this._simQuatToThree(b.quat, this._qb);
    this.booster.quaternion.slerpQuaternions(this._qa, this._qb, alpha);

    for (let i = 0; i < 3; i++) {
      const pivot = this.gimbalPivots[i];
      if (!pivot) continue;
      const al = -(a.engines[i][1] + (b.engines[i][1] - a.engines[i][1]) * alpha);
      const be = -(a.engines[i][0] + (b.engines[i][0] - a.engines[i][0]) * alpha);
      this._qx.setFromAxisAngle(this._axX, al);
      this._qy.setFromAxisAngle(this._axY, be);
      pivot.quaternion.copy(this.gimbalBaseQuat[i]).multiply(this._qx).multiply(this._qy);

      const arrow = this.thrustArrows[i];
      if (arrow && this.thrustArrowBaseLen > 0) {
        const thrust = a.engines[i][2] + (b.engines[i][2] - a.engines[i][2]) * alpha;
        const frac   = Math.max(0, Math.min(1, thrust / this.thrustRefN));
        const len    = this.thrustArrowBaseLen * frac;
        arrow.setLength(len, len * 0.08, len * 0.04);
      }
    }
  }

  private _simPosToThree(pos: [number, number, number], out: THREE.Vector3): void {
    out.set(
      this.initialPos.x + pos[0] * 1000,
      this.initialPos.y + pos[2] * 1000,
      this.initialPos.z - pos[1] * 1000,
    );
  }

  private _simQuatToThree(q: [number, number, number, number], out: THREE.Quaternion): void {
    out.set(q[1], q[2], q[3], q[0]);
    this._qtmp.copy(this._C).multiply(out).multiply(this._Cinv);
    out.copy(this._qtmp);
  }

  private _initTrail(): void {
    if (this._trailLine) {
      this._scene.remove(this._trailLine);
      this._trailLine.geometry.dispose();
      this._trailLine = null;
    }
    const max = TrajectoryPlayer.TRAIL_MAX;
    this._trailBuf   = new Float32Array(max * 3);
    this._trailCount = 0;

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(this._trailBuf, 3));
    geo.setDrawRange(0, 0);

    const mat = new THREE.LineBasicMaterial({
      color: 0x00d4ff,
      transparent: true,
      opacity: 0.6,
      depthWrite: false,
    });
    this._trailLine = new THREE.Line(geo, mat);
    this._trailLine.frustumCulled = false;
    this._scene.add(this._trailLine);
  }

  private _appendTrail(): void {
    if (!this._trailLine || !this._trailBuf) return;
    if (this._trailCount >= TrajectoryPlayer.TRAIL_MAX) return;

    const p   = this.booster.position;
    const idx = this._trailCount * 3;
    this._trailBuf[idx]     = p.x;
    this._trailBuf[idx + 1] = p.y;
    this._trailBuf[idx + 2] = p.z;
    this._trailCount++;

    const attr = this._trailLine.geometry.getAttribute("position") as THREE.BufferAttribute;
    attr.needsUpdate = true;
    this._trailLine.geometry.setDrawRange(0, this._trailCount);
  }

  private _restoreBooster(): void {
    this.booster.position.copy(this.initialPos);
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
