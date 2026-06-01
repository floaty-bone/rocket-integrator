// live_plots.ts — 12 live telemetry charts (optimised).

const PAD  = { top: 18, right: 52, bottom: 20, left: 46 };
const CW   = 400;
const CH   = 118;

// Minimum ms between redraws (~20 Hz is plenty for telemetry charts)
const RENDER_INTERVAL_MS = 50;

// ── Running min/max tracker ────────────────────────────────────────────────────
class Range {
  min = Infinity; max = -Infinity;
  reset() { this.min = Infinity; this.max = -Infinity; }
  add(v: number) { if (v < this.min) this.min = v; if (v > this.max) this.max = v; }
  addOther(r: Range) { if (r.min < this.min) this.min = r.min; if (r.max > this.max) this.max = r.max; }

  padded(fallback = 1): [number, number] {
    let lo = this.min, hi = this.max;
    if (!isFinite(lo)) { lo = -fallback; hi = fallback; }
    if (hi - lo < 1e-6) { lo -= fallback; hi += fallback; }
    const p = (hi - lo) * 0.15;
    return [lo - p, hi + p];
  }
}

// ── Static draw helpers ────────────────────────────────────────────────────────
function drawAxes(
  ctx: CanvasRenderingContext2D, pw: number, ph: number,
  yMinL: number, yMaxL: number, yMinR: number | null, yMaxR: number | null,
  title: string, leftColor: string, rightColor: string | null, leftUnit?: string,
) {
  const xp = PAD.left, yb = PAD.top + ph;
  ctx.strokeStyle = "rgba(255,255,255,0.05)"; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = PAD.top + (i / 4) * ph;
    ctx.beginPath(); ctx.moveTo(xp, y); ctx.lineTo(xp + pw, y); ctx.stroke();
  }
  if (yMinL < 0 && yMaxL > 0) {
    const y0 = PAD.top + ph - ((0 - yMinL) / (yMaxL - yMinL)) * ph;
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.beginPath(); ctx.moveTo(xp, y0); ctx.lineTo(xp + pw, y0); ctx.stroke();
  }
  ctx.strokeStyle = "rgba(255,255,255,0.2)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(xp, PAD.top); ctx.lineTo(xp, yb); ctx.lineTo(xp + pw, yb); ctx.stroke();
  if (rightColor) { ctx.beginPath(); ctx.moveTo(xp + pw, PAD.top); ctx.lineTo(xp + pw, yb); ctx.stroke(); }
  ctx.font = "9px monospace"; ctx.textAlign = "right"; ctx.fillStyle = leftColor;
  for (let i = 0; i <= 2; i++) {
    const v = yMinL + (i / 2) * (yMaxL - yMinL);
    const y = PAD.top + ph - ((v - yMinL) / (yMaxL - yMinL)) * ph;
    ctx.fillText(v.toFixed(1), xp - 4, y + 3);
  }
  if (rightColor && yMinR !== null && yMaxR !== null) {
    ctx.textAlign = "left"; ctx.fillStyle = rightColor;
    for (let i = 0; i <= 2; i++) {
      const v = yMinR + (i / 2) * (yMaxR - yMinR);
      const y = PAD.top + ph - ((v - yMinR) / (yMaxR - yMinR)) * ph;
      ctx.fillText(v.toFixed(1), xp + pw + 4, y + 3);
    }
  }
  ctx.fillStyle = "#8ab4d4"; ctx.font = "bold 9px monospace"; ctx.textAlign = "left"; ctx.fillText(title, xp + 3, PAD.top - 4);
  ctx.font = "8px monospace"; ctx.fillStyle = leftColor; ctx.textAlign = "left"; ctx.fillText(leftUnit ?? (leftColor === "#55cc88" ? "m" : "deg"), 2, PAD.top + 4);
  if (rightColor) { ctx.textAlign = "right"; ctx.fillStyle = rightColor; ctx.fillText("kN", CW - 2, PAD.top + 4); }
}

function polyline(
  ctx: CanvasRenderingContext2D, times: number[], data: number[],
  totalTime: number, pw: number, ph: number,
  yMin: number, yMax: number, color: string, lineWidth = 1.5, dashed = false,
) {
  const n = data.length;
  if (n < 2) return;
  ctx.strokeStyle = color; ctx.lineWidth = lineWidth;
  if (dashed) ctx.setLineDash([6, 4]); else ctx.setLineDash([]);
  const tMax = Math.max(totalTime, 1e-3);
  const yRange = yMax - yMin;
  ctx.beginPath();
  for (let j = 0; j < n; j++) {
    const x = PAD.left + (times[j] / tMax) * pw;
    const y = PAD.top + ph - ((data[j] - yMin) / yRange) * ph;
    j === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);
}

// Draw a horizontal dashed setpoint line without allocating an array
function hline(
  ctx: CanvasRenderingContext2D, value: number, pw: number, ph: number,
  yMin: number, yMax: number, color: string,
) {
  const y = PAD.top + ph - ((value - yMin) / (yMax - yMin)) * ph;
  ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash([6, 4]);
  ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + pw, y); ctx.stroke();
  ctx.setLineDash([]);
}

function legend(ctx: CanvasRenderingContext2D, ph: number, items: [string, string, number | null][]) {
  ctx.font = "9px monospace"; ctx.textAlign = "left";
  let lx = PAD.left + 3; const ly = PAD.top + ph + 13;
  for (const [color, lbl, val] of items) {
    ctx.fillStyle = color;
    const txt = val !== null ? `${lbl}=${val.toFixed(2)}` : lbl;
    ctx.fillText(txt, lx, ly);
    lx += ctx.measureText(txt).width + 10;
  }
}

// ── Per-chart draw functions ───────────────────────────────────────────────────
function drawPositionPlot(
  ctx: CanvasRenderingContext2D, axis: "X"|"Y"|"Z",
  times: number[], data: number[], setpoint: number,
  totalTime: number, range: Range,
) {
  const pw = CW - PAD.left - PAD.right, ph = CH - PAD.top - PAD.bottom;
  ctx.fillStyle = "#0a0c10"; ctx.fillRect(0, 0, CW, CH);
  const [yMin, yMax] = range.padded();
  drawAxes(ctx, pw, ph, yMin, yMax, null, null, `Position ${axis}`, "#55cc88", null);
  hline(ctx, setpoint, pw, ph, yMin, yMax, "#ff9955");
  polyline(ctx, times, data, totalTime, pw, ph, yMin, yMax, "#55cc88", 1.5);
  const last = data.length ? data[data.length - 1] : null;
  legend(ctx, ph, [["#55cc88", `pos ${axis}`, last], ["#ff9955", `sp ${axis}`, setpoint]]);
}

function drawEnginePlot(
  ctx: CanvasRenderingContext2D, label: string,
  times: number[], alpha: number[], beta: number[], thrust: number[],
  totalTime: number, abRange: Range, tRange: Range,
) {
  const pw = CW - PAD.left - PAD.right, ph = CH - PAD.top - PAD.bottom;
  ctx.fillStyle = "#0a0c10"; ctx.fillRect(0, 0, CW, CH);
  const [aMin, aMax] = abRange.padded();
  const [tMin, tMax] = tRange.padded();
  drawAxes(ctx, pw, ph, aMin, aMax, tMin, tMax, label, "#7ab4d4", "#ff9900");

  const tMaxT = Math.max(totalTime, 1e-3);
  const drawA = (d: number[], color: string) => {
    const n = d.length; if (n < 2) return;
    const span = aMax - aMin;
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.setLineDash([]);
    ctx.beginPath();
    for (let j = 0; j < n; j++) {
      const x = PAD.left + (times[j] / tMaxT) * pw;
      const y = PAD.top + ph - ((d[j] - aMin) / span) * ph;
      j === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  };
  const drawT = (d: number[], color: string) => {
    const n = d.length; if (n < 2) return;
    const span = tMax - tMin;
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.setLineDash([]);
    ctx.beginPath();
    for (let j = 0; j < n; j++) {
      const x = PAD.left + (times[j] / tMaxT) * pw;
      const y = PAD.top + ph - ((d[j] - tMin) / span) * ph;
      j === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  };
  drawA(alpha, "#00d4ff"); drawA(beta, "#55ff88"); drawT(thrust, "#ff9900");

  const lastA = alpha.length ? alpha[alpha.length - 1] : null;
  const lastB = beta.length  ? beta[beta.length - 1]   : null;
  const lastT = thrust.length ? thrust[thrust.length - 1] : null;
  // running abs max tracked in abMaxRange — recompute from Range.max (already abs values stored)
  const maxA = isFinite(abRange.max) ? abRange.max : null;
  const maxB = maxA; // same range object covers both
  legend(ctx, ph, [
    ["#00d4ff", `α=${lastA !== null ? lastA.toFixed(2) : "—"} mx=${maxA !== null ? maxA.toFixed(2) : "—"}`, null],
    ["#55ff88", `β=${lastB !== null ? lastB.toFixed(2) : "—"} mx=${maxB !== null ? maxB.toFixed(2) : "—"}`, null],
    ["#ff9900", lastT !== null ? `T=${lastT.toFixed(1)}kN` : "T=—", null],
  ]);
}

function drawOmegaPlot(
  ctx: CanvasRenderingContext2D, axis: "X"|"Y"|"Z",
  times: number[], data: number[], totalTime: number, range: Range,
) {
  const pw = CW - PAD.left - PAD.right, ph = CH - PAD.top - PAD.bottom;
  ctx.fillStyle = "#0a0c10"; ctx.fillRect(0, 0, CW, CH);
  const [yMin, yMax] = range.padded(0.1);
  drawAxes(ctx, pw, ph, yMin, yMax, null, null, `Body Rate ω${axis}`, "#c084fc", null, "rad/s");
  polyline(ctx, times, data, totalTime, pw, ph, yMin, yMax, "#c084fc", 1.5);
  const last = data.length ? data[data.length - 1] : null;
  legend(ctx, ph, [["#c084fc", `ω${axis}`, last]]);
}

function drawCartThrustPlot(
  ctx: CanvasRenderingContext2D, label: string,
  times: number[], fx: number[], fy: number[], fz: number[],
  totalTime: number, range: Range,
) {
  const pw = CW - PAD.left - PAD.right, ph = CH - PAD.top - PAD.bottom;
  ctx.fillStyle = "#0a0c10"; ctx.fillRect(0, 0, CW, CH);
  const [yMin, yMax] = range.padded();
  drawAxes(ctx, pw, ph, yMin, yMax, null, null, label, "#ff6666", null, "kN");
  polyline(ctx, times, fx, totalTime, pw, ph, yMin, yMax, "#ff6666", 1.5);
  polyline(ctx, times, fy, totalTime, pw, ph, yMin, yMax, "#66ff66", 1.5);
  polyline(ctx, times, fz, totalTime, pw, ph, yMin, yMax, "#6688ff", 1.5);
  const lastX = fx.length ? fx[fx.length - 1] : null;
  const lastY = fy.length ? fy[fy.length - 1] : null;
  const lastZ = fz.length ? fz[fz.length - 1] : null;
  legend(ctx, ph, [["#ff6666", "Fx", lastX], ["#66ff66", "Fy", lastY], ["#6688ff", "Fz", lastZ]]);
}

// ── LivePlots class ────────────────────────────────────────────────────────────
export class LivePlots {
  private readonly ctxs: CanvasRenderingContext2D[] = [];
  private _wrapperEl: HTMLElement | null = null;
  private _collapsed = true;
  private _dirty     = false;
  private _lastRender = 0;

  private times: number[] = [];
  private totalTime = 30;
  private setpoint: [number, number, number] = [0, 0, 0];
  private pos:    [number[], number[], number[]] = [[], [], []];
  private alpha:  [number[], number[], number[]] = [[], [], []];
  private beta:   [number[], number[], number[]] = [[], [], []];
  private thrust: [number[], number[], number[]] = [[], [], []];
  private omega:  [number[], number[], number[]] = [[], [], []];
  private cartFx: [number[], number[], number[]] = [[], [], []];
  private cartFy: [number[], number[], number[]] = [[], [], []];
  private cartFz: [number[], number[], number[]] = [[], [], []];

  // Running ranges — recomputed incrementally in addFrame, reset in setMeta
  private posRange:    Range[] = [new Range(), new Range(), new Range()];
  // For engine plots, abRange covers combined alpha+beta; tRange covers thrust
  private abRange:     Range[] = [new Range(), new Range(), new Range()];
  private thrustRange: Range[] = [new Range(), new Range(), new Range()];
  private omegaRange:  Range[] = [new Range(), new Range(), new Range()];
  private cartRange:   Range[] = [new Range(), new Range(), new Range()];

  constructor() { this._buildPanel(); }

  setMeta(totalTime: number, setpoint: [number, number, number]): void {
    this.totalTime = totalTime; this.setpoint = setpoint;
    this.times = []; this.pos = [[], [], []]; this.alpha = [[], [], []]; this.beta = [[], [], []];
    this.thrust = [[], [], []]; this.omega = [[], [], []];
    this.cartFx = [[], [], []]; this.cartFy = [[], [], []]; this.cartFz = [[], [], []];
    for (let i = 0; i < 3; i++) {
      this.posRange[i].reset(); this.abRange[i].reset();
      this.thrustRange[i].reset(); this.omegaRange[i].reset(); this.cartRange[i].reset();
      // Seed posRange with setpoint so the axis always includes it
      this.posRange[i].add(setpoint[i]);
    }
    this._dirty = true;
  }

  addFrame(
    t: number, pos: [number, number, number],
    engines: [number, number, number][],
    omega: [number, number, number],
    u_cart?: [number, number, number][],
  ): void {
    const R2D = 180 / Math.PI;
    this.times.push(t);
    for (let i = 0; i < 3; i++) {
      this.pos[i].push(pos[i]);
      this.posRange[i].add(pos[i]);

      const a = engines[i][0] * R2D, b = engines[i][1] * R2D;
      this.alpha[i].push(a); this.beta[i].push(b);
      this.abRange[i].add(a); this.abRange[i].add(b);

      const T = engines[i][2];
      const tk = (T !== undefined && isFinite(T)) ? T / 1000 : NaN;
      this.thrust[i].push(tk);
      if (isFinite(tk)) this.thrustRange[i].add(tk);

      this.omega[i].push(omega[i]);
      this.omegaRange[i].add(omega[i]);
    }
    if (u_cart) {
      for (let i = 0; i < 3; i++) {
        const fx = u_cart[i][0] / 1000, fy = u_cart[i][1] / 1000, fz = u_cart[i][2] / 1000;
        this.cartFx[i].push(fx); this.cartFy[i].push(fy); this.cartFz[i].push(fz);
        this.cartRange[i].add(fx); this.cartRange[i].add(fy); this.cartRange[i].add(fz);
      }
    }
    this._dirty = true;
  }

  render(): void {
    // Skip when collapsed or nothing changed or rendering too fast
    if (this._collapsed || !this._dirty) return;
    const now = performance.now();
    if (now - this._lastRender < RENDER_INTERVAL_MS) return;
    this._lastRender = now;
    this._dirty = false;

    const axLabels: ("X"|"Y"|"Z")[] = ["X", "Y", "Z"];
    for (let i = 0; i < 3; i++) {
      drawPositionPlot(this.ctxs[i], axLabels[i], this.times, this.pos[i], this.setpoint[i], this.totalTime, this.posRange[i]);
      drawEnginePlot(this.ctxs[3 + i], `Engine ${i + 1}`, this.times, this.alpha[i], this.beta[i], this.thrust[i], this.totalTime, this.abRange[i], this.thrustRange[i]);
      drawOmegaPlot(this.ctxs[6 + i], axLabels[i], this.times, this.omega[i], this.totalTime, this.omegaRange[i]);
      drawCartThrustPlot(this.ctxs[9 + i], `Engine ${i + 1} Cart`, this.times, this.cartFx[i], this.cartFy[i], this.cartFz[i], this.totalTime, this.cartRange[i]);
    }
  }

  dispose(): void {
    if (this._wrapperEl?.parentNode) this._wrapperEl.parentNode.removeChild(this._wrapperEl);
  }

  private _buildPanel(): void {
    const wrapper = document.createElement("div");
    this._wrapperEl = wrapper;
    wrapper.style.cssText = "position:fixed;top:20px;left:20px;z-index:1000;font:13px/1.6 monospace;color:#cdd6e0;";

    const titleBar = document.createElement("div");
    titleBar.style.cssText = "display:flex;align-items:center;justify-content:space-between;background:rgba(10,12,16,0.95);border:1px solid rgba(255,255,255,0.08);border-radius:10px 10px 0 0;padding:8px 14px;cursor:pointer;user-select:none;";
    const titleText = document.createElement("span");
    titleText.textContent = "Telemetry";
    titleText.style.cssText = "font-weight:bold;color:#8ab4d4;font-size:11px;letter-spacing:.08em;text-transform:uppercase;";
    const chevron = document.createElement("span");
    chevron.textContent = "▲"; chevron.style.cssText = "font-size:10px;color:#556070;transition:transform 0.2s;";
    titleBar.appendChild(titleText); titleBar.appendChild(chevron);

    const body = document.createElement("div");
    body.style.cssText = "background:rgba(10,12,16,0.88);border:1px solid rgba(255,255,255,0.08);border-top:none;border-radius:0 0 10px 10px;padding:10px 12px;display:flex;flex-direction:column;gap:8px;max-height:90vh;overflow-y:auto;will-change:transform;";

    const dpr = window.devicePixelRatio || 1;
    for (let i = 0; i < 12; i++) {
      const canvas = document.createElement("canvas");
      canvas.width = CW * dpr; canvas.height = CH * dpr;
      canvas.style.cssText = `width:${CW}px;height:${CH}px;display:block;border-radius:4px;`;
      body.appendChild(canvas);
      const ctx = canvas.getContext("2d")!;
      ctx.scale(dpr, dpr);
      this.ctxs.push(ctx);
    }

    // Start collapsed
    body.style.display = "none";
    chevron.style.transform = "rotate(180deg)";
    titleBar.addEventListener("click", () => {
      this._collapsed = !this._collapsed;
      body.style.display = this._collapsed ? "none" : "flex";
      chevron.style.transform = this._collapsed ? "rotate(180deg)" : "";
      // Force one redraw when opening so charts aren't blank
      if (!this._collapsed) { this._dirty = true; this._lastRender = 0; this.render(); }
    });

    wrapper.appendChild(titleBar); wrapper.appendChild(body);
    document.body.appendChild(wrapper);
  }
}
