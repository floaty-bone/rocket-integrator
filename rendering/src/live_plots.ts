// live_plots.ts — 12 live telemetry charts fed from the WebSocket stream.
//   Plots 0-2  : X / Y / Z position vs time, with setpoint dashed line.
//   Plots 3-5  : Engine 1 / 2 / 3 — α & β (left axis, deg) + thrust (right axis, kN).
//   Plots 6-8  : Body angular rates ωx / ωy / ωz (rad/s).
//   Plots 9-11 : Engine 1 / 2 / 3 — raw Cartesian thrust Fx / Fy / Fz (kN).


const PAD  = { top: 18, right: 52, bottom: 20, left: 46 };
const CW   = 400;
const CH   = 118;

// ─── helpers ──────────────────────────────────────────────────────────────────

function yRange(arr: number[], fallback = 1): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (const v of arr) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!isFinite(lo)) { lo = -fallback; hi = fallback; }
  if (hi - lo < 1e-6) { lo -= fallback; hi += fallback; }
  const p = (hi - lo) * 0.15;
  return [lo - p, hi + p];
}

function drawAxes(
  ctx: CanvasRenderingContext2D,
  pw: number, ph: number,
  yMinL: number, yMaxL: number,
  yMinR: number | null, yMaxR: number | null,
  title: string,
  leftColor: string,
  rightColor: string | null,
  leftUnit?: string,
) {
  const xp = PAD.left, yb = PAD.top + ph;

  // grid
  ctx.strokeStyle = "rgba(255,255,255,0.05)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = PAD.top + (i / 4) * ph;
    ctx.beginPath(); ctx.moveTo(xp, y); ctx.lineTo(xp + pw, y); ctx.stroke();
  }

  // zero line (left axis)
  if (yMinL < 0 && yMaxL > 0) {
    const y0 = PAD.top + ph - ((0 - yMinL) / (yMaxL - yMinL)) * ph;
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.beginPath(); ctx.moveTo(xp, y0); ctx.lineTo(xp + pw, y0); ctx.stroke();
  }

  // frame
  ctx.strokeStyle = "rgba(255,255,255,0.2)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(xp, PAD.top); ctx.lineTo(xp, yb); ctx.lineTo(xp + pw, yb);
  ctx.stroke();
  if (rightColor) {
    ctx.beginPath(); ctx.moveTo(xp + pw, PAD.top); ctx.lineTo(xp + pw, yb); ctx.stroke();
  }

  // left Y ticks
  ctx.font = "9px monospace"; ctx.textAlign = "right";
  ctx.fillStyle = leftColor;
  for (let i = 0; i <= 2; i++) {
    const v = yMinL + (i / 2) * (yMaxL - yMinL);
    const y = PAD.top + ph - ((v - yMinL) / (yMaxL - yMinL)) * ph;
    ctx.fillText(v.toFixed(1), xp - 4, y + 3);
  }

  // right Y ticks
  if (rightColor && yMinR !== null && yMaxR !== null) {
    ctx.textAlign = "left";
    ctx.fillStyle = rightColor;
    for (let i = 0; i <= 2; i++) {
      const v = yMinR + (i / 2) * (yMaxR - yMinR);
      const y = PAD.top + ph - ((v - yMinR) / (yMaxR - yMinR)) * ph;
      ctx.fillText(v.toFixed(1), xp + pw + 4, y + 3);
    }
  }

  // title
  ctx.fillStyle = "#8ab4d4";
  ctx.font = "bold 9px monospace"; ctx.textAlign = "left";
  ctx.fillText(title, xp + 3, PAD.top - 4);

  // unit labels
  ctx.font = "8px monospace";
  ctx.fillStyle = leftColor;
  ctx.textAlign = "left";
  ctx.fillText(leftUnit ?? (leftColor === "#55cc88" ? "m" : "deg"), 2, PAD.top + 4);
  if (rightColor) {
    ctx.textAlign = "right";
    ctx.fillStyle = rightColor;
    ctx.fillText("kN", CW - 2, PAD.top + 4);
  }
}

function polyline(
  ctx: CanvasRenderingContext2D,
  times: number[], data: number[],
  totalTime: number, pw: number, ph: number,
  yMin: number, yMax: number,
  color: string, lineWidth = 1.5,
  dashed = false,
) {
  if (data.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  if (dashed) ctx.setLineDash([6, 4]); else ctx.setLineDash([]);
  ctx.beginPath();
  for (let j = 0; j < data.length; j++) {
    const x = PAD.left + (times[j] / Math.max(totalTime, 1e-3)) * pw;
    const y = PAD.top + ph - ((data[j] - yMin) / (yMax - yMin)) * ph;
    j === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);
}

function legend(
  ctx: CanvasRenderingContext2D, ph: number,
  items: [string, string, number | null][],  // [color, label, lastValue]
) {
  ctx.font = "9px monospace"; ctx.textAlign = "left";
  let lx = PAD.left + 3;
  const ly = PAD.top + ph + 13;
  for (const [color, lbl, val] of items) {
    ctx.fillStyle = color;
    const txt = val !== null ? `${lbl}=${val.toFixed(2)}` : lbl;
    ctx.fillText(txt, lx, ly);
    lx += ctx.measureText(txt).width + 10;
  }
}

// ─── draw functions ───────────────────────────────────────────────────────────

function drawPositionPlot(
  ctx: CanvasRenderingContext2D,
  axis: "X" | "Y" | "Z",
  times: number[], data: number[], setpoint: number,
  totalTime: number,
) {
  const pw = CW - PAD.left - PAD.right;
  const ph = CH - PAD.top  - PAD.bottom;

  ctx.fillStyle = "#0a0c10";
  ctx.fillRect(0, 0, CW, CH);

  const combined = [...data, setpoint];
  const [yMin, yMax] = yRange(combined);

  drawAxes(ctx, pw, ph, yMin, yMax, null, null, `Position ${axis}`, "#55cc88", null);

  // setpoint dashed line
  const spArr = times.map(() => setpoint);
  polyline(ctx, times.length ? times : [0, totalTime], times.length ? spArr : [setpoint, setpoint],
    totalTime, pw, ph, yMin, yMax, "#ff9955", 1, true);

  // actual position
  polyline(ctx, times, data, totalTime, pw, ph, yMin, yMax, "#55cc88", 1.5);

  const last = data.length ? data[data.length - 1] : null;
  legend(ctx, ph, [
    ["#55cc88", `pos ${axis}`, last],
    ["#ff9955", `sp ${axis}`,  setpoint],
  ]);
}

function drawEnginePlot(
  ctx: CanvasRenderingContext2D,
  label: string,
  times: number[],
  alpha: number[], beta: number[], thrust: number[],
  totalTime: number,
) {
  const pw = CW - PAD.left - PAD.right;
  const ph = CH - PAD.top  - PAD.bottom;

  ctx.fillStyle = "#0a0c10";
  ctx.fillRect(0, 0, CW, CH);

  const [aMin, aMax] = yRange([...alpha, ...beta]);
  const [tMin, tMax] = yRange(thrust);

  drawAxes(ctx, pw, ph, aMin, aMax, tMin, tMax, label, "#7ab4d4", "#ff9900");

  const ypA = (v: number) => PAD.top + ph - ((v - aMin) / (aMax - aMin)) * ph;
  const ypT = (v: number) => PAD.top + ph - ((v - tMin) / (tMax - tMin)) * ph;

  // draw using the angle Y scale
  const drawA = (data: number[], color: string) => {
    if (data.length < 2) return;
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.setLineDash([]);
    ctx.beginPath();
    for (let j = 0; j < data.length; j++) {
      const x = PAD.left + (times[j] / Math.max(totalTime, 1e-3)) * pw;
      j === 0 ? ctx.moveTo(x, ypA(data[j])) : ctx.lineTo(x, ypA(data[j]));
    }
    ctx.stroke();
  };

  // draw using the thrust Y scale
  const drawT = (data: number[], color: string) => {
    if (data.length < 2) return;
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.setLineDash([]);
    ctx.beginPath();
    for (let j = 0; j < data.length; j++) {
      const x = PAD.left + (times[j] / Math.max(totalTime, 1e-3)) * pw;
      j === 0 ? ctx.moveTo(x, ypT(data[j])) : ctx.lineTo(x, ypT(data[j]));
    }
    ctx.stroke();
  };

  drawA(alpha,  "#00d4ff");
  drawA(beta,   "#55ff88");
  drawT(thrust, "#ff9900");

  const lastA = alpha.length  ? alpha[alpha.length - 1]   : null;
  const lastB = beta.length   ? beta[beta.length - 1]     : null;
  const lastT = thrust.length ? thrust[thrust.length - 1] : null;
  const maxA  = alpha.length  ? Math.max(...alpha.map(Math.abs))  : null;
  const maxB  = beta.length   ? Math.max(...beta.map(Math.abs))   : null;
  legend(ctx, ph, [
    ["#00d4ff", `α=${lastA !== null ? lastA.toFixed(2) : "—"} mx=${maxA !== null ? maxA.toFixed(2) : "—"}`, null],
    ["#55ff88", `β=${lastB !== null ? lastB.toFixed(2) : "—"} mx=${maxB !== null ? maxB.toFixed(2) : "—"}`, null],
    ["#ff9900", lastT !== null ? `T=${lastT.toFixed(1)}kN` : "T=—", null],
  ]);
}

function drawOmegaPlot(
  ctx: CanvasRenderingContext2D,
  axis: "X" | "Y" | "Z",
  times: number[], data: number[],
  totalTime: number,
) {
  const pw = CW - PAD.left - PAD.right;
  const ph = CH - PAD.top  - PAD.bottom;

  ctx.fillStyle = "#0a0c10";
  ctx.fillRect(0, 0, CW, CH);

  const [yMin, yMax] = yRange(data.length ? data : [0], 0.1);

  drawAxes(ctx, pw, ph, yMin, yMax, null, null, `Body Rate ω${axis}`, "#c084fc", null, "rad/s");

  polyline(ctx, times, data, totalTime, pw, ph, yMin, yMax, "#c084fc", 1.5);

  const last = data.length ? data[data.length - 1] : null;
  legend(ctx, ph, [["#c084fc", `ω${axis}`, last]]);
}

function drawCartThrustPlot(
  ctx: CanvasRenderingContext2D,
  label: string,
  times: number[],
  fx: number[], fy: number[], fz: number[],
  totalTime: number,
) {
  const pw = CW - PAD.left - PAD.right;
  const ph = CH - PAD.top  - PAD.bottom;

  ctx.fillStyle = "#0a0c10";
  ctx.fillRect(0, 0, CW, CH);

  const [yMin, yMax] = yRange([...fx, ...fy, ...fz]);

  drawAxes(ctx, pw, ph, yMin, yMax, null, null, label, "#ff6666", null, "kN");

  polyline(ctx, times, fx, totalTime, pw, ph, yMin, yMax, "#ff6666", 1.5);
  polyline(ctx, times, fy, totalTime, pw, ph, yMin, yMax, "#66ff66", 1.5);
  polyline(ctx, times, fz, totalTime, pw, ph, yMin, yMax, "#6688ff", 1.5);

  const lastX = fx.length ? fx[fx.length - 1] : null;
  const lastY = fy.length ? fy[fy.length - 1] : null;
  const lastZ = fz.length ? fz[fz.length - 1] : null;
  legend(ctx, ph, [
    ["#ff6666", "Fx", lastX],
    ["#66ff66", "Fy", lastY],
    ["#6688ff", "Fz", lastZ],
  ]);
}

// ─── LivePlots class ──────────────────────────────────────────────────────────

export class LivePlots {
  private readonly ctxs: CanvasRenderingContext2D[] = [];

  private times:  number[] = [];
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

  constructor() { this._buildPanel(); }

  setMeta(totalTime: number, setpoint: [number, number, number]): void {
    this.totalTime = totalTime;
    this.setpoint  = setpoint;
    this.times = [];
    this.pos    = [[], [], []];
    this.alpha  = [[], [], []];
    this.beta   = [[], [], []];
    this.thrust = [[], [], []];
    this.omega  = [[], [], []];
    this.cartFx = [[], [], []];
    this.cartFy = [[], [], []];
    this.cartFz = [[], [], []];
  }

  addFrame(
    t: number,
    pos: [number, number, number],
    engines: [number, number, number][],
    omega: [number, number, number],
    u_cart?: [number, number, number][],
  ): void {
    const R2D = 180 / Math.PI;
    this.times.push(t);
    for (let i = 0; i < 3; i++) this.pos[i].push(pos[i]);
    for (let i = 0; i < 3; i++) {
      this.alpha[i].push(engines[i][0]  * R2D);
      this.beta[i].push(engines[i][1]   * R2D);
      const T = engines[i][2];
      this.thrust[i].push((T !== undefined && isFinite(T)) ? T / 1000 : NaN);
    }
    for (let i = 0; i < 3; i++) this.omega[i].push(omega[i]);
    if (u_cart) {
      for (let i = 0; i < 3; i++) {
        this.cartFx[i].push(u_cart[i][0] / 1000);
        this.cartFy[i].push(u_cart[i][1] / 1000);
        this.cartFz[i].push(u_cart[i][2] / 1000);
      }
    }
  }

  render(): void {
    const axLabels: ("X" | "Y" | "Z")[] = ["X", "Y", "Z"];
    for (let i = 0; i < 3; i++) {
      drawPositionPlot(
        this.ctxs[i], axLabels[i],
        this.times, this.pos[i], this.setpoint[i],
        this.totalTime,
      );
    }
    for (let i = 0; i < 3; i++) {
      drawEnginePlot(
        this.ctxs[3 + i], `Engine ${i + 1}`,
        this.times, this.alpha[i], this.beta[i], this.thrust[i],
        this.totalTime,
      );
    }
    const omegaLabels: ("X" | "Y" | "Z")[] = ["X", "Y", "Z"];
    for (let i = 0; i < 3; i++) {
      drawOmegaPlot(
        this.ctxs[6 + i], omegaLabels[i],
        this.times, this.omega[i],
        this.totalTime,
      );
    }
    for (let i = 0; i < 3; i++) {
      drawCartThrustPlot(
        this.ctxs[9 + i], `Engine ${i + 1} Cart`,
        this.times, this.cartFx[i], this.cartFy[i], this.cartFz[i],
        this.totalTime,
      );
    }
  }

  private _buildPanel(): void {
    const wrapper = document.createElement("div");
    wrapper.style.cssText = `
      position:fixed; top:20px; left:20px; z-index:1000;
      font:13px/1.6 monospace; color:#cdd6e0;
    `;

    const titleBar = document.createElement("div");
    titleBar.style.cssText = `
      display:flex; align-items:center; justify-content:space-between;
      background:rgba(10,12,16,0.95); border:1px solid rgba(255,255,255,0.08);
      border-radius:10px 10px 0 0; padding:8px 14px; cursor:pointer; user-select:none;
    `;
    const titleText = document.createElement("span");
    titleText.textContent = "Telemetry";
    titleText.style.cssText = "font-weight:bold; color:#8ab4d4; font-size:11px; letter-spacing:.08em; text-transform:uppercase;";
    const chevron = document.createElement("span");
    chevron.textContent = "▲";
    chevron.style.cssText = "font-size:10px; color:#556070; transition:transform 0.2s;";
    titleBar.appendChild(titleText);
    titleBar.appendChild(chevron);

    const body = document.createElement("div");
    body.style.cssText = `
      background:rgba(10,12,16,0.88); border:1px solid rgba(255,255,255,0.08);
      border-top:none; border-radius:0 0 10px 10px;
      padding:10px 12px; display:flex; flex-direction:column; gap:8px;
      max-height:90vh; overflow-y:auto;
    `;

    const dpr = window.devicePixelRatio || 1;
    for (let i = 0; i < 12; i++) {
      const canvas = document.createElement("canvas");
      canvas.width  = CW * dpr;
      canvas.height = CH * dpr;
      canvas.style.width  = `${CW}px`;
      canvas.style.height = `${CH}px`;
      canvas.style.display = "block";
      canvas.style.borderRadius = "4px";
      body.appendChild(canvas);
      const ctx = canvas.getContext("2d")!;
      ctx.scale(dpr, dpr);
      this.ctxs.push(ctx);
    }

    let collapsed = false;
    titleBar.addEventListener("click", () => {
      collapsed = !collapsed;
      body.style.display = collapsed ? "none" : "flex";
      chevron.style.transform = collapsed ? "rotate(180deg)" : "";
    });

    wrapper.appendChild(titleBar);
    wrapper.appendChild(body);
    document.body.appendChild(wrapper);

    this.render();
  }
}
