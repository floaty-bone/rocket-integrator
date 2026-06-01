import { useEffect, useRef } from 'react';

// ── Shaders ──────────────────────────────────────────────────────────────────

// Lines are rendered as screen-space quads so we control line width + AA
const LINE_VERT = `
attribute vec2 a_ndc;
attribute float a_perp;
attribute vec4 a_color;
varying float v_perp;
varying vec4 v_color;
void main() {
  gl_Position = vec4(a_ndc, 0.0, 1.0);
  v_perp = a_perp;
  v_color = a_color;
}
`;

const LINE_FRAG = `
precision mediump float;
varying float v_perp;
varying vec4 v_color;
void main() {
  float aa = 1.0 - smoothstep(0.55, 1.0, abs(v_perp));
  gl_FragColor = vec4(v_color.rgb, v_color.a * aa);
}
`;

const POINT_VERT = `
attribute vec3 a_pos;
uniform mat4 u_mvp;
void main() {
  vec4 clip = u_mvp * vec4(a_pos, 1.0);
  gl_Position = clip;
  gl_PointSize = clamp(8.0 / clip.w, 3.0, 8.0);
}
`;

const POINT_FRAG = `
precision mediump float;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  float r = dot(d, d);
  if (r > 0.25) discard;
  float aa = 1.0 - smoothstep(0.15, 0.25, r);
  gl_FragColor = vec4(1.0, 1.0, 1.0, aa);
}
`;

// ── Math ─────────────────────────────────────────────────────────────────────

type M4 = Float32Array;

function mul(a: M4, b: M4): M4 {
  const o = new Float32Array(16);
  for (let r = 0; r < 4; r++)
    for (let c = 0; c < 4; c++)
      for (let k = 0; k < 4; k++)
        o[r + c * 4] += a[r + k * 4] * b[k + c * 4];
  return o;
}

function perspective(fovY: number, aspect: number, near: number, far: number): M4 {
  const f = 1 / Math.tan(fovY / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect; m[5] = f;
  m[10] = (far + near) / (near - far); m[11] = -1;
  m[14] = (2 * far * near) / (near - far);
  return m;
}

function rotY(a: number): M4 {
  const c = Math.cos(a), s = Math.sin(a);
  return new Float32Array([c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1]);
}

function rotX(a: number): M4 {
  const c = Math.cos(a), s = Math.sin(a);
  return new Float32Array([1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]);
}

function translate(tx: number, ty: number, tz: number): M4 {
  const m = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
  m[12] = tx; m[13] = ty; m[14] = tz;
  return m;
}

// Project a 3D world point through MVP → NDC [x, y, depth_w]
function project(mvp: M4, x: number, y: number, z: number): [number, number, number] {
  const w = mvp[3]*x + mvp[7]*y + mvp[11]*z + mvp[15];
  return [
    (mvp[0]*x + mvp[4]*y + mvp[8]*z  + mvp[12]) / w,
    (mvp[1]*x + mvp[5]*y + mvp[9]*z  + mvp[13]) / w,
    w,
  ];
}


// ── Color palette ─────────────────────────────────────────────────────────────

function hsl(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let r = 0, g = 0, b = 0;
  if      (h < 60)  { r = c; g = x; }
  else if (h < 120) { r = x; g = c; }
  else if (h < 180) { g = c; b = x; }
  else if (h < 240) { g = x; b = c; }
  else if (h < 300) { r = x; b = c; }
  else              { r = c; b = x; }
  return [r + m, g + m, b + m];
}

// Assign each particle a stable vibrant hue based on its 3D angle
// Uses atan2 of x/z to spread hues around the full 360° wheel,
// then biases away from dull yellow-green (60-120°) toward neons.
function particleHues(particles: Float32Array, n: number): Float32Array {
  const hues = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const raw = (Math.atan2(particles[i*3+2], particles[i*3]) / Math.PI + 1) * 180; // 0-360
    // Remap 60-120 (yellow-green) → push toward electric lime / cyan instead
    const h = (raw + i * 137.5) % 360; // golden-angle offset so nearby particles differ
    hues[i] = h;
  }
  return hues;
}

// Blend hues of two endpoints, choose shorter arc around the wheel
function blendHue(a: number, b: number): number {
  let d = b - a;
  if (d >  180) d -= 360;
  if (d < -180) d += 360;
  return (a + d * 0.5 + 360) % 360;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function compile(gl: WebGLRenderingContext, type: number, src: string) {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src);
  gl.compileShader(s);
  return s;
}

function makeProgram(gl: WebGLRenderingContext, vs: string, fs: string) {
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  return p;
}

// ── Particles ─────────────────────────────────────────────────────────────────

const N    = 170;
const CONN = 0.75;
const LINE_W = 2.2; // pixels

function seedParticles(): Float32Array {
  const p = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const u  = Math.random() * 2 - 1;
    const v  = Math.random() * 2 * Math.PI;
    const r  = Math.cbrt(Math.random());
    const sq = Math.sqrt(Math.max(0, 1 - u * u));
    p[i*3]   = r * sq * Math.cos(v) * 2.2; // wide x
    p[i*3+1] = r * sq * Math.sin(v) * 1.4;
    p[i*3+2] = r * u  * 2.0;               // wide z
  }
  return p;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function RippleMesh({ className = '' }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const gl = canvas.getContext('webgl', { antialias: true, alpha: false })!;

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0, 0, 0, 1);

    const lineProg  = makeProgram(gl, LINE_VERT,  LINE_FRAG);
    const pointProg = makeProgram(gl, POINT_VERT, POINT_FRAG);

    const particles = seedParticles();
    const hues      = particleHues(particles, N);

    // Point buffer (static)
    const ptBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, ptBuf);
    gl.bufferData(gl.ARRAY_BUFFER, particles, gl.STATIC_DRAW);

    // Line buffer (dynamic — screen-space quads)
    // Per line: 6 vertices, each vertex: [ndcX, ndcY, perp, r, g, b, a] = 7 floats
    const FLOATS_PER_VERT = 7;
    const MAX_LINES = N * N;
    const lineData = new Float32Array(MAX_LINES * 6 * FLOATS_PER_VERT);
    const lineBuf  = gl.createBuffer()!;

    const ptMVP = gl.getUniformLocation(pointProg, 'u_mvp');

    let animId: number;
    let lastT = performance.now() / 1000;
    let ry = 0, rx = 0;

    const resize = () => {
      canvas.width  = canvas.offsetWidth  * devicePixelRatio;
      canvas.height = canvas.offsetHeight * devicePixelRatio;
      gl.viewport(0, 0, canvas.width, canvas.height);
    };
    resize();
    window.addEventListener('resize', resize);

    const draw = () => {
      const now = performance.now() / 1000;
      const dt  = Math.min(now - lastT, 0.05); // cap at 50ms
      lastT = now;
      const W = canvas.width;
      const H = canvas.height;

      ry += 0.18 * dt;
      rx = Math.sin(now * 0.07) * 0.3;

      gl.clear(gl.COLOR_BUFFER_BIT);

      const aspect = W / H;
      const proj  = perspective(0.80, aspect, 0.5, 20);
      const view  = translate(0, 0, -4.0);
      const model = mul(rotY(ry), rotX(rx));
      const mvp   = mul(proj, mul(view, model));

      // ── Pre-project all particles to NDC ──────────────────────────────
      const ndcs = new Float32Array(N * 3); // [ndcX, ndcY, w]
      for (let i = 0; i < N; i++) {
        const px = particles[i*3], py = particles[i*3+1], pz = particles[i*3+2];
        const [nx, ny, w] = project(mvp, px, py, pz);
        ndcs[i*3] = nx; ndcs[i*3+1] = ny; ndcs[i*3+2] = w;
      }

      // ── Build screen-space quad geometry for each line ────────────────
      let vc = 0; // vertex count
      const hw = (LINE_W * 0.5 * devicePixelRatio); // half-width in pixels

      for (let i = 0; i < N; i++) {
        const wi = ndcs[i*3+2];
        if (wi < 0) continue; // behind camera

        const ax = ndcs[i*3],   ay = ndcs[i*3+1];
        const sax = (ax + 1) * 0.5 * W;
        const say = (ay + 1) * 0.5 * H;

        for (let j = i + 1; j < N; j++) {
          const wj = ndcs[j*3+2];
          if (wj < 0) continue;

          const dx = particles[i*3]   - particles[j*3];
          const dy = particles[i*3+1] - particles[j*3+1];
          const dz = particles[i*3+2] - particles[j*3+2];
          const d  = Math.sqrt(dx*dx + dy*dy + dz*dz);
          if (d >= CONN) continue;

          const fade = 1 - d / CONN;
          const h = blendHue(hues[i], hues[j]);
          // Closer connections brighter (l 0.55→0.72), all fully saturated
          const l = 0.55 + fade * 0.17;
          const [r, g, b] = hsl(h, 1.0, l);
          const a = fade * 0.95;

          const bx = ndcs[j*3],   by = ndcs[j*3+1];
          const sbx = (bx + 1) * 0.5 * W;
          const sby = (by + 1) * 0.5 * H;

          // screen direction + perpendicular
          let ldx = sbx - sax, ldy = sby - say;
          const ll = Math.sqrt(ldx*ldx + ldy*ldy);
          if (ll < 0.5) continue;
          ldx /= ll; ldy /= ll;
          const px = -ldy, py = ldx; // perpendicular

          // 4 corners in screen space → NDC
          const corners: [number, number, number][] = [
            [(sax + px*hw) / (W*0.5) - 1, (say + py*hw) / (H*0.5) - 1, -1],
            [(sax - px*hw) / (W*0.5) - 1, (say - py*hw) / (H*0.5) - 1, +1],
            [(sbx + px*hw) / (W*0.5) - 1, (sby + py*hw) / (H*0.5) - 1, -1],
            [(sbx - px*hw) / (W*0.5) - 1, (sby - py*hw) / (H*0.5) - 1, +1],
          ];

          // Two triangles: [0,1,2] and [1,3,2]
          const tris = [0,1,2, 1,3,2];
          for (const ci of tris) {
            const [cx, cy, perp] = corners[ci];
            const base = vc * FLOATS_PER_VERT;
            lineData[base]   = cx;
            lineData[base+1] = cy;
            lineData[base+2] = perp;
            lineData[base+3] = r;
            lineData[base+4] = g;
            lineData[base+5] = b;
            lineData[base+6] = a;
            vc++;
          }

          if (vc + 6 >= MAX_LINES * 6) break;
        }
        if (vc + 6 >= MAX_LINES * 6) break;
      }

      // ── Upload and draw lines ─────────────────────────────────────────
      gl.useProgram(lineProg);
      gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
      gl.bufferData(gl.ARRAY_BUFFER, lineData.subarray(0, vc * FLOATS_PER_VERT), gl.DYNAMIC_DRAW);

      const stride = FLOATS_PER_VERT * 4;
      const ndcLoc   = gl.getAttribLocation(lineProg, 'a_ndc');
      const perpLoc  = gl.getAttribLocation(lineProg, 'a_perp');
      const colorLoc = gl.getAttribLocation(lineProg, 'a_color');

      gl.enableVertexAttribArray(ndcLoc);
      gl.vertexAttribPointer(ndcLoc,   2, gl.FLOAT, false, stride, 0);
      gl.enableVertexAttribArray(perpLoc);
      gl.vertexAttribPointer(perpLoc,  1, gl.FLOAT, false, stride, 2 * 4);
      gl.enableVertexAttribArray(colorLoc);
      gl.vertexAttribPointer(colorLoc, 4, gl.FLOAT, false, stride, 3 * 4);

      gl.drawArrays(gl.TRIANGLES, 0, vc);

      // ── Draw particles ────────────────────────────────────────────────
      gl.useProgram(pointProg);
      gl.uniformMatrix4fv(ptMVP, false, mvp);

      gl.bindBuffer(gl.ARRAY_BUFFER, ptBuf);
      const posLoc = gl.getAttribLocation(pointProg, 'a_pos');
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

      gl.drawArrays(gl.POINTS, 0, N);

      animId = requestAnimationFrame(draw);
    };
    draw();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ display: 'block', width: '100%', height: '100%' }}
    />
  );
}
