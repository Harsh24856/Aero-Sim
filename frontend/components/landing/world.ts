/* =====================================================================
   AERO-SIM landing world — a dusk airfield, flown by the scroll.

   Everything here is generated at runtime except the blackout airframe and
   the four engine models already shipped in /public/models. The page owns
   the scroll (it hands us a chapter progress 0…5); this module owns the
   renderer, the camera rig, the live views cut into the page, and teardown.
   ===================================================================== */
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

export type WorldOptions = {
  canvas: HTMLCanvasElement;
  /** Searched for [data-view] frames: "chase" or "engine" (with data-model). */
  root: HTMLElement;
  /** Chapter progress, 0 at the hero to 5 at the footer. */
  getProgress: () => number;
  onStep: (fraction: number, label: string) => void;
  onLost: () => void;
  reduce: boolean;
  coarse: boolean;
};

export type World = {
  start(): void;
  setFocus(on: boolean): void;
  dispose(): void;
};

/* ------------------------------------------------------------ basics */
const TAU = Math.PI * 2;
const clamp = (v: number, a: number, b: number) => (v < a ? a : v > b ? b : v);
const sat = (v: number) => clamp(v, 0, 1);
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const smooth = (e0: number, e1: number, x: number) => {
  const t = sat((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
};
/* frame-rate independent damping */
const damp = (cur: number, to: number, rate: number, dt: number) => lerp(cur, to, 1 - Math.exp(-rate * dt));
const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);

function mulberry32(seed: number) {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type Noise = (x: number, y: number) => number;
function noise2D(seed: number): Noise {
  const rnd = mulberry32(seed);
  const p = new Uint8Array(256);
  const perm = new Uint8Array(512);
  for (let i = 0; i < 256; i++) p[i] = i;
  for (let i = 255; i > 0; i--) {
    const j = (rnd() * (i + 1)) | 0;
    const t = p[i]; p[i] = p[j]; p[j] = t;
  }
  for (let i = 0; i < 512; i++) perm[i] = p[i & 255];
  const G = [[1, 1], [-1, 1], [1, -1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]];
  const fade = (t: number) => t * t * t * (t * (t * 6 - 15) + 10);
  const g = (h: number, dx: number, dy: number) => { const q = G[h & 7]; return q[0] * dx + q[1] * dy; };
  return (x, y) => {
    const xi = Math.floor(x), yi = Math.floor(y);
    const X = xi & 255, Y = yi & 255, xf = x - xi, yf = y - yi;
    const u = fade(xf), v = fade(yf);
    const aa = perm[perm[X] + Y], ab = perm[perm[X] + Y + 1];
    const ba = perm[perm[X + 1] + Y], bb = perm[perm[X + 1] + Y + 1];
    return lerp(lerp(g(aa, xf, yf), g(ba, xf - 1, yf), u), lerp(g(ab, xf, yf - 1), g(bb, xf - 1, yf - 1), u), v);
  };
}
function fbm(n: Noise, x: number, y: number, oct: number) {
  let a = 0.5, f = 1, s = 0, m = 0;
  for (let i = 0; i < oct; i++) { s += a * n(x * f, y * f); m += a; a *= 0.5; f *= 2; }
  return s / m; /* −1 … 1 */
}

/* ------------------------------------------------------------ the palette
   The site's own tokens (globals.css), converted to linear once here so
   every shader grades against the same orange the UI uses. */
const TERTIARY = new THREE.Color("#FF9100");
const EMBER = new THREE.Color("#ff6a22");
const FOG_DARK = new THREE.Color("#0d0a08");
const FOG_SUN = new THREE.Color("#4a220b");
const SKY_ZENITH = new THREE.Color("#050505");
const SKY_MID = new THREE.Color("#140d09");
const SKY_HORIZON = new THREE.Color("#3d1d0b");
const SKY_LOW = new THREE.Color("#0a0a0a");
const hdr = (c: THREE.Color, k: number) => c.clone().multiplyScalar(k);

/* the low sun the whole field is lit and graded from */
const SUN_DIR = new THREE.Vector3(0.22, 0.13, -1).normalize();
const FOG_DENSITY = 0.00082;

/* ------------------------------------------------------------ site plan
   Runway on the z axis: threshold at z = 0, far end at −600, the approach
   lights reaching back toward the hero camera. Everything downstream —
   waypoints, light rows, the wordmark — is pinned to these. */
const RUNWAY_LEN = 600;
const RUNWAY_W = 46;
const WORD_Z = 90;

const n1 = noise2D(11), n2 = noise2D(29), n3 = noise2D(47);
function heightAt(x: number, z: number) {
  const ax = Math.abs(x);
  const side = smooth(70, 380, ax);          /* valley walls */
  const far = smooth(-760, -1350, z);        /* the range the sun sets behind */
  const behind = smooth(260, 720, z);        /* hills behind the hero camera */
  const amp = Math.max(side * 125, far * 225, behind * 110);
  const base = fbm(n1, x * 0.0032, z * 0.0032, 5) * 0.5 + 0.5;
  const r = 1 - Math.abs(fbm(n2, x * 0.0021 + 3.1, z * 0.0021, 4));
  let h = amp * (0.5 * base + 0.5 * r * r);
  /* soft swells on the valley floor, kept clear of the strip itself */
  h += smooth(40, 130, ax) * (fbm(n3, x * 0.01, z * 0.01, 3) * 0.5 + 0.5) * 6;
  return h;
}

/* ------------------------------------------------------------ the camera rig
   One waypoint per chapter: pre-flight, twin, fleet, watch, launch, footer. */
type Way = { p: [number, number, number]; t: [number, number, number]; fov: number };
const CAM: Way[] = [
  { p: [0, 10, 128], t: [0, 6, -300], fov: 40 },        /* 0 hero: down the runway at the sun */
  { p: [-78, 36, 36], t: [24, 28, -170], fov: 46 },     /* 1 twin: across the field */
  { p: [66, 74, -90], t: [-30, 10, -330], fov: 44 },    /* 2 fleet */
  { p: [14, 230, -210], t: [0, 0, -400], fov: 52 },     /* 3 watch: the contour map from above */
  { p: [0, 6, -560], t: [0, 16, -1200], fov: 42 },      /* 4 launch: runway end, into the sun */
  { p: [0, 44, -640], t: [0, 110, -1400], fov: 46 },    /* 5 footer: up into the sky */
];

/* ------------------------------------------------------------ shared GLSL */
const LIGHT_POINT_VS = /* glsl */ `
attribute vec3 aColor; attribute float aSize; attribute float aMode; attribute float aParam;
uniform float uT; uniform float uPx; uniform float uBoost;
varying vec3 vC;
void main(){
  float a = 1.0;
  if (aMode > 0.5 && aMode < 1.5) {            /* rabbit: a flash racing to the threshold */
    float dd = fract(uT * 0.9) - aParam;
    a = 0.12 + 3.2 * exp(-dd * dd * 1400.0);
  } else if (aMode > 1.5 && aMode < 2.5) {     /* obstruction: slow blink */
    a = 0.08 + 1.6 * step(0.55, fract(uT * 0.55 + aParam));
  } else if (aMode > 2.5 && aMode < 3.5) {     /* distant town: twinkle */
    a = 0.55 + 0.45 * sin(uT * (1.3 + aParam * 2.0) + aParam * 40.0);
  } else if (aMode > 3.5) {                    /* airframe strobe: double flash */
    float p = fract(uT * 0.8 + aParam);
    a = 4.0 * (step(p, 0.035) + step(abs(p - 0.11), 0.03));
  }
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  float dist = -mv.z;
  vC = aColor * a * (1.0 + uBoost) * mix(1.0, 0.35, smoothstep(300.0, 2200.0, dist));
  gl_PointSize = clamp(uPx * aSize / max(dist, 1.0), 1.6, 90.0);
  gl_Position = projectionMatrix * mv;
}`;
const LIGHT_POINT_FS = /* glsl */ `
varying vec3 vC;
void main(){
  float d = length(gl_PointCoord - 0.5) * 2.0;
  float g = exp(-d * d * 28.0) + 0.28 * exp(-d * d * 5.0);
  gl_FragColor = vec4(vC * g, g);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

type LightSpec = { p: THREE.Vector3; c: THREE.Color; s: number; mode: number; param: number };
function lightPoints(list: LightSpec[], uniforms: Record<string, THREE.IUniform>) {
  const n = list.length;
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
  const size = new Float32Array(n), mode = new Float32Array(n), param = new Float32Array(n);
  list.forEach((l, i) => {
    pos.set([l.p.x, l.p.y, l.p.z], i * 3);
    col.set([l.c.r, l.c.g, l.c.b], i * 3);
    size[i] = l.s; mode[i] = l.mode; param[i] = l.param;
  });
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("aColor", new THREE.BufferAttribute(col, 3));
  g.setAttribute("aSize", new THREE.BufferAttribute(size, 1));
  g.setAttribute("aMode", new THREE.BufferAttribute(mode, 1));
  g.setAttribute("aParam", new THREE.BufferAttribute(param, 1));
  const pts = new THREE.Points(g, new THREE.ShaderMaterial({
    uniforms, vertexShader: LIGHT_POINT_VS, fragmentShader: LIGHT_POINT_FS,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  }));
  pts.frustumCulled = false;
  return pts;
}

/* ================================================================= world */
export async function createWorld(o: WorldOptions): Promise<World> {
  const { canvas, root, getProgress, onStep, onLost, reduce, coarse } = o;
  const vpW = () => document.documentElement.clientWidth || innerWidth;
  const vpH = () => document.documentElement.clientHeight || innerHeight;
  const DPR_CAP = coarse ? 1.35 : 1.8;
  const PERF = { scale: 1, acc: 0, n: 0 };

  const disposables: { dispose(): void }[] = [];
  const track = <T extends { dispose(): void }>(x: T) => { disposables.push(x); return x; };
  const cleanups: (() => void)[] = [];
  const on = <K extends keyof WindowEventMap>(t: K, f: (e: WindowEventMap[K]) => void) => {
    addEventListener(t, f as EventListener, { passive: true });
    cleanups.push(() => removeEventListener(t, f as EventListener));
  };
  let dead = false;
  const tick = () => new Promise<void>((r) => setTimeout(r, 16));

  /* ------------------------------------------------ 1 · renderer */
  onStep(0.04, "Warming the renderer");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: !coarse, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, DPR_CAP));
  renderer.setSize(vpW(), vpH(), true);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.setClearColor(0x0a0a0a, 1);
  const onContextLost = (e: Event) => { e.preventDefault(); running = false; onLost(); };
  canvas.addEventListener("webglcontextlost", onContextLost);
  cleanups.push(() => canvas.removeEventListener("webglcontextlost", onContextLost));

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(FOG_DARK.getHex(), FOG_DENSITY);
  const camera = new THREE.PerspectiveCamera(CAM[0].fov, vpW() / vpH(), 1, 6000);
  camera.layers.enable(1); camera.layers.enable(2);
  scene.add(camera);

  const pmrem = track(new THREE.PMREMGenerator(renderer));
  const envTex = track(pmrem.fromScene(new RoomEnvironment(), 0.04).texture);
  scene.environment = envTex;
  scene.environmentIntensity = 0.3;

  const uT = { value: 0 };
  const uPx = { value: vpH() * renderer.getPixelRatio() * 0.5 };
  const uBoost = { value: 0 };
  await tick();

  /* ------------------------------------------------ 2 · terrain */
  onStep(0.14, "Surveying the valley");
  {
    const SEG = coarse ? 220 : 320;
    const geo = new THREE.PlaneGeometry(3400, 3400, SEG, SEG);
    geo.rotateX(-Math.PI / 2);
    geo.translate(0, 0, -520);
    const p = geo.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < p.count; i++) p.setY(i, heightAt(p.getX(i), p.getZ(i)));
    geo.computeVertexNormals();
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uT, uBoost,
        uSun: { value: SUN_DIR }, uSunCol: { value: TERTIARY },
        uFogA: { value: FOG_DARK }, uFogB: { value: FOG_SUN }, uFogD: { value: FOG_DENSITY },
      },
      vertexShader: /* glsl */ `
        varying vec3 vW; varying vec3 vN;
        void main(){
          vec4 w = modelMatrix * vec4(position, 1.0);
          vW = w.xyz; vN = normalize(mat3(modelMatrix) * normal);
          gl_Position = projectionMatrix * viewMatrix * w;
        }`,
      /* Topographic contours rather than a texture: the valley reads as a
         survey, and a scan ring sweeps it like the twin is listening. */
      fragmentShader: /* glsl */ `
        uniform float uT, uBoost, uFogD;
        uniform vec3 uSun, uSunCol, uFogA, uFogB;
        varying vec3 vW; varying vec3 vN;
        float lineAt(float v){ float f = fract(v); float w = fwidth(v); return (1.0 - smoothstep(0.0, w * 1.3, min(f, 1.0 - f))) * (1.0 - smoothstep(0.25, 0.7, w)); }
        void main(){
          vec3 n = normalize(vN);
          vec3 col = vec3(0.012, 0.010, 0.009);
          col += uSunCol * pow(max(dot(n, uSun), 0.0), 2.0) * 0.07;
          col += vec3(0.010, 0.009, 0.008) * max(n.y, 0.0);
          vec3 v = vW - cameraPosition; float dist = length(v); vec3 vd = v / dist;
          float lift = smoothstep(0.6, 3.0, vW.y);
          float far = 1.0 - smoothstep(700.0, 1700.0, dist);
          float r = distance(vW.xz, vec2(0.0, -260.0));
          float scan = exp(-pow((r - mod(uT * 150.0, 1700.0)) / 22.0, 2.0));
          float minor = lineAt(vW.y / 6.0), major = lineAt(vW.y / 30.0);
          col += uSunCol * (minor * 0.06 + major * 0.16) * lift * far * (1.0 + scan * 7.0 + uBoost);
          vec2 gv = vW.xz / 40.0; vec2 gd = abs(fract(gv - 0.5) - 0.5) / fwidth(gv);
          float grid = (1.0 - min(min(gd.x, gd.y), 1.0)) * (1.0 - smoothstep(0.2, 0.6, max(fwidth(gv).x, fwidth(gv).y)));
          col += uSunCol * grid * 0.02 * (1.0 - lift) * far * (1.0 + scan * 9.0);
          float sunAmt = pow(max(dot(vd, uSun), 0.0), 6.0);
          vec3 fogC = mix(uFogA, uFogB, sunAmt);
          float fog = 1.0 - exp(-uFogD * uFogD * dist * dist);
          gl_FragColor = vec4(mix(col, fogC, fog), 1.0);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    });
    const terrain = new THREE.Mesh(geo, mat);
    scene.add(terrain);
  }
  await tick();

  /* ------------------------------------------------ 3 · sky */
  onStep(0.24, "Painting the dusk");
  const sky = new THREE.Mesh(
    new THREE.SphereGeometry(3000, 64, 32),
    new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false,
      uniforms: {
        uT, uBoost, uSun: { value: SUN_DIR }, uSunCol: { value: TERTIARY },
        uZen: { value: SKY_ZENITH }, uMid: { value: SKY_MID }, uHor: { value: SKY_HORIZON }, uLow: { value: SKY_LOW },
      },
      vertexShader: /* glsl */ `
        varying vec3 vDir;
        void main(){ vDir = position; vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0); gl_Position = p.xyww; }`,
      fragmentShader: /* glsl */ `
        uniform float uT, uBoost; uniform vec3 uSun, uSunCol, uZen, uMid, uHor, uLow;
        varying vec3 vDir;
        float hash(vec3 p){ p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
        void main(){
          vec3 d = normalize(vDir); float e = d.y;
          vec3 col = mix(uHor, uMid, smoothstep(0.0, 0.16, e));
          col = mix(col, uZen, smoothstep(0.14, 0.7, e));
          col = mix(col, uLow, smoothstep(0.0, -0.12, e));
          float s = max(dot(d, uSun), 0.0);
          float band = exp(-abs(e - 0.02) * 10.0);
          col += uSunCol * (0.06 * band * (0.3 + 0.7 * pow(s, 3.0)) + 0.2 * pow(s, 12.0) + 0.6 * pow(s, 120.0)) * (1.0 + uBoost * 0.5);
          col += uSunCol * smoothstep(0.99955, 0.99972, s) * 9.0;
          vec3 c = fract(d * 380.0) - 0.5; float h = hash(floor(d * 380.0));
          float star = step(0.9972, h) * smoothstep(0.35, 0.0, length(c)) * smoothstep(0.05, 0.32, e)
                     * (0.55 + 0.45 * sin(uT * 1.7 + h * 80.0)) * (1.0 - pow(s, 3.0));
          col += vec3(1.0, 0.93, 0.85) * star * 1.3;
          gl_FragColor = vec4(col, 1.0);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    }),
  );
  sky.frustumCulled = false; sky.renderOrder = -1;
  scene.add(sky);

  const sunLight = new THREE.DirectionalLight(0xffa04a, 1.6);
  sunLight.position.copy(SUN_DIR).multiplyScalar(1000);
  scene.add(sunLight, new THREE.HemisphereLight(0x3a2418, 0x050505, 0.5));
  await tick();

  /* ------------------------------------------------ 4 · the airfield */
  onStep(0.36, "Lighting the runway");
  const lights: LightSpec[] = [];
  const L = (x: number, y: number, z: number, c: THREE.Color, s: number, mode = 0, param = 0) =>
    lights.push({ p: new THREE.Vector3(x, y, z), c, s, mode, param });
  {
    /* the strip, with its markings drawn rather than loaded */
    const cv = document.createElement("canvas");
    cv.width = 256; cv.height = 4096;
    const x = cv.getContext("2d")!;
    x.scale(2, 2);
    x.fillStyle = "#101010"; x.fillRect(0, 0, 128, 2048);
    const rnd = mulberry32(5);
    for (let i = 0; i < 900; i++) { x.fillStyle = `rgba(255,255,255,${rnd() * 0.035})`; x.fillRect(rnd() * 128, rnd() * 2048, 1 + rnd() * 3, 1 + rnd() * 6); }
    x.fillStyle = "rgba(235,230,220,.8)";
    for (const y0 of [8, 2048 - 44]) for (let k = 0; k < 8; k++) x.fillRect(10 + k * 14.5 + (k > 3 ? 8 : 0), y0, 6, 36);
    x.font = "700 30px 'JetBrains Mono', monospace"; x.textAlign = "center";
    x.save(); x.translate(64, 90); x.fillText("09", 0, 0); x.restore();
    x.save(); x.translate(64, 2048 - 70); x.rotate(Math.PI); x.fillText("27", 0, 0); x.restore();
    for (let y = 120; y < 2048 - 120; y += 44) x.fillRect(62, y, 4, 22);
    x.fillStyle = "rgba(255,145,0,.55)";
    x.fillRect(4, 0, 2, 2048); x.fillRect(122, 0, 2, 2048);
    const tex = track(new THREE.CanvasTexture(cv));
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
    const strip = new THREE.Mesh(
      new THREE.PlaneGeometry(RUNWAY_W, RUNWAY_LEN).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ map: tex, color: 0x6b6560, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 }),
    );
    strip.position.set(0, 0.08, -RUNWAY_LEN / 2);
    scene.add(strip);

    const warm = hdr(new THREE.Color("#ffd9a8"), 2.2);
    const white = hdr(new THREE.Color("#fff4e6"), 1.0);
    const red = hdr(new THREE.Color("#e63721"), 2.2);
    const green = hdr(new THREE.Color("#5cb266"), 1.8);
    const amber = hdr(TERTIARY, 2.2);
    for (let z = 0; z >= -RUNWAY_LEN; z -= 30) {
      const c = z < -RUNWAY_LEN + 180 ? amber : warm;          /* caution zone */
      L(-RUNWAY_W / 2 - 1, 0.6, z, c, 1.6); L(RUNWAY_W / 2 + 1, 0.6, z, c, 1.6);
    }
    for (let z = -10; z > -RUNWAY_LEN; z -= 15) L(0, 0.2, z, white, 0.55);
    for (let x0 = -RUNWAY_W / 2; x0 <= RUNWAY_W / 2; x0 += 4) { L(x0, 0.4, 1.5, green, 1.1); L(x0, 0.4, -RUNWAY_LEN - 1, red, 1.1); }
    /* approach: steady bars, the rabbit running through them, one crossbar */
    for (let z = 12, k = 0; z <= 142; z += 10, k++) {
      for (let x0 = -4; x0 <= 4; x0 += 2) L(x0, 1.2, z, warm, 0.9);
      L(0, 2.2, z, hdr(new THREE.Color("#ffffff"), 2.4), 1.4, 1, 1 - k / 13);
    }
    for (let x0 = -16; x0 <= 16; x0 += 2) if (Math.abs(x0) > 5) L(x0, 1.2, 72, warm, 0.9);
    /* PAPI: two white over two red — on the glide path */
    [-34, -38, -42, -46].forEach((x0, i) => L(x0, 1, -46, i < 2 ? white : red, 1.1));

    /* tower and hangars, dark against the dusk, one warm line of light each */
    const dark = new THREE.MeshStandardMaterial({ color: 0x0d0c0b, roughness: 0.9, metalness: 0.1 });
    const lit = new THREE.MeshBasicMaterial({ color: hdr(new THREE.Color("#ffb266"), 1.6) });
    const box = (w: number, h: number, d: number, px: number, py: number, pz: number, m: THREE.Material = dark) => {
      const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m); b.position.set(px, py, pz); scene.add(b); return b;
    };
    box(6, 24, 6, -96, 12, -250);
    box(11, 5, 11, -96, 26.5, -250);
    box(11.2, 1.6, 11.2, -96, 26.6, -250, lit);
    box(12, 0.6, 12, -96, 29.3, -250);
    L(-96, 31, -250, red, 1.6, 2, 0.3);
    for (const hz of [-390, -436, -482]) {
      box(42, 13, 34, -116, 6.5, hz);
      const door = new THREE.Mesh(new THREE.PlaneGeometry(30, 0.7), lit);
      door.rotation.y = Math.PI / 2; door.position.set(-94.9, 1.2, hz);
      scene.add(door);
    }

    /* obstruction lights on the ridges, and a scatter of distant town */
    const rnd2 = mulberry32(91);
    for (let i = 0; i < 16; i++) {
      const x0 = (rnd2() - 0.5) * 2200, z0 = -700 - rnd2() * 900;
      L(x0, heightAt(x0, z0) + 6, z0, red, 3.2, 2, rnd2());
    }
    for (let i = 0, placed = 0; i < 2000 && placed < 320; i++) {
      const x0 = (rnd2() - 0.5) * 1900, z0 = -760 - rnd2() * 800;
      const h = heightAt(x0, z0);
      if (h > 70) continue;
      L(x0, h + 1.5, z0, hdr(rnd2() > 0.3 ? new THREE.Color("#ffb266") : TERTIARY, 0.9 + rnd2()), 1.4, 3, rnd2());
      placed++;
    }
  }
  const fieldLights = lightPoints(lights, { uT, uPx, uBoost });
  scene.add(fieldLights);
  await tick();

  /* ------------------------------------------------ 5 · the airframe */
  onStep(0.48, "Fueling the airframe");
  const loader = new GLTFLoader();
  const planeRig = new THREE.Group();
  const planeBody = new THREE.Group();
  planeRig.add(planeBody);
  scene.add(planeRig);
  const tailLocal = new THREE.Vector3(0, 0, -8);
  try {
    const gltf = await loader.loadAsync("/models/blackout-plane.glb");
    const model = gltf.scene;
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3());
    const k = 16 / Math.max(size.x, size.y, size.z);
    model.scale.setScalar(k);
    model.position.copy(box.getCenter(new THREE.Vector3()).multiplyScalar(-k));
    model.traverse((ob) => {
      const mesh = ob as THREE.Mesh;
      if (!mesh.isMesh) return;
      (Array.isArray(mesh.material) ? mesh.material : [mesh.material]).forEach((mat) => {
        if (mat instanceof THREE.MeshStandardMaterial) { mat.envMap = envTex; mat.envMapIntensity = 1.3; }
      });
    });
    planeBody.add(model);
    const hs = size.clone().multiplyScalar(k * 0.5);
    tailLocal.set(0, hs.y * 0.2, -hs.z);
    const nav: LightSpec[] = [
      { p: new THREE.Vector3(hs.x * 0.97, 0, 0), c: hdr(new THREE.Color("#e63721"), 3), s: 1.1, mode: 0, param: 0 },
      { p: new THREE.Vector3(-hs.x * 0.97, 0, 0), c: hdr(new THREE.Color("#5cb266"), 3), s: 1.1, mode: 0, param: 0 },
      { p: new THREE.Vector3(0, hs.y * 0.6, -hs.z * 0.96), c: hdr(new THREE.Color("#ffffff"), 1.4), s: 1.6, mode: 4, param: 0 },
      { p: new THREE.Vector3(hs.x * 0.97, 0, 0), c: hdr(new THREE.Color("#ffffff"), 1.2), s: 1.3, mode: 4, param: 0.5 },
      { p: new THREE.Vector3(-hs.x * 0.97, 0, 0), c: hdr(new THREE.Color("#ffffff"), 1.2), s: 1.3, mode: 4, param: 0.5 },
    ];
    planeBody.add(lightPoints(nav, { uT, uPx, uBoost: { value: 0 } }));
  } catch (err) {
    console.error("[aero-sim] airframe failed to load", err);
  }
  /* the circuit it flies: over the strip toward the sun, round the right
     side of the valley and back past the hero camera */
  const circuit = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, 46, 60), new THREE.Vector3(0, 58, -330), new THREE.Vector3(70, 80, -640),
    new THREE.Vector3(250, 98, -620), new THREE.Vector3(330, 92, -300), new THREE.Vector3(280, 78, 70),
    new THREE.Vector3(150, 60, 230), new THREE.Vector3(40, 50, 170),
  ], true, "catmullrom", 0.5);
  const CIRCUIT_LEN = circuit.getLength();
  const PLANE = { u: 0.02, roll: 0, prevYaw: 0 };

  /* exhaust motes, shed behind the tail */
  const TRAIL_N = 120;
  const trailPos = new Float32Array(TRAIL_N * 3), trailAge = new Float32Array(TRAIL_N).fill(9);
  const trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute("position", new THREE.BufferAttribute(trailPos, 3));
  trailGeo.setAttribute("aAge", new THREE.BufferAttribute(trailAge, 1));
  const trail = new THREE.Points(trailGeo, new THREE.ShaderMaterial({
    uniforms: { uPx, uCol: { value: hdr(EMBER, 1.6) } },
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */ `
      attribute float aAge; uniform float uPx; varying float vA;
      void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vA = (1.0 - smoothstep(0.0, 2.2, aAge)) * smoothstep(0.0, 0.1, aAge);
        gl_PointSize = uPx * (0.14 + aAge * 0.4) / max(-mv.z, 1.0); gl_Position = projectionMatrix * mv; }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uCol; varying float vA;
      void main(){ float d = length(gl_PointCoord - 0.5) * 2.0; float g = exp(-d * d * 6.0) * vA;
        gl_FragColor = vec4(uCol * g * 0.3, g * 0.6);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
  }));
  trail.frustumCulled = false;
  scene.add(trail);
  let trailI = 0;
  await tick();

  /* ------------------------------------------------ 6 · the wordmark */
  onStep(0.62, "Casting the wordmark");
  const WORD: { glyphs: THREE.Mesh[]; group: THREE.Group | null; ink: { cx: number; w: number; asc: number } | null; reveal: number } =
    { glyphs: [], group: null, ink: null, reveal: 0 };
  try { await document.fonts.load("700 320px 'Space Grotesk'"); } catch { /* the fallback face still sets */ }
  {
    const SZ = 320, TRACK = 0.12, PAD = 24;
    const FONT = "700 " + SZ + "px 'Space Grotesk', sans-serif";
    const m = document.createElement("canvas").getContext("2d")!;
    m.font = FONT;
    const word = "AERO-SIM";
    let pen = 0, ascMax = 0, descMax = 0, xMin = 1e9, xMax = -1e9;
    const gl = [...word].map((ch) => {
      const t = m.measureText(ch);
      const g = { ch, asc: t.actualBoundingBoxAscent, desc: t.actualBoundingBoxDescent, l: t.actualBoundingBoxLeft, r: t.actualBoundingBoxRight, pen };
      ascMax = Math.max(ascMax, g.asc); descMax = Math.max(descMax, g.desc);
      xMin = Math.min(xMin, pen - g.l); xMax = Math.max(xMax, pen + g.r);
      pen += t.width + TRACK * SZ;
      return g;
    });
    const group = new THREE.Group();
    gl.forEach((g) => {
      const cw = Math.ceil(g.l + g.r) + PAD * 2, chh = Math.ceil(g.asc + g.desc) + PAD * 2;
      const c = document.createElement("canvas"); c.width = cw; c.height = chh;
      const x = c.getContext("2d")!;
      x.font = FONT;
      /* one vertical ramp for the whole word, fading into the ground */
      const y0 = PAD + g.asc - ascMax, y1 = PAD + g.asc + descMax;
      const grad = x.createLinearGradient(0, y0, 0, y1);
      if (g.ch === "-") { grad.addColorStop(0, "rgb(255,145,0)"); grad.addColorStop(1, "rgba(255,106,34,.5)"); }
      else {
        grad.addColorStop(0, "rgba(236,230,222,1)");
        grad.addColorStop(0.55, "rgba(196,186,176,.92)");
        grad.addColorStop(1, "rgba(150,138,128,.16)");
      }
      x.fillStyle = grad;
      x.fillText(g.ch, PAD + g.l, PAD + g.asc);
      const tex = track(new THREE.CanvasTexture(c));
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(cw, chh),
        new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, fog: false, color: 0xd8d0c8 }));
      mesh.position.set(g.pen + (g.r - g.l) / 2, (g.asc - g.desc) / 2, 0);
      mesh.userData.baseY = mesh.position.y;
      mesh.renderOrder = 12;
      mesh.frustumCulled = false;
      mesh.layers.set(2);
      group.add(mesh);
      WORD.glyphs.push(mesh);
    });
    group.position.z = WORD_Z;
    scene.add(group);
    WORD.group = group;
    WORD.ink = { cx: (xMin + xMax) / 2, w: xMax - xMin, asc: ascMax };
  }
  await tick();

  /* ------------------------------------------------ 7 · sparks + wisps */
  onStep(0.74, "Raising the sparks");
  {
    const N = coarse ? 260 : 620;
    const rnd = mulberry32(66);
    const pos = new Float32Array(N * 3), seed = new Float32Array(N);
    for (let i = 0; i < N; i++) { pos.set([(rnd() - 0.5) * 140, rnd() * 45, (rnd() - 0.5) * 140], i * 3); seed[i] = rnd(); }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("aSeed", new THREE.BufferAttribute(seed, 1));
    /* x and z wrap around whichever camera is drawing, y stays world-true,
       so the sparks hug the ground wherever the walk goes */
    const embers = new THREE.Points(g, new THREE.ShaderMaterial({
      uniforms: { uT, uPx, uCol: { value: hdr(EMBER, 2.4) } },
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      vertexShader: /* glsl */ `
        attribute float aSeed; uniform float uT, uPx; varying float vA;
        void main(){
          vec3 p = position;
          p.y = mod(p.y + uT * (0.8 + aSeed * 1.8), 45.0);
          p.x += sin(uT * 0.3 + aSeed * 20.0) * 3.0;
          p.z += cos(uT * 0.25 + aSeed * 14.0) * 3.0;
          p.x = cameraPosition.x + mod(p.x - cameraPosition.x + 70.0, 140.0) - 70.0;
          p.z = cameraPosition.z + mod(p.z - cameraPosition.z + 70.0, 140.0) - 70.0;
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          vA = (0.25 + 0.75 * aSeed) * smoothstep(45.0, 28.0, p.y) * smoothstep(0.0, 2.0, p.y);
          gl_PointSize = uPx * (0.07 + aSeed * 0.12) / max(-mv.z, 1.0);
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 uCol; varying float vA;
        void main(){ float d = length(gl_PointCoord - 0.5) * 2.0; float g = exp(-d * d * 7.0) * vA;
          gl_FragColor = vec4(uCol * g, g);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    }));
    embers.frustumCulled = false;
    scene.add(embers);
  }

  /* cursor wisps: hung off the camera so they stay under the hand while the
     rig drifts, emitted by distance travelled rather than by time */
  const WISP_D = 3.4;
  const WISP = { list: [] as { life: number; max: number; vx: number; vy: number; sz: number; ph: number }[], i: 0, acc: 0, ex: 0, ey: 0, lx: 0, ly: 0, idle: 0, seen: false, mesh: null as THREE.Points | null };
  if (!coarse && !reduce) {
    const N = 170;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(N * 3), 3));
    g.setAttribute("aA", new THREE.BufferAttribute(new Float32Array(N), 1));
    g.setAttribute("aS", new THREE.BufferAttribute(new Float32Array(N), 1));
    const pts = new THREE.Points(g, new THREE.ShaderMaterial({
      uniforms: { uPx: { value: vpH() * renderer.getPixelRatio() }, uCol: { value: hdr(new THREE.Color("#ffc58a"), 1.3) } },
      transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
      vertexShader: /* glsl */ `
        attribute float aA; attribute float aS; uniform float uPx; varying float vA;
        void main(){ vA = aA; vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = uPx * aS / max(-mv.z, 0.4); gl_Position = projectionMatrix * mv; }`,
      fragmentShader: /* glsl */ `
        uniform vec3 uCol; varying float vA;
        void main(){ if (vA <= 0.0) discard; float d = length(gl_PointCoord - 0.5) * 2.0;
          float g = exp(-d * d * 60.0) + 0.22 * exp(-d * d * 5.0);
          gl_FragColor = vec4(uCol * g * vA, g * vA);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    }));
    pts.frustumCulled = false; pts.renderOrder = 9;
    pts.layers.set(1);
    camera.add(pts);
    WISP.mesh = pts;
    for (let i = 0; i < N; i++) WISP.list.push({ life: 0, max: 1, vx: 0, vy: 0, sz: 0, ph: 0 });
  }
  await tick();

  /* ------------------------------------------------ 8 · live views + post */
  onStep(0.86, "Calibrating the chase cam");
  const composer = track(new EffectComposer(renderer, new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: coarse ? 0 : 4 })));
  composer.setPixelRatio(renderer.getPixelRatio());
  composer.setSize(vpW(), vpH());
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(vpW(), vpH()), 0.72, 0.55, 0.82);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  type EngineView = {
    kind: "engine"; el: HTMLElement; url: string; scene: THREE.Scene; cam: THREE.PerspectiveCamera;
    pivot: THREE.Group; state: "idle" | "loading" | "ready"; want: number; push: number; tx: number; ty: number;
  };
  type ChaseView = { kind: "chase"; el: HTMLElement; cam: THREE.PerspectiveCamera };
  const views: (EngineView | ChaseView)[] = [];
  const chaseCam = new THREE.PerspectiveCamera(48, 16 / 10, 1, 6000);
  chaseCam.position.set(0, 60, 120);

  const ringMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false,
    uniforms: { uT, uCol: { value: TERTIARY } },
    vertexShader: /* glsl */ `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */ `
      uniform float uT; uniform vec3 uCol; varying vec2 vUv;
      void main(){ float r = length(vUv - 0.5) * 2.0; float v = r * 7.0 - uT * 0.35;
        float f = fract(v); float w = fwidth(v); float line = 1.0 - smoothstep(0.0, w * 1.4, min(f, 1.0 - f));
        float a = line * (1.0 - smoothstep(0.55, 1.0, r)) * 0.55 + (1.0 - smoothstep(0.0, 0.8, r)) * 0.06;
        gl_FragColor = vec4(uCol * a, a);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
  });
  const cardBg = track((() => {
    const c = document.createElement("canvas"); c.width = c.height = 256;
    const x = c.getContext("2d")!;
    const g = x.createRadialGradient(128, 170, 8, 128, 150, 190);
    g.addColorStop(0, "#3a1d0c"); g.addColorStop(0.45, "#170e09"); g.addColorStop(1, "#0a0908");
    x.fillStyle = g; x.fillRect(0, 0, 256, 256);
    const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
  })());
  root.querySelectorAll<HTMLElement>("[data-view]").forEach((el) => {
    if (el.dataset.view === "chase") { views.push({ kind: "chase", el, cam: chaseCam }); return; }
    const s = new THREE.Scene();
    s.background = cardBg;
    s.environment = envTex;
    s.environmentIntensity = 0.55;
    const key = new THREE.DirectionalLight(0xfff1e0, 2.2); key.position.set(3, 4, 5);
    const rim = new THREE.DirectionalLight(TERTIARY, 3.2); rim.position.set(-4, 2, -5);
    s.add(key, rim, new THREE.HemisphereLight(0x2a1a12, 0x050505, 0.6));
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(7, 7).rotateX(-Math.PI / 2), ringMat);
    floor.position.y = -1.15;
    s.add(floor);
    const pivot = new THREE.Group();
    s.add(pivot);
    const cam = new THREE.PerspectiveCamera(28, 1, 0.1, 100);
    views.push({ kind: "engine", el, url: el.dataset.model || "", scene: s, cam, pivot, state: "idle", want: 0, push: 0, tx: 0, ty: 0 });
    const card = el.closest<HTMLElement>("[data-card]") || el;
    const enter = () => { (views.find((v) => v.el === el) as EngineView).want = 1; };
    const leave = () => { const v = views.find((q) => q.el === el) as EngineView; v.want = 0; v.tx = 0; v.ty = 0; };
    const move = (e: PointerEvent) => {
      const r = el.getBoundingClientRect(); const v = views.find((q) => q.el === el) as EngineView;
      v.tx = ((e.clientX - r.left) / r.width) * 2 - 1; v.ty = ((e.clientY - r.top) / r.height) * 2 - 1;
    };
    card.addEventListener("pointerenter", enter);
    card.addEventListener("pointerleave", leave);
    card.addEventListener("pointermove", move);
    cleanups.push(() => { card.removeEventListener("pointerenter", enter); card.removeEventListener("pointerleave", leave); card.removeEventListener("pointermove", move); });
  });

  /* engines load only when their card is about to be seen — four models
     are ~10 MB, none of it needed for the hero */
  const loadEngine = (v: EngineView) => {
    if (v.state !== "idle" || !v.url) return;
    v.state = "loading";
    v.el.dataset.state = "loading";
    loader.loadAsync(v.url).then((g) => {
      if (dead) return;
      const model = g.scene;
      const box = new THREE.Box3().setFromObject(model);
      const size = box.getSize(new THREE.Vector3());
      const k = 2.3 / Math.max(size.x, size.y, size.z);
      model.scale.setScalar(k);
      model.position.copy(box.getCenter(new THREE.Vector3()).multiplyScalar(-k));
      v.pivot.add(model);
      v.state = "ready";
      v.el.dataset.state = "ready";
    }).catch((err) => { console.error("[aero-sim] engine failed to load", v.url, err); v.el.dataset.state = "error"; });
  };
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return;
    const v = views.find((q) => q.el === e.target);
    if (v && v.kind === "engine") loadEngine(v);
  }), { rootMargin: "900px 0px" });
  views.forEach((v) => v.kind === "engine" && io.observe(v.el));
  cleanups.push(() => io.disconnect());
  await tick();

  /* ================================================= the rig */
  const curveP = new THREE.CatmullRomCurve3(CAM.map((c) => new THREE.Vector3(...c.p)), false, "catmullrom", 0.42);
  const curveT = new THREE.CatmullRomCurve3(CAM.map((c) => new THREE.Vector3(...c.t)), false, "catmullrom", 0.42);
  const RIG = { smooth: 0, mx: 0, my: 0, tmx: 0, tmy: 0, intro: reduce ? 1 : 0, focus: 0, focusAmt: 0 };
  const INTRO = { t0: 0 };
  const _p = new THREE.Vector3(), _t = new THREE.Vector3(), _d = new THREE.Vector3();
  /* every waypoint is composed wide; on a tall frame step back and open up */
  const aspectFix = () => clamp((1.62 - vpW() / vpH()) / 1.05, 0, 1);
  const fitAspect = (p: THREE.Vector3, t: THREE.Vector3, fov: number) => {
    const nf = aspectFix();
    if (nf <= 0) return fov;
    _d.subVectors(p, t).normalize();
    p.addScaledVector(_d, nf * 40); p.y += nf * 6;
    return fov * (1 + nf * 0.35);
  };
  const applyCamera = () => {
    const N = CAM.length - 1;
    const u = clamp(RIG.smooth / N, 0, 1);
    curveP.getPoint(u, _p); curveT.getPoint(u, _t);
    const i = clamp(Math.floor(RIG.smooth), 0, N - 1), f = clamp(RIG.smooth - i, 0, 1);
    let fov = fitAspect(_p, _t, lerp(CAM[i].fov, CAM[i + 1].fov, f));
    const io2 = 1 - RIG.intro;                     /* the opening dolly */
    _p.z += io2 * 34; _p.y += io2 * 5; fov += io2 * 7;
    const par = 1 - smooth(0, 1.6, RIG.smooth) * 0.5;
    _p.x += RIG.mx * 3.2 * par; _p.y += RIG.my * 1.6 * par;
    _t.x -= RIG.mx * 8 * par; _t.y -= RIG.my * 5 * par;
    camera.position.copy(_p);
    camera.lookAt(_t);
    if (Math.abs(camera.fov - fov) > 1e-4) { camera.fov = fov; camera.updateProjectionMatrix(); }
  };

  const tmpCam = new THREE.PerspectiveCamera(40, 1, 1, 6000);
  const layoutWord = () => {
    if (!WORD.group || !WORD.ink) return;
    const c = CAM[0];
    const hp = new THREE.Vector3(...c.p), ht = new THREE.Vector3(...c.t);
    tmpCam.fov = fitAspect(hp, ht, c.fov);
    tmpCam.aspect = vpW() / vpH();
    tmpCam.position.copy(hp); tmpCam.lookAt(ht);
    tmpCam.updateProjectionMatrix(); tmpCam.updateMatrixWorld(true);
    const hit = (nx: number) => {
      const v = new THREE.Vector3(nx, -0.2, 0.5).unproject(tmpCam).sub(tmpCam.position).normalize();
      return tmpCam.position.clone().addScaledVector(v, (WORD_Z - tmpCam.position.z) / v.z);
    };
    const s = ((hit(1).x - hit(-1).x) * (vpW() / vpH() < 1.05 ? 0.94 : 0.84)) / WORD.ink.w;
    WORD.group.scale.setScalar(s);
    /* standing on the ground, so it can rise out of it */
    WORD.group.position.set(-WORD.ink.cx * s, 0.4, WORD_Z);
  };

  /* ================================================= per-frame */
  const _v = new THREE.Vector3(), _a = new THREE.Vector3(), _b = new THREE.Vector3();
  const updatePlane = (dt: number) => {
    PLANE.u = (PLANE.u + (dt * 34) / CIRCUIT_LEN) % 1;
    circuit.getPointAt(PLANE.u, _a);
    circuit.getTangentAt(PLANE.u, _b);
    planeRig.position.copy(_a);
    planeRig.lookAt(_v.copy(_a).add(_b));
    /* bank into the turn: yaw rate, not position, decides the roll */
    const yaw = Math.atan2(_b.x, _b.z);
    let dy = yaw - PLANE.prevYaw;
    if (dy > Math.PI) dy -= TAU; else if (dy < -Math.PI) dy += TAU;
    PLANE.prevYaw = yaw;
    PLANE.roll = damp(PLANE.roll, clamp((dy / Math.max(dt, 1e-3)) * 1.4, -0.7, 0.7), 2.5, dt);
    planeBody.rotation.z = PLANE.roll;
    planeBody.position.y = Math.sin(clock * 0.9) * 0.4;

    /* the exhaust */
    const pos = trailGeo.attributes.position as THREE.BufferAttribute;
    const age = trailGeo.attributes.aAge as THREE.BufferAttribute;
    planeBody.updateMatrixWorld(true);
    for (let k = 0; k < 1; k++) {
      _v.copy(tailLocal); _v.x += (Math.random() - 0.5) * 0.6; _v.y += (Math.random() - 0.5) * 0.6;
      planeBody.localToWorld(_v);
      trailPos.set([_v.x, _v.y, _v.z], trailI * 3);
      trailAge[trailI] = 0;
      trailI = (trailI + 1) % TRAIL_N;
    }
    for (let i = 0; i < TRAIL_N; i++) { trailAge[i] += dt; trailPos[i * 3 + 1] += dt * 0.6; }
    pos.needsUpdate = true; age.needsUpdate = true;

    /* the chase cam rides behind and above, damped so it swings into turns */
    _v.set(0, 3.2, -32); planeBody.localToWorld(_v);
    if (!chaseCam.userData.seeded) { chaseCam.position.copy(_v); chaseCam.userData.seeded = true; }
    chaseCam.position.x = damp(chaseCam.position.x, _v.x, 3, dt);
    chaseCam.position.y = damp(chaseCam.position.y, _v.y, 3, dt);
    chaseCam.position.z = damp(chaseCam.position.z, _v.z, 3, dt);
    _v.set(0, 2.4, 30); planeBody.localToWorld(_v);
    chaseCam.lookAt(_v);
  };

  const updateWisps = (dt: number) => {
    if (!WISP.mesh) return;
    const g = WISP.mesh.geometry;
    const P = g.attributes.position.array as Float32Array, A = g.attributes.aA.array as Float32Array, S = g.attributes.aS.array as Float32Array;
    const Lw = WISP.list, N = Lw.length;
    const hh = Math.tan((camera.fov * Math.PI) / 360) * WISP_D;
    const px = RIG.tmx * hh * camera.aspect, py = RIG.tmy * hh;
    if (!WISP.seen) { WISP.ex = WISP.lx = px; WISP.ey = WISP.ly = py; WISP.seen = true; }
    WISP.ex = damp(WISP.ex, px, 16, dt); WISP.ey = damp(WISP.ey, py, 16, dt);
    const dx = WISP.ex - WISP.lx, dy = WISP.ey - WISP.ly, moved = Math.hypot(dx, dy);
    const ang = moved > 1e-5 ? Math.atan2(dy, dx) : 0;
    const spawn = (x: number, y: number, a: number, weak: boolean) => {
      const i = WISP.i; WISP.i = (i + 1) % N;
      const w = Lw[i], k = i * 3;
      P[k] = x + (Math.random() + Math.random() - 1) * 0.28;
      P[k + 1] = y + (Math.random() + Math.random() - 1) * 0.28;
      P[k + 2] = -WISP_D + (Math.random() - 0.5) * 0.9;
      w.life = 0; w.max = (weak ? 2 : 1.4) + Math.random() * 1.2;
      w.vx = -Math.cos(a) * 0.09 + (Math.random() - 0.5) * 0.36;
      w.vy = -Math.sin(a) * 0.09 + (Math.random() - 0.5) * 0.3 + 0.02;
      w.sz = (weak ? 0.018 : 0.024) + Math.random() * 0.024;
      w.ph = Math.random() * TAU;
    };
    WISP.acc += moved;
    const STEP = 0.03;
    let guard = 0;
    while (WISP.acc >= STEP && guard++ < 14) {
      WISP.acc -= STEP;
      const t = moved > 1e-6 ? Math.min(1, (guard * STEP) / moved) : 0;
      spawn(WISP.lx + dx * t, WISP.ly + dy * t, ang, false);
    }
    WISP.idle += dt;
    if (WISP.idle > 0.42) { WISP.idle = 0; spawn(WISP.ex, WISP.ey, Math.random() * TAU, true); }
    WISP.lx = WISP.ex; WISP.ly = WISP.ey;
    for (let i = 0; i < N; i++) {
      const w = Lw[i], k = i * 3;
      if (w.life >= w.max) { A[i] = 0; continue; }
      w.life += dt;
      const u = w.life / w.max;
      P[k] += (w.vx + Math.sin(clock * 1.3 + w.ph) * 0.17) * dt;
      P[k + 1] += (w.vy + Math.cos(clock * 1.1 + w.ph * 1.7) * 0.14) * dt;
      w.vx *= 1 - 0.5 * dt; w.vy = w.vy * (1 - 0.5 * dt) + 0.022 * dt;
      A[i] = smooth(0, 0.12, u) * (1 - smooth(0.22, 1, u)) * 0.9;
      S[i] = w.sz * (1 + u * 0.55);
    }
    g.attributes.position.needsUpdate = true; g.attributes.aA.needsUpdate = true; g.attributes.aS.needsUpdate = true;
  };

  /* A view lands in the composite at its element's screen rect, and the
     canvas knows nothing of the DOM over it — so a frame the page is fading
     would keep painting a live view with no frame around it. Cull on the
     element's effective opacity as well as its rect. */
  const alphaOf = (el: HTMLElement) => {
    let a = 1;
    for (let n: HTMLElement | null = el; n && n !== document.body; n = n.parentElement) {
      const s = getComputedStyle(n);
      if (s.display === "none" || s.visibility === "hidden") return 0;
      a *= +s.opacity;
      if (a < 0.001) return 0;
    }
    return a;
  };
  const renderViews = (dt: number) => {
    const H = vpH();
    let drew = false;
    renderer.autoClear = false;
    for (const v of views) {
      const r = v.el.getBoundingClientRect();
      if (r.bottom < -20 || r.top > H + 20 || r.width < 4) continue;
      /* the chase frame fades out with the hero, so it must stop drawing the
         instant it does; an engine card only needs to be mostly revealed */
      if (alphaOf(v.el) < (v.kind === "chase" ? 0.995 : 0.5)) continue;
      renderer.setViewport(r.left, H - r.bottom, r.width, r.height);
      renderer.setScissor(r.left, H - r.bottom, r.width, r.height);
      renderer.setScissorTest(true);
      renderer.setClearColor(0x0d0a08, 1);
      renderer.clear(true, true, false);
      if (v.kind === "chase") {
        v.cam.aspect = r.width / r.height; v.cam.updateProjectionMatrix();
        sky.position.copy(v.cam.position);
        renderer.render(scene, v.cam);
      } else {
        v.push = damp(v.push, v.want, 3.4, dt);
        v.pivot.rotation.y += dt * (0.22 + v.push * 0.5);
        const dist = (r.height > 420 ? 7.2 : 5.4) - v.push * 1.4;
        v.cam.aspect = r.width / r.height;
        v.cam.fov = r.width / r.height < 1 ? 34 : 26;
        v.cam.position.set(v.tx * 0.6 * v.push, 1.35 - v.ty * 0.5 * v.push, dist);
        v.cam.lookAt(0, -0.05, 0);
        v.cam.updateProjectionMatrix();
        renderer.render(v.scene, v.cam);
      }
      drew = true;
    }
    renderer.autoClear = true;
    renderer.setClearColor(0x0a0a0a, 1);
    if (drew) { renderer.setScissorTest(false); renderer.setViewport(0, 0, vpW(), vpH()); }
  };

  /* ================================================= the loop */
  let running = false, raf = 0, tPrev = 0, clock = 0, frameN = 0;
  const resize = () => {
    const w = vpW(), h = vpH();
    const pr = Math.min(devicePixelRatio || 1, DPR_CAP) * PERF.scale;
    renderer.setPixelRatio(pr);
    renderer.setSize(w, h, true);
    composer.setPixelRatio(pr);
    composer.setSize(w, h);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    uPx.value = h * pr * 0.5;
    if (WISP.mesh) (WISP.mesh.material as THREE.ShaderMaterial).uniforms.uPx.value = h * pr;
    layoutWord();
  };

  const frame = (now: number) => {
    if (!running || dead) return;
    const raw = (now - tPrev) / 1000 || 0;
    const dt = Math.min(raw, 0.05);
    tPrev = now; clock += dt; frameN++;
    uT.value = clock;

    /* the renderer trades resolution for frame rate on its own: pixels are
       the only knob worth turning on unknown hardware */
    if (clock > 2.5) {
      PERF.acc += raw; PERF.n++;
      if (PERF.n >= 40) {
        const avg = PERF.acc / PERF.n; PERF.acc = 0; PERF.n = 0;
        if (avg > 0.024 && PERF.scale > 0.55) { PERF.scale = Math.max(0.55, PERF.scale * 0.86); resize(); }
        else if (avg < 0.0135 && PERF.scale < 1) { PERF.scale = Math.min(1, PERF.scale + 0.08); resize(); }
      }
    }

    const prog = getProgress();
    RIG.smooth = reduce ? prog : damp(RIG.smooth, prog, 4.6, dt);
    RIG.mx = damp(RIG.mx, RIG.tmx, 2.4, dt);
    RIG.my = damp(RIG.my, RIG.tmy, 2.4, dt);
    if (INTRO.t0) {
      const el = (now - INTRO.t0) / 1000;
      RIG.intro = reduce ? 1 : sat(el / 2.6);
      WORD.reveal = reduce ? 1.2 : Math.min(1.2, el / 1.6);
    }
    RIG.focusAmt = damp(RIG.focusAmt, RIG.focus, 5, dt);
    uBoost.value = RIG.focusAmt * 0.6;

    applyCamera();
    sky.position.copy(camera.position);
    updatePlane(dt);
    updateWisps(dt);

    /* the type rises out of the ground, then dissolves as the walk begins */
    if (WORD.group && WORD.ink) {
      const near = smooth(0.02, 0.7, RIG.smooth);
      WORD.glyphs.forEach((g, i) => {
        const e = easeOut(clamp((WORD.reveal - i * 0.07) / 0.62, 0, 1));
        g.position.y = g.userData.baseY - (1 - e) * WORD.ink!.asc * 1.1;
        const m = g.material as THREE.MeshBasicMaterial;
        m.opacity = e * (1 - near);
        g.visible = m.opacity > 0.004;
      });
    }

    composer.render();
    renderViews(dt);
    raf = requestAnimationFrame(frame);
  };

  const onMove = (e: PointerEvent) => {
    RIG.tmx = (e.clientX / vpW()) * 2 - 1;
    RIG.tmy = -((e.clientY / vpH()) * 2 - 1);
  };
  if (!coarse) on("pointermove", onMove);
  on("resize", resize);
  const onVis = () => {
    if (document.hidden) { running = false; cancelAnimationFrame(raf); }
    else if (!running && INTRO.t0) { running = true; tPrev = performance.now(); raf = requestAnimationFrame(frame); }
  };
  document.addEventListener("visibilitychange", onVis);
  cleanups.push(() => document.removeEventListener("visibilitychange", onVis));

  resize();
  onStep(1, "Clear for take-off");

  return {
    start() {
      if (dead || running) return;
      INTRO.t0 = performance.now();
      running = true; tPrev = performance.now();
      raf = requestAnimationFrame(frame);
    },
    setFocus(v: boolean) { RIG.focus = v ? 1 : 0; },
    dispose() {
      dead = true; running = false;
      cancelAnimationFrame(raf);
      cleanups.forEach((f) => f());
      const kill = (s: THREE.Object3D) => s.traverse((ob) => {
        const m = ob as THREE.Mesh;
        m.geometry?.dispose();
        const mats = Array.isArray(m.material) ? m.material : m.material ? [m.material] : [];
        mats.forEach((mat) => {
          Object.values(mat).forEach((val) => { if (val instanceof THREE.Texture) val.dispose(); });
          mat.dispose();
        });
      });
      kill(scene);
      views.forEach((v) => v.kind === "engine" && kill(v.scene));
      ringMat.dispose();
      bloom.dispose();
      disposables.forEach((d) => d.dispose());
      renderer.dispose();
    },
  };
}
