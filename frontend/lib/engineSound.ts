// Procedural Rotax engine sound, synthesised live from telemetry with the Web Audio
// API. No audio files: every component follows the physics.
//
//   firing note   A four-stroke flat-four fires 2 times per crank revolution, so the
//                 fundamental is rpm/60 * 2 Hz (~47 Hz at 1,400 rpm, ~190 Hz at
//                 5,800). A harmonic-rich wave through a load-dependent low-pass:
//                 more throttle opens the filter, so the engine gets brighter and
//                 louder under power, the way a real one does.
//   exhaust       brown noise band-passed around the firing note - the "burble".
//   propeller     3-blade pass frequency, prop_rpm/60 * 3, a soft thrum.
//   wind          filtered noise rising with airspeed squared.
//   rough running When the AI reports misfire, the engine gain randomly drops out for
//                 single pulses; combustion instability adds pitch/level jitter.
//
// Every parameter moves with setTargetAtTime, so there are no clicks or zipper noise.
// Browsers only allow audio after a user gesture: call start() from a click handler.

export type EngineSoundInput = {
  rpm: number;          // engine rpm
  propRpm: number;      // propeller rpm
  throttle: number;     // 0..1
  load: number;         // 0..1, power / max power
  airspeed: number;     // m/s
  misfire?: boolean;
  instability?: boolean;
};

const PROP_BLADES = 3;
const SMOOTH = 0.08;       // s, time constant for continuous parameters

function noiseBuffer(ctx: AudioContext, seconds: number, brown: boolean): AudioBuffer {
  const len = Math.floor(ctx.sampleRate * seconds);
  const buf = ctx.createBuffer(1, len, ctx.sampleRate);
  const data = buf.getChannelData(0);
  let last = 0;
  for (let i = 0; i < len; i++) {
    const white = Math.random() * 2 - 1;
    if (brown) {
      last = (last + 0.02 * white) / 1.02;
      data[i] = last * 3.5;
    } else {
      data[i] = white;
    }
  }
  return buf;
}

// Pulse-like wave: strong low harmonics with alternating phase, like a cylinder pressure pulse.
function firingWave(ctx: AudioContext): PeriodicWave {
  const n = 24;
  const real = new Float32Array(n);
  const imag = new Float32Array(n);
  for (let k = 1; k < n; k++) {
    imag[k] = (k % 2 === 0 ? 0.6 : 1) / Math.pow(k, 0.85);
    real[k] = k % 3 === 0 ? 0.25 / k : 0;
  }
  return ctx.createPeriodicWave(real, imag);
}

export class EngineSound {
  private ctx: AudioContext | null = null;
  private master!: GainNode;
  private engineGain!: GainNode;
  private firing!: OscillatorNode;
  private sub!: OscillatorNode;
  private loadFilter!: BiquadFilterNode;
  private exhaustFilter!: BiquadFilterNode;
  private exhaustGain!: GainNode;
  private prop!: OscillatorNode;
  private propGain!: GainNode;
  private windFilter!: BiquadFilterNode;
  private windGain!: GainNode;
  private roughTimer: ReturnType<typeof setInterval> | null = null;
  private last: EngineSoundInput | null = null;
  private volume = 0.6;
  private muted = false;
  private running = false;

  get isRunning() { return this.running; }

  /** Must be called from a user gesture (click). Safe to call repeatedly. */
  async start(): Promise<void> {
    if (typeof window === "undefined") return;
    if (!this.ctx) {
      const Ctx = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (!Ctx) return;
      this.ctx = new Ctx();
      this.build(this.ctx);
    }
    if (this.ctx.state === "suspended") await this.ctx.resume();
    this.running = true;
    this.applyMaster(0.8);
    if (!this.roughTimer) this.roughTimer = setInterval(() => this.roughTick(), 45);
  }

  /** Fade out and suspend (pause/stop). The graph is kept for a quick restart. */
  stop(): void {
    this.running = false;
    if (!this.ctx) return;
    this.master.gain.setTargetAtTime(0, this.ctx.currentTime, 0.25);
    const ctx = this.ctx;
    setTimeout(() => { if (!this.running && ctx.state === "running") ctx.suspend().catch(() => {}); }, 1500);
  }

  dispose(): void {
    if (this.roughTimer) clearInterval(this.roughTimer);
    this.roughTimer = null;
    this.running = false;
    this.ctx?.close().catch(() => {});
    this.ctx = null;
  }

  setVolume(v: number): void {
    this.volume = Math.min(1, Math.max(0, v));
    this.applyMaster(0.1);
  }

  setMuted(m: boolean): void {
    this.muted = m;
    this.applyMaster(0.1);
  }

  /** Feed the latest telemetry (any rate; ~20 Hz is plenty). Non-finite input is ignored. */
  update(input: EngineSoundInput): void {
    if (!this.ctx || !this.running) return;
    const vals = [input.rpm, input.propRpm, input.throttle, input.load, input.airspeed];
    if (!vals.every((v) => Number.isFinite(v))) return;
    this.last = input;
    const t = this.ctx.currentTime;
    const rpm = Math.max(0, input.rpm);
    const load = Math.min(1, Math.max(0, input.load));
    const thr = Math.min(1, Math.max(0, input.throttle));

    const f = Math.max(20, (rpm / 60) * 2);
    this.firing.frequency.setTargetAtTime(f, t, SMOOTH);
    this.sub.frequency.setTargetAtTime(f / 2, t, SMOOTH);
    this.exhaustFilter.frequency.setTargetAtTime(f * 1.5, t, SMOOTH);

    // Load opens the filter and raises level: brighter, meaner under power.
    this.loadFilter.frequency.setTargetAtTime(260 + 2600 * Math.pow(0.5 * load + 0.5 * thr, 1.3), t, 0.15);
    this.loadFilter.Q.setTargetAtTime(0.8 + 2 * load, t, 0.2);
    const running = rpm > 300 ? 1 : 0;
    this.engineGain.gain.setTargetAtTime(running * (0.28 + 0.42 * load), t, 0.12);
    this.exhaustGain.gain.setTargetAtTime(running * (0.1 + 0.35 * thr), t, 0.12);

    const bpf = Math.max(5, (input.propRpm / 60) * PROP_BLADES);
    this.prop.frequency.setTargetAtTime(bpf, t, SMOOTH);
    this.propGain.gain.setTargetAtTime(Math.min(0.22, 0.03 + 0.19 * Math.pow(input.propRpm / 2600, 2)), t, 0.15);

    const v = Math.min(1, Math.max(0, input.airspeed / 70));
    this.windGain.gain.setTargetAtTime(0.02 + 0.3 * v * v, t, 0.3);
    this.windFilter.frequency.setTargetAtTime(500 + 3500 * v, t, 0.3);
  }

  // ---------------------------------------------------------------------------
  private build(ctx: AudioContext) {
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -14;
    comp.ratio.value = 4;
    comp.connect(ctx.destination);

    this.master = ctx.createGain();
    this.master.gain.value = 0;
    this.master.connect(comp);

    // Engine: firing wave + sub-octave, soft saturation, load low-pass.
    const shaper = ctx.createWaveShaper();
    const curve = new Float32Array(1024);
    for (let i = 0; i < curve.length; i++) { const x = (i / 511.5) - 1; curve[i] = Math.tanh(2.2 * x); }
    shaper.curve = curve;
    shaper.oversample = "2x";

    this.loadFilter = ctx.createBiquadFilter();
    this.loadFilter.type = "lowpass";
    this.loadFilter.frequency.value = 400;
    this.engineGain = ctx.createGain();
    this.engineGain.gain.value = 0;

    this.firing = ctx.createOscillator();
    this.firing.setPeriodicWave(firingWave(ctx));
    this.firing.frequency.value = 50;
    const firingLevel = ctx.createGain(); firingLevel.gain.value = 0.55;
    this.sub = ctx.createOscillator();
    this.sub.type = "triangle";
    this.sub.frequency.value = 25;
    const subLevel = ctx.createGain(); subLevel.gain.value = 0.35;

    this.firing.connect(firingLevel).connect(shaper);
    this.sub.connect(subLevel).connect(shaper);
    shaper.connect(this.loadFilter).connect(this.engineGain).connect(this.master);

    // Exhaust burble: brown noise around the firing note.
    const brown = ctx.createBufferSource();
    brown.buffer = noiseBuffer(ctx, 3, true);
    brown.loop = true;
    this.exhaustFilter = ctx.createBiquadFilter();
    this.exhaustFilter.type = "bandpass";
    this.exhaustFilter.Q.value = 1.4;
    this.exhaustGain = ctx.createGain();
    this.exhaustGain.gain.value = 0;
    brown.connect(this.exhaustFilter).connect(this.exhaustGain).connect(this.master);

    // Propeller thrum.
    this.prop = ctx.createOscillator();
    this.prop.type = "triangle";
    this.prop.frequency.value = 30;
    const propLp = ctx.createBiquadFilter(); propLp.type = "lowpass"; propLp.frequency.value = 220;
    this.propGain = ctx.createGain();
    this.propGain.gain.value = 0;
    this.prop.connect(propLp).connect(this.propGain).connect(this.master);

    // Wind.
    const white = ctx.createBufferSource();
    white.buffer = noiseBuffer(ctx, 3, false);
    white.loop = true;
    const windHp = ctx.createBiquadFilter(); windHp.type = "highpass"; windHp.frequency.value = 300;
    this.windFilter = ctx.createBiquadFilter();
    this.windFilter.type = "lowpass";
    this.windFilter.frequency.value = 1000;
    this.windGain = ctx.createGain();
    this.windGain.gain.value = 0;
    white.connect(windHp).connect(this.windFilter).connect(this.windGain).connect(this.master);

    [this.firing, this.sub, this.prop].forEach((o) => o.start());
    brown.start();
    white.start();
  }

  private applyMaster(timeConstant: number) {
    if (!this.ctx) return;
    const target = this.running && !this.muted ? this.volume * 0.7 : 0;
    this.master.gain.setTargetAtTime(target, this.ctx.currentTime, timeConstant);
  }

  // Rough running: random single-pulse dropouts (misfire) and jitter (instability).
  private roughTick() {
    const s = this.last;
    if (!this.ctx || !this.running || !s) return;
    const t = this.ctx.currentTime;
    if (s.misfire && Math.random() < 0.18) {
      const g = this.engineGain.gain;
      g.cancelScheduledValues(t);
      g.setTargetAtTime(0.05, t, 0.008);
      g.setTargetAtTime(0.28 + 0.42 * s.load, t + 0.03, 0.02);
    }
    if (s.instability) {
      const f = Math.max(20, (s.rpm / 60) * 2) * (1 + (Math.random() - 0.5) * 0.06);
      this.firing.frequency.setTargetAtTime(f, t, 0.02);
    }
  }
}
