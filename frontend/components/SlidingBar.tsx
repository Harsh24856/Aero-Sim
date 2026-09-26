import React, { useCallback, useEffect, useRef, useState } from "react";

export interface SlidingBarProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  format?: (v: number) => string;
  minLabel?: string;
  maxLabel?: string;
  segments?: number;
  smoothing?: number; // 0–1, higher = snappier
  disabled?: boolean;
  style?: React.CSSProperties;
}

const MONO = "'IBM Plex Mono', ui-monospace, Menlo, monospace";
const SANS = "Archivo, system-ui, sans-serif";

function useSmoothed(target: number, k: number) {
  const [v, setV] = useState(target);
  const cur = useRef(target);
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const diff = target - cur.current;
      if (Math.abs(diff) < 0.01) cur.current = target;
      else { cur.current += diff * k; raf = requestAnimationFrame(tick); }
      setV(cur.current);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, k]);
  return v;
}

function mix(a: string, b: string, t: number) {
  const h = (x: string) => [1, 3, 5].map((i) => parseInt(x.slice(i, i + 2), 16));
  const A = h(a), B = h(b);
  return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(",")})`;
}

const segColor = (t: number) =>
  t < 0.5 ? mix("#ff971e", "#ff5b1c", t / 0.5) : mix("#ff5b1c", "#ed3919", (t - 0.5) / 0.5);

export function SlidingBar({
  label, value, onChange, min = 0, max = 100, step = 1,
  format = (v) => String(Math.round(v)), minLabel, maxLabel,
  segments = 36, smoothing = 0.2, disabled, style,
}: SlidingBarProps) {
  const shown = useSmoothed(value, smoothing);
  const p = (shown - min) / (max - min || 1);
  const rect = useRef<DOMRect | null>(null);

  const commit = useCallback((raw: number) => {
    const v = Math.min(max, Math.max(min, Math.round((raw - min) / step) * step + min));
    if (v !== value) onChange(v);
  }, [min, max, step, value, onChange]);

  const fromX = (x: number) => {
    const r = rect.current;
    if (r) commit(min + ((x - r.left) / r.width) * (max - min));
  };

  const colors = React.useMemo(
    () => Array.from({ length: segments }, (_, i) => segColor(i / (segments - 1))),
    [segments]
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14, ...style }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span style={{ font: `600 14px ${SANS}`, letterSpacing: ".16em", color: "#e8dcd0", textTransform: "uppercase" }}>{label}</span>
        <span style={{ font: `600 20px ${MONO}`, color: "#ff8050", fontVariantNumeric: "tabular-nums" }}>{format(value)}</span>
      </div>

      <div
        role="slider"
        tabIndex={disabled ? -1 : 0}
        aria-label={label}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={value}
        aria-valuetext={format(value)}
        aria-disabled={disabled}
        onPointerDown={disabled ? undefined : (e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          rect.current = e.currentTarget.getBoundingClientRect();
          fromX(e.clientX);
        }}
        onPointerMove={(e) => { if (rect.current) fromX(e.clientX); }}
        onPointerUp={() => { rect.current = null; }}
        onPointerCancel={() => { rect.current = null; }}
        onKeyDown={disabled ? undefined : (e) => {
          const s = e.shiftKey ? step * 10 : step;
          const next = ({ ArrowRight: value + s, ArrowUp: value + s, ArrowLeft: value - s, ArrowDown: value - s, Home: min, End: max } as Record<string, number>)[e.key];
          if (next !== undefined) { e.preventDefault(); commit(next); }
        }}
        style={{
          display: "flex", gap: 3, height: 28, padding: 4, borderRadius: 6,
          background: "#140e0b", boxShadow: "inset 0 0 0 1px #3a2a20",
          cursor: disabled ? "not-allowed" : "pointer", opacity: disabled ? 0.5 : 1,
          touchAction: "none", outline: "none",
        }}
      >
        {colors.map((c, i) => (
          <div key={i} style={{ flex: 1, position: "relative", borderRadius: 2, background: "#261b15", overflow: "hidden" }}>
            <div style={{ position: "absolute", inset: 0, background: c, opacity: Math.min(1, Math.max(0, p * segments - i)), boxShadow: `0 0 8px ${c}` }} />
          </div>
        ))}
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", font: `400 12px ${MONO}`, color: "#8a7a6c" }}>
        <span>{minLabel ?? format(min)}</span>
        <span>{maxLabel ?? format(max)}</span>
      </div>
    </div>
  );
}

export default SlidingBar;

/* Usage
const [throttle, setThrottle] = useState(37);
const [speed, setSpeed] = useState(112);

<SlidingBar label="Throttle" value={throttle} onChange={setThrottle} format={(v) => `${Math.round(v)}%`} />
<SlidingBar label="Speed Target" min={68} max={135} value={speed} onChange={setSpeed} format={(v) => `${Math.round(v)} KT`} />

Fonts: load Archivo + IBM Plex Mono from Google Fonts, or change SANS / MONO above.
*/
