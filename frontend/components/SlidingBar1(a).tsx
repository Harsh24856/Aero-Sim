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
  smoothing?: number; // 0–1, higher = snappier
  disabled?: boolean;
  compact?: boolean;   // the cockpit panel's small size: 9px text, thin track, small thumb
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

export function SlidingBar({
  label, value, onChange, min = 0, max = 100, step = 1,
  format = (v) => String(Math.round(v)), minLabel, maxLabel,
  smoothing = 0.2, disabled, compact, style,
}: SlidingBarProps) {
  const z = compact
    ? { gap: 4, label: 9, value: 9, track: 18, bar: 4, thumb: 11, tickBig: 5, tickSmall: 3, ends: 8, tip: 9, ring: [2, 4] }
    : { gap: 12, label: 14, value: 20, track: 34, bar: 8, thumb: 22, tickBig: 9, tickSmall: 5, ends: 12, tip: 12, ring: [4, 7] };
  const shown = useSmoothed(value, smoothing);
  const pct = `${((shown - min) / (max - min || 1)) * 100}%`;
  const rect = useRef<DOMRect | null>(null);
  const [active, setActive] = useState(false);

  const commit = useCallback((raw: number) => {
    const v = Math.min(max, Math.max(min, Math.round((raw - min) / step) * step + min));
    if (v !== value) onChange(v);
  }, [min, max, step, value, onChange]);

  const fromX = (x: number) => {
    const r = rect.current;
    if (r) commit(min + ((x - r.left) / r.width) * (max - min));
  };
  const end = () => { rect.current = null; setActive(false); };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: z.gap, ...style }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span style={{ font: `600 ${z.label}px ${SANS}`, letterSpacing: compact ? ".1em" : ".16em", color: "#e8dcd0", textTransform: "uppercase" }}>{label}</span>
        <span style={{ font: `600 ${z.value}px ${MONO}`, color: "#ff8050", fontVariantNumeric: "tabular-nums" }}>{format(value)}</span>
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
          setActive(true);
          fromX(e.clientX);
        }}
        onPointerMove={(e) => { if (rect.current) fromX(e.clientX); }}
        onPointerUp={end}
        onPointerCancel={end}
        onKeyDown={disabled ? undefined : (e) => {
          const s = e.shiftKey ? step * 10 : step;
          const next = ({ ArrowRight: value + s, ArrowUp: value + s, ArrowLeft: value - s, ArrowDown: value - s, Home: min, End: max } as Record<string, number>)[e.key];
          if (next !== undefined) { e.preventDefault(); commit(next); }
        }}
        style={{
          position: "relative", height: z.track, touchAction: "none", outline: "none",
          cursor: disabled ? "not-allowed" : "pointer", opacity: disabled ? 0.5 : 1,
        }}
      >
        <div style={{ position: "absolute", left: 0, right: 0, top: "50%", height: z.bar, transform: "translateY(-50%)", borderRadius: 99, background: "#1f1611", boxShadow: "inset 0 1px 2px rgba(0,0,0,.7), 0 0 0 1px #3a2a20" }} />
        <div style={{ position: "absolute", left: 0, top: "50%", height: z.bar, width: pct, transform: "translateY(-50%)", borderRadius: 99, background: "linear-gradient(90deg,#ff971e,#ff5b1c 50%,#ed3919)", boxShadow: `0 0 14px rgba(255,91,28,${active ? 0.7 : 0.4})` }} />
        <div style={{ position: "absolute", top: "50%", left: pct, width: z.thumb, height: z.thumb, borderRadius: "50%", background: "radial-gradient(circle at 35% 35%, #fff4e8 0%, #ffb27a 35%, #ff5b1c 70%)", transform: `translate(-50%,-50%) scale(${active ? 1.25 : 1})`, transition: "transform .22s cubic-bezier(.3,1.7,.5,1), box-shadow .2s", boxShadow: `0 0 0 ${active ? z.ring[1] : z.ring[0]}px rgba(255,97,27,.16), 0 0 ${compact ? 8 : 18}px rgba(255,97,27,.65)` }} />
        {active && (
          <div style={{ position: "absolute", bottom: "100%", left: pct, transform: "translate(-50%,-4px)", padding: "3px 7px", borderRadius: 4, background: "#ff611b", color: "#140a05", font: `700 ${z.tip}px ${MONO}`, whiteSpace: "nowrap", boxShadow: "0 4px 14px rgba(0,0,0,.5)", pointerEvents: "none" }}>{format(value)}</div>
        )}
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", marginTop: compact ? -3 : -6 }}>
        {Array.from({ length: 11 }, (_, i) => (
          <div key={i} style={{ width: 1, height: i % 5 === 0 ? z.tickBig : z.tickSmall, background: i % 5 === 0 ? "#5a4636" : "#3a2a20" }} />
        ))}
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", font: `400 ${z.ends}px ${MONO}`, color: "#8a7a6c" }}>
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
