// Original stylized illustration - a turbofan cross-section, not a photo. Built as an
// SVG so it renders crisply at any size with zero licensing concerns.
export default function TurbofanIllustration() {
  return (
    <svg viewBox="0 0 300 160" className="w-full h-full" fill="none">
      <defs>
        <linearGradient id="tf-body" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#3a3a3a" />
          <stop offset="100%" stopColor="#1a1a1a" />
        </linearGradient>
      </defs>
      <rect x="40" y="45" width="200" height="70" rx="35" fill="url(#tf-body)" stroke="#555" strokeWidth="1.5" />
      <circle cx="75" cy="80" r="32" fill="#0d0d0d" stroke="#666" strokeWidth="1.5" />
      {Array.from({ length: 14 }).map((_, i) => (
        <line
          key={i}
          x1={75 + 6 * Math.cos((i * Math.PI * 2) / 14)}
          y1={80 + 6 * Math.sin((i * Math.PI * 2) / 14)}
          x2={75 + 30 * Math.cos((i * Math.PI * 2) / 14)}
          y2={80 + 30 * Math.sin((i * Math.PI * 2) / 14)}
          stroke="#888"
          strokeWidth="2.5"
          strokeLinecap="round"
        />
      ))}
      <circle cx="75" cy="80" r="7" fill="#444" stroke="#777" strokeWidth="1" />
      <rect x="115" y="60" width="18" height="40" fill="#2a2a2a" stroke="#555" strokeWidth="1" />
      <rect x="138" y="55" width="18" height="50" fill="#2a2a2a" stroke="#555" strokeWidth="1" />
      <rect x="161" y="58" width="18" height="44" fill="#2a2a2a" stroke="#555" strokeWidth="1" />
      <path d="M 240 55 L 275 65 L 275 95 L 240 105 Z" fill="#151515" stroke="#555" strokeWidth="1.5" />
      <path d="M 140 45 L 155 20 L 175 20 L 165 45 Z" fill="#252525" stroke="#555" strokeWidth="1.5" />
    </svg>
  );
}
