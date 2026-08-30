// Original stylized illustration - a rugged V-block combustion engine, not a photo.
export default function CombustionIllustration() {
  return (
    <svg viewBox="0 0 300 160" className="w-full h-full" fill="none">
      <defs>
        <linearGradient id="ce-body" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#333" />
          <stop offset="100%" stopColor="#161616" />
        </linearGradient>
      </defs>
      {/* engine block */}
      <rect x="70" y="70" width="160" height="60" rx="6" fill="url(#ce-body)" stroke="#555" strokeWidth="1.5" />
      {/* V-bank cylinder heads */}
      <path d="M 90 70 L 110 30 L 140 30 L 150 70 Z" fill="#2a2a2a" stroke="#555" strokeWidth="1.5" />
      <path d="M 150 70 L 160 30 L 190 30 L 210 70 Z" fill="#2a2a2a" stroke="#555" strokeWidth="1.5" />
      {/* cylinder head fins */}
      {Array.from({ length: 4 }).map((_, i) => (
        <line key={`l-${i}`} x1={98 + i * 9} y1={68 - i * 2} x2={98 + i * 9} y2={34} stroke="#666" strokeWidth="1.5" />
      ))}
      {Array.from({ length: 4 }).map((_, i) => (
        <line key={`r-${i}`} x1={168 + i * 9} y1={34} x2={168 + i * 9} y2={68 - i * 2} stroke="#666" strokeWidth="1.5" />
      ))}
      {/* exhaust manifold */}
      <path d="M 60 95 Q 40 95 40 115 L 40 125" fill="none" stroke="#444" strokeWidth="6" strokeLinecap="round" />
      <path d="M 240 95 Q 260 95 260 115 L 260 125" fill="none" stroke="#444" strokeWidth="6" strokeLinecap="round" />
      {/* oil pan */}
      <path d="M 90 130 L 210 130 L 200 150 L 100 150 Z" fill="#1a1a1a" stroke="#555" strokeWidth="1.5" />
      {/* bolt details */}
      {[85, 115, 145, 175, 205, 215].map((x, i) => (
        <circle key={i} cx={x} cy={100} r="2.5" fill="#666" />
      ))}
    </svg>
  );
}
