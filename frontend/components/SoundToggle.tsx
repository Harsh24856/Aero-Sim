"use client";

import { Volume2, VolumeX } from "lucide-react";

type Props = {
  muted: boolean;
  volume: number;               // 0..1
  onToggle: () => void;
  onVolume: (v: number) => void;
};

// Engine-sound control, floated over the simulator view.
export default function SoundToggle({ muted, volume, onToggle, onVolume }: Props) {
  return (
    <div className="pointer-events-auto flex items-center gap-1.5 rounded border border-[#4c3025] bg-black/60 px-1.5 py-1 backdrop-blur-sm">
      <button
        type="button"
        onClick={onToggle}
        aria-label={muted ? "Unmute engine sound" : "Mute engine sound"}
        title={muted ? "Engine sound off" : "Engine sound on"}
        className="text-[#e8c9a0] transition-colors hover:text-[#ff8050]"
      >
        {muted ? <VolumeX className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
      </button>
      <input
        type="range"
        min={0}
        max={100}
        value={Math.round(volume * 100)}
        onChange={(e) => onVolume(Number(e.target.value) / 100)}
        aria-label="Engine sound volume"
        className="h-1 w-16 cursor-pointer accent-[#ff8050]"
      />
    </div>
  );
}
