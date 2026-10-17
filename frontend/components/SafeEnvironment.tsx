"use client";

import { Component, type ComponentProps, type ReactNode } from "react";
import { Environment } from "@react-three/drei";

// drei's <Environment preset> downloads its HDR from a CDN (raw.githack.com), so the 3D
// views lost their lighting - and before the boundary below, the whole page - whenever
// that host was unreachable (an offline demo, a blocked network). The preset files are
// served from public/hdri instead: the same file drei's preset uses
// (pmndrs/drei-assets@456060a, sha256 7afe4c2f...a95696), so the lighting is unchanged.
const PRESET_FILES = {
  city: "/hdri/potsdamer_platz_1k.hdr",
} as const;

type Props = Omit<ComponentProps<typeof Environment>, "preset" | "files"> & {
  preset: keyof typeof PRESET_FILES;
};

// The boundary stays as a safety net: if the file ever fails to load, only the
// reflections go, and the scene keeps its own lights.
class DropOnError extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: Error) {
    console.warn(`Environment lighting unavailable, rendering without it: ${error.message}`);
  }

  render() {
    return this.state.failed ? null : this.props.children;
  }
}

export default function SafeEnvironment({ preset, ...props }: Props) {
  return (
    <DropOnError>
      <Environment files={PRESET_FILES[preset]} {...props} />
    </DropOnError>
  );
}
