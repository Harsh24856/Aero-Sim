"use client";

import { Component, type ComponentProps, type ReactNode } from "react";
import { Environment } from "@react-three/drei";

// drei's <Environment preset> downloads its HDR from a CDN (raw.githack.com). Offline,
// or on a blocked network, the load throws and took the whole page down with it
// (/engine, /simulate, /engine_info all showed a runtime error). This boundary drops
// only the reflections: the scene keeps its own lights and renders. Online it is
// <Environment> exactly - it still suspends in the parent <Suspense> while loading.
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

export default function SafeEnvironment(props: ComponentProps<typeof Environment>) {
  return (
    <DropOnError>
      <Environment {...props} />
    </DropOnError>
  );
}
