"use client";

import { Suspense, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import * as THREE from "three";
import { useGLTF, Environment, ContactShadows, Bounds, Center } from "@react-three/drei";

const MODEL_PATH = "/models/blackout-plane.glb";

// Only the body/fuselage materials get tinted - canopy_glass and rubber are left
// alone so a recolored plane still looks like a real aircraft.
const BODY_MATERIAL_NAMES = new Set(["matte_black", "gunmetal", "gloss_black"]);

function PlaneMesh({ tintColor }: { tintColor?: string }) {
  const { scene } = useGLTF(MODEL_PATH);
  const model = useMemo(() => {
    const cloned = scene.clone(true);
    if (tintColor) {
      const color = new THREE.Color(tintColor);
      const tintMaterial = (mat: THREE.Material) => {
        if (!BODY_MATERIAL_NAMES.has(mat.name)) return mat;
        const tinted = mat.clone();
        if (tinted instanceof THREE.MeshStandardMaterial || tinted instanceof THREE.MeshPhysicalMaterial) {
          tinted.color.set(color);
        }
        return tinted;
      };
      cloned.traverse((child) => {
        if (!(child instanceof THREE.Mesh)) return;
        // Preserve whether the original was a single material or an array -
        // forcing everything to an array (even single-material meshes that never
        // had one) is a real correctness bug, not just a style choice: Three.js
        // treats the two cases differently internally.
        if (Array.isArray(child.material)) {
          child.material = child.material.map(tintMaterial);
        } else {
          child.material = tintMaterial(child.material);
        }
      });
    }
    return cloned;
  }, [scene, tintColor]);
  return <primitive object={model} />;
}

function LoadingFallback() {
  return (
    <mesh>
      <boxGeometry args={[0.4, 0.4, 0.4]} />
      <meshStandardMaterial color="#FF9100" wireframe />
    </mesh>
  );
}

export type GamePlaneProps = {
  // Per-engine body color override. Undefined = the model's original materials
  // (used for the base/914 Simulator, which has no assigned tint).
  tintColor?: string;
};

// Dedicated in-game plane viewer - always non-interactive (the game itself
// controls orientation via CSS rotation to represent flight pitch, so
// draggable OrbitControls or auto-rotate would fight against that and look
// broken) and always uses the confirmed side-profile camera angle. This is
// deliberately separate from BlackoutPlane.tsx, which stays the interactive
// showcase used on the home page hero - the two components no longer share
// conditional interactive/cameraPosition props for two genuinely different
// use cases.
export default function GamePlane({ tintColor }: GamePlaneProps) {
  return (
    <div style={{ width: "100%", height: "100%" }}>
      <Canvas camera={{ fov: 40, position: [-5, 0.4, 0] }} dpr={[1, 2]}>
        <ambientLight intensity={0.4} />
        <directionalLight position={[5, 8, 5]} intensity={1.2} castShadow />
        <directionalLight position={[-5, 2, -5]} intensity={0.3} color="#FF9100" />
        <Suspense fallback={<LoadingFallback />}>
          <Bounds fit clip margin={0.9}>
            <Center>
              <PlaneMesh tintColor={tintColor} />
            </Center>
          </Bounds>
          <Environment preset="city" />
          <ContactShadows position={[0, -1.1, 0]} opacity={0.5} scale={10} blur={2} far={2} />
        </Suspense>
      </Canvas>
    </div>
  );
}

useGLTF.preload(MODEL_PATH);
