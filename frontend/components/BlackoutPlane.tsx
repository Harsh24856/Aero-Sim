"use client";

import { Suspense, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import * as THREE from "three";
import {
  useGLTF,
  OrbitControls,
  Environment,
  ContactShadows,
  Bounds,
  Center,
} from "@react-three/drei";

const MODEL_PATH = "/models/blackout-plane.glb";

// Only the body/fuselage materials get tinted - canopy_glass and rubber are left
// alone so a recolored plane still looks like a real aircraft (a white canopy or
// orange tires would look wrong, not just differently colored).
const BODY_MATERIAL_NAMES = new Set(["matte_black", "gunmetal", "gloss_black"]);

function PlaneModel({ tintColor }: { tintColor?: string }) {
  const { scene } = useGLTF(MODEL_PATH);
  // A GLTF scene is cached by useGLTF - clone before mounting so Three can safely
  // attach it to this canvas independently, and so each engine's themed viewer
  // can tint its OWN clone without affecting the shared cached original or any
  // other simultaneously-mounted viewer.
  const model = useMemo(() => {
    const cloned = scene.clone(true);
    if (tintColor) {
      const color = new THREE.Color(tintColor);
      cloned.traverse((child) => {
        if (!(child instanceof THREE.Mesh)) return;
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        child.material = materials.map((mat) => {
          if (!BODY_MATERIAL_NAMES.has(mat.name)) return mat;
          const tinted = mat.clone();
          if (tinted instanceof THREE.MeshStandardMaterial || tinted instanceof THREE.MeshPhysicalMaterial) {
            tinted.color.set(color);
          }
          return tinted;
        });
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

export type BlackoutPlaneProps = {
  // Home page hero (default): interactive showcase, auto-rotating, draggable.
  // In-game use (Simulator.tsx): interactive=false, since the game itself controls
  // orientation via CSS rotation to represent actual flight pitch - an
  // uncontrolled auto-rotate would fight against that and look broken.
  interactive?: boolean;
  // Explicit starting camera position. R3F's default (unspecified) looks down the
  // -Z axis from [0,0,5], which was rendering as a head-on/nose view of this model
  // (confirmed via screenshot) - meaning the model's nose points along Z. A
  // side-profile view (wings visible, not the nose) needs the camera positioned
  // along X instead. Defaults to the original home-page value (undefined = R3F's
  // own default) so the hero showcase is unaffected.
  cameraPosition?: [number, number, number];
  // Per-engine body color: white for the night theme (916), grey for beach (915),
  // orange for desert (912). Undefined = the model's own original materials
  // (the home page hero and the default/914 simulator).
  tintColor?: string;
  // Frame shape. Default: the wide strip the simulator uses. The home hero passes a
  // taller frame so the aircraft fills its half of the screen.
  frameClassName?: string;
};

export default function BlackoutPlane({ interactive = true, cameraPosition, tintColor, frameClassName = "aspect-[21/8]" }: BlackoutPlaneProps) {
  return (
    <div className={`relative w-full ${frameClassName}`}>
      <div className="absolute inset-0 bg-tertiary/20 blur-[100px] rounded-full z-0 mix-blend-screen pointer-events-none" />
      <Canvas
        camera={cameraPosition ? { fov: 40, position: cameraPosition } : { fov: 40 }}
        dpr={[1, 2]}
        className="absolute inset-0 z-10 block h-full w-full"
      >
        <ambientLight intensity={0.4} />
        <directionalLight position={[5, 8, 5]} intensity={1.2} castShadow />
        <directionalLight position={[-5, 2, -5]} intensity={0.3} color="#FF9100" />
        <Suspense fallback={<LoadingFallback />}>
          <Bounds fit clip observe margin={0.9}>
            <Center>
              <PlaneModel tintColor={tintColor} />
            </Center>
          </Bounds>
          <Environment preset="city" />
          <ContactShadows position={[0, -1.1, 0]} opacity={0.5} scale={10} blur={2} far={2} />
        </Suspense>
        {interactive && (
          <OrbitControls
            autoRotate
            autoRotateSpeed={1.2}
            enableZoom={false}
            enablePan={false}
            minPolarAngle={Math.PI / 3}
            maxPolarAngle={Math.PI / 1.8}
          />
        )}
      </Canvas>
    </div>
  );
}

useGLTF.preload(MODEL_PATH);
