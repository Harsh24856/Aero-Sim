"use client";

import { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import {
  useGLTF,
  OrbitControls,
  Environment,
  ContactShadows,
  Bounds,
  Center,
} from "@react-three/drei";

const MODEL_PATH = "/models/blackout-plane.glb";

function PlaneModel() {
  const { scene } = useGLTF(MODEL_PATH);
  return <primitive object={scene} />;
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
};

export default function BlackoutPlane({ interactive = true, cameraPosition }: BlackoutPlaneProps) {
  return (
    <div className="w-full h-full aspect-[21/8] relative">
      <div className="absolute inset-0 bg-tertiary/20 blur-[100px] rounded-full z-0 mix-blend-screen pointer-events-none" />
      <Canvas camera={cameraPosition ? { fov: 40, position: cameraPosition } : { fov: 40 }} className="relative z-10">
        <ambientLight intensity={0.4} />
        <directionalLight position={[5, 8, 5]} intensity={1.2} castShadow />
        <directionalLight position={[-5, 2, -5]} intensity={0.3} color="#FF9100" />
        <Suspense fallback={<LoadingFallback />}>
          <Bounds fit clip margin={0.9}>
            <Center>
              <PlaneModel />
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
