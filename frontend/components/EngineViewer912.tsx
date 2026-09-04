"use client";

import { Suspense, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import {
  useGLTF,
  OrbitControls,
  Environment,
  ContactShadows,
  Bounds,
  Center,
} from "@react-three/drei";

const MODEL_PATH = "/models/rotax-style-912-boxer-uav-engine.glb";

function Engine912Model() {
  const { scene } = useGLTF(MODEL_PATH);
  // A GLTF scene is cached by useGLTF. Clone it before mounting so Three can
  // safely attach it to this canvas (and any future viewer) independently.
  const model = useMemo(() => scene.clone(true), [scene]);
  // 90deg Y-axis rotation - the model's default orientation was confirmed to look
  // wrong in the viewer. This is a best-effort assumption without a screenshot to
  // reference the exact wrong angle - if this over/under-rotates, tell me exactly
  // what you see and I'll adjust the precise value instead of guessing again.
  return <primitive object={model} rotation={[0, Math.PI / 4, 0]} />;
}

function LoadingFallback() {
  return (
    <mesh>
      <boxGeometry args={[0.3, 0.3, 0.3]} />
      <meshStandardMaterial color="#ff5500" wireframe />
    </mesh>
  );
}

export default function EngineViewer912() {
  return (
    <div className="w-full h-full relative">
      <Canvas
        camera={{ fov: 40, position: [3.2, 2.2, 4.2] }}
        dpr={[1, 2]}
        className="block h-full w-full"
      >
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 8, 5]} intensity={1.3} castShadow />
        <directionalLight position={[-5, 2, -5]} intensity={0.4} color="#ff5500" />
        <Suspense fallback={<LoadingFallback />}>
          <Bounds fit clip observe margin={1.1}>
            <Center>
              <Engine912Model />
            </Center>
          </Bounds>
          <Environment preset="city" />
          <ContactShadows position={[0, -1, 0]} opacity={0.4} scale={8} blur={2} far={2} />
        </Suspense>
        <OrbitControls
          enableZoom={false}
          enablePan={false}
          minPolarAngle={Math.PI / 3}
          maxPolarAngle={Math.PI / 1.8}
        />
      </Canvas>
    </div>
  );
}

useGLTF.preload(MODEL_PATH);
