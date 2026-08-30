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

const MODEL_PATH = "/models/uav-engine-914.glb";

function EngineModel() {
  const { scene } = useGLTF(MODEL_PATH);
  return <primitive object={scene} />;
}

function LoadingFallback() {
  return (
    <mesh>
      <boxGeometry args={[0.3, 0.3, 0.3]} />
      <meshStandardMaterial color="#ff5500" wireframe />
    </mesh>
  );
}

export default function EngineViewer() {
  return (
    <div className="w-full h-full relative">
      <Canvas camera={{ fov: 40 }} className="relative z-10">
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 8, 5]} intensity={1.3} castShadow />
        <directionalLight position={[-5, 2, -5]} intensity={0.4} color="#ff5500" />
        <Suspense fallback={<LoadingFallback />}>
          <Bounds fit clip margin={1.1}>
            <Center>
              <EngineModel />
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
