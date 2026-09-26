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
import * as THREE from "three";

const MODEL_PATH = "/models/rotax-916-engine.glb";

/**
 * Clear coat for the 916's coated surfaces.
 *
 * This model carries 10 materials across 249 meshes and its palette is its own
 * - it has no black_crinkle and no painted accent at all, and instead carries
 * ignition, induction and fuel-system parts the other models do not:
 *
 *   plug_ceramic   0.73 0.71 0.66   glazed spark-plug insulator
 *   filter_media   0.20 0.02 0.06   red paper air filter
 *   black_polymer  0.017            moulded plastic
 *   bronze_shaft   0.21 0.10 0.03   metal, metalness 1.0
 *   fuel_line      0.43 0.25 0.05   amber extruded hose
 *
 * The PBR values are already correct - the same export produced
 * rotax-916-engine.mtl, whose Kd values match baseColorFactor exactly. No
 * colour is corrected here.
 *
 * What glTF's metallic-roughness model cannot express is a coated surface.
 * Glaze, moulded polymer and extruded hose have one; bare alloy, bronze and
 * paper filter media do not. Coating filter_media in particular would turn a
 * dry paper element into something that looks oil-soaked.
 */
const CLEARCOAT: Record<string, { clearcoat: number; clearcoatRoughness: number }> = {
  plug_ceramic: { clearcoat: 0.9, clearcoatRoughness: 0.1 },
  black_polymer: { clearcoat: 0.45, clearcoatRoughness: 0.35 },
  fuel_line: { clearcoat: 0.3, clearcoatRoughness: 0.4 },
  // Deliberately absent: cast_alloy, machined_alloy, finned_alloy,
  // exhaust_steel, bronze_shaft (bare metal); rubber_boot (rubber);
  // filter_media (paper).
};

function Engine916Model() {
  const { scene } = useGLTF(MODEL_PATH);

  const model = useMemo(() => {
    // useGLTF caches the parsed scene, and clone() shares material instances
    // with that cache - editing one would mutate every other mount of this
    // model and persist across navigations. Materials are cloned before use.
    const root = scene.clone(true);
    const upgraded = new Map<string, THREE.Material>();

    root.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return;
      child.castShadow = true;
      child.receiveShadow = true;

      const source = child.material;
      if (Array.isArray(source) || !(source instanceof THREE.MeshStandardMaterial)) return;

      const coat = CLEARCOAT[source.name];
      if (!coat) return;

      // One upgraded instance per material name, shared by the meshes using it,
      // rather than one per mesh across 249 meshes.
      let next = upgraded.get(source.name);
      if (!next) {
        const physical = new THREE.MeshPhysicalMaterial();
        THREE.MeshStandardMaterial.prototype.copy.call(physical, source);
        physical.clearcoat = coat.clearcoat;
        physical.clearcoatRoughness = coat.clearcoatRoughness;
        physical.name = source.name;
        upgraded.set(source.name, physical);
        next = physical;
      }
      child.material = next;
    });

    return root;
  }, [scene]);

  return <primitive object={model} />;
}

function LoadingFallback() {
  return (
    <mesh>
      <boxGeometry args={[0.3, 0.3, 0.3]} />
      <meshStandardMaterial color="#ff5500" wireframe />
    </mesh>
  );
}

export default function EngineViewer916() {
  return (
    <div className="w-full h-full relative">
      <Canvas
        camera={{ fov: 40, position: [3.2, 2.2, 4.2] }}
        dpr={[1, 2]}
        // `castShadow` on a light does nothing unless the canvas enables
        // shadows. It was set before and silently had no effect.
        shadows
        // Five of the ten materials are metalness = 1.0 with a base colour
        // between 0.08 and 0.32, so nearly all of their appearance is reflected
        // environment rather than diffuse light. Linear tone mapping crushes
        // that into mud; ACES keeps the highlight roll-off.
        gl={{
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: 1.15,
          antialias: true,
        }}
        className="block h-full w-full"
      >
        {/* Low ambient on purpose: ambient light does nothing for a metal, and
            flattens the ceramic, polymer and filter parts that would otherwise
            benefit. The environment map does the lifting. */}
        <ambientLight intensity={0.15} />
        <directionalLight
          position={[5, 8, 5]}
          intensity={2.1}
          castShadow
          shadow-mapSize={[1024, 1024]}
          shadow-bias={-0.0004}
        />
        {/* Cool rim rather than the previous warm orange. This model already
            carries three warm materials - filter_media, bronze_shaft and
            fuel_line - and an orange fill pushed them toward looking like the
            same substance. */}
        <directionalLight position={[-6, 2.5, -4]} intensity={0.55} color="#8fbcd4" />

        <Suspense fallback={<LoadingFallback />}>
          {/* Tighter than the other viewers. `Bounds` margin is the padding
              multiplier around the fitted model, so a smaller number zooms in;
              the 916 is the most compact of the four models and sat noticeably
              smaller in frame at the shared 1.1. */}
          <Bounds fit clip observe margin={0.9}>
            <Center>
              <Engine916Model />
            </Center>
          </Bounds>
          <Environment preset="city" environmentIntensity={1.25} />
          <ContactShadows position={[0, -1, 0]} opacity={0.45} scale={8} blur={2.2} far={2} />
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
