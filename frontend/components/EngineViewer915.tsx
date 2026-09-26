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

const MODEL_PATH = "/models/rotax-style-915-turbo-uav-engine.glb";

/**
 * Clear coat for the 915's painted surfaces.
 *
 * This model carries 7 materials across 412 meshes, and they already hold
 * correct PBR values - the same export produced rotax-915-engine.mtl, whose Kd
 * values match baseColorFactor exactly and whose Ns produced the metalness and
 * roughness. No colour is corrected here.
 *
 * What glTF's metallic-roughness model cannot express is a coated surface:
 * pigment under a smooth transparent layer. The 915 has exactly one such
 * material. Its other six - cast_alloy, machined_alloy, finned_alloy,
 * cast_iron, exhaust_steel (bare metal) and rubber_boot - are uncoated, and
 * giving them a coat would make them look wet.
 */
const CLEARCOAT: Record<string, { clearcoat: number; clearcoatRoughness: number }> = {
  black_crinkle: { clearcoat: 0.25, clearcoatRoughness: 0.55 },
};

function Engine915Model() {
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

      // One upgraded instance per material name, shared by the meshes using it.
      // This model has 412 meshes; a material per mesh would mean hundreds of
      // shader compilations for no visual difference.
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

export default function EngineViewer915() {
  return (
    <div className="w-full h-full relative">
      <Canvas
        camera={{ fov: 40, position: [3.2, 2.2, 4.2] }}
        dpr={[1, 2]}
        // `castShadow` on a light does nothing unless the canvas enables
        // shadows. It was set before and silently had no effect.
        shadows
        // Five of the seven materials are metalness = 1.0 with a base colour
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
            flattens the painted and rubber parts that would otherwise benefit.
            The environment map does the lifting. */}
        <ambientLight intensity={0.15} />
        <directionalLight
          position={[5, 8, 5]}
          intensity={2.1}
          castShadow
          shadow-mapSize={[1024, 1024]}
          shadow-bias={-0.0004}
        />
        {/* Cool rim rather than the previous warm orange. The 915's palette is
            entirely neutral greys and browns, and an orange fill pushed the
            exhaust_steel and cast_iron further orange than they are. */}
        <directionalLight position={[-6, 2.5, -4]} intensity={0.55} color="#8fbcd4" />

        <Suspense fallback={<LoadingFallback />}>
          <Bounds fit clip observe margin={1.1}>
            <Center>
              <Engine915Model />
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
