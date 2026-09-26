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

const MODEL_PATH = "/models/rotax-style-912-boxer-uav-engine.glb";

// Y-axis rotation applied to the model. Named so it can be nudged in one place
// rather than hunting a bare Math.PI/4 in the JSX.
const MODEL_YAW = Math.PI / 4;

/**
 * Painted surfaces get a clear coat.
 *
 * The nine materials baked into this GLB already carry the right PBR values -
 * they came from the same export as rotax-912-engine.mtl, and the Kd values
 * match its baseColorFactor exactly, with metalness and roughness derived from
 * its Ns. So the colours need no correction.
 *
 * What glTF's metallic-roughness model cannot express is a *painted* surface: a
 * pigmented base under a smooth transparent layer. The MTL distinguishes these
 * by name and by shininess - rotax_teal at Ns 116 and black_crinkle at Ns 52
 * are paint, while rubber_boot at Ns 12 and the alloys are not - and paint
 * without a coat reads as dyed plastic. MeshPhysicalMaterial has the extra
 * layer, so those two materials are upgraded and the rest are left exactly as
 * the exporter set them.
 */
const CLEARCOAT: Record<string, { clearcoat: number; clearcoatRoughness: number }> = {
  rotax_teal: { clearcoat: 0.85, clearcoatRoughness: 0.18 },
  black_crinkle: { clearcoat: 0.25, clearcoatRoughness: 0.55 },
};

function Engine912Model() {
  const { scene } = useGLTF(MODEL_PATH);

  const model = useMemo(() => {
    // useGLTF caches the parsed scene, and clone() shares material instances
    // with that cache - so editing a material here would mutate every other
    // mount of this model and persist across navigations. Each material is
    // cloned before being touched.
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

      // One upgraded instance per material name, shared across the meshes that
      // use it - 398 meshes here, and a material per mesh would cost 398
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

  return <primitive object={model} rotation={[0, MODEL_YAW, 0]} />;
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
        // `castShadow` on a light does nothing unless the canvas enables
        // shadows. It was set before and silently had no effect.
        shadows
        // Six of the nine materials are metalness = 1.0 with a base colour
        // between 0.07 and 0.32, so almost all of their appearance is reflected
        // environment rather than diffuse light. Linear tone mapping crushes
        // that into mud; ACES keeps the highlight roll-off and lets the dark
        // alloys read as metal.
        gl={{
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: 1.15,
          antialias: true,
        }}
        className="block h-full w-full"
      >
        {/* Low ambient on purpose. Ambient light does nothing for a metal - it
            has no diffuse response - and flattens the painted and rubber parts,
            which are the only materials that would benefit. The environment map
            does the lifting instead. */}
        <ambientLight intensity={0.15} />
        <directionalLight
          position={[5, 8, 5]}
          intensity={2.1}
          castShadow
          shadow-mapSize={[1024, 1024]}
          shadow-bias={-0.0004}
        />
        {/* Cool rim light rather than the previous warm orange, which fought
            the rotax_teal accent - the one saturated colour on the engine. */}
        <directionalLight position={[-6, 2.5, -4]} intensity={0.55} color="#8fbcd4" />

        <Suspense fallback={<LoadingFallback />}>
          <Bounds fit clip observe margin={1.1}>
            <Center>
              <Engine912Model />
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
