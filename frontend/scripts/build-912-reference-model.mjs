import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { GLTFExporter } from "three/addons/exporters/GLTFExporter.js";

// GLTFExporter uses FileReader for binary exports. Node provides Blob but not
// FileReader, so provide the tiny browser-compatible piece it needs.
class NodeFileReader {
  result = null;

  readAsArrayBuffer(blob) {
    blob
      .arrayBuffer()
      .then((buffer) => {
        this.result = buffer;
        this.onloadend?.({ target: this });
      })
      .catch((error) => this.onerror?.(error));
  }
}

globalThis.FileReader = NodeFileReader;

const appDirectory = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const referencePath = resolve(appDirectory, "public/models/uav-engine-914.glb");
const outputPath = resolve(appDirectory, "public/models/rotax-style-912-boxer-uav-engine.glb");

// Keep the highly detailed normally aspirated boxer core of the 914 reference
// and remove only assemblies a 912 ULS does not use.
const turboOnlyAssemblies = new Set([
  "turbocharger",
  "boost_tap",
  "turbine_feed",
  "turbo_oil_feed",
  "turbo_oil_return",
  "charge_hose",
  "charge_hose_clamp",
]);

function removeTurboOnlyAssemblies(parent) {
  for (const child of [...parent.children]) {
    if (turboOnlyAssemblies.has(child.name)) {
      parent.remove(child);
      continue;
    }

    removeTurboOnlyAssemblies(child);
  }
}

const referenceBytes = await readFile(referencePath);
const loader = new GLTFLoader();
const { scene } = await loader.parseAsync(
  referenceBytes.buffer.slice(
    referenceBytes.byteOffset,
    referenceBytes.byteOffset + referenceBytes.byteLength,
  ),
  "",
);

removeTurboOnlyAssemblies(scene);
scene.name = "rotax_style_912_boxer_uav_engine";
scene.userData = {
  engineModel: "Rotax 912 ULS",
  source: "Detailed 914 boxer reference, with turbo-only assemblies removed",
};
scene.updateMatrixWorld(true);

const exporter = new GLTFExporter();
const output = await exporter.parseAsync(scene, { binary: true, trs: false });
await writeFile(outputPath, new Uint8Array(output));

let meshCount = 0;
scene.traverse((object) => {
  if (object.isMesh) meshCount += 1;
});

console.log(`Built ${outputPath} from the detailed 914 reference (${meshCount} meshes).`);
