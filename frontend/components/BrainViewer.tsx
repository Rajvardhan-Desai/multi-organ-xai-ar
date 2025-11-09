// components/BrainViewer.tsx
"use client";

import React, { useEffect, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Environment, Html, useGLTF } from "@react-three/drei";
import * as THREE from "three";

type Weights = Record<number, number>; // {label_id: normalized weight 0..1}
type NameToId = Record<string, number>; // optional meshName -> label_id

type Props = {
  glbUrl?: string;                       // default: /static/brain/brain.glb
  weightsByLabel?: Weights;              // per-ROI weights from SHAP/top_regions
  nameToLabelId?: NameToId;              // optional explicit mapping (meshName -> label_id)
  fadeOthers?: boolean;                  // default true; fade if weight==0
  maxEmissiveIntensity?: number;         // default 0.8
};

function Scene({
  glbUrl = "/static/brain/brain.glb",
  weightsByLabel = {},
  nameToLabelId = {},
  fadeOthers = true,
  maxEmissiveIntensity = 0.8,
}) {
  // load once (three/drei caches internally)
  const { scene } = useGLTF(glbUrl) as unknown as { scene: THREE.Group };

  // build a quick matcher that tries:
  //  1) exact name match in nameToLabelId
  //  2) fuzzy: if node.name contains label_name keywords, you could extend here
  const resolveLabelId = (meshName: string): number | undefined => {
    if (nameToLabelId && nameToLabelId[meshName] != null) return nameToLabelId[meshName];
    // fallback: attempt to parse trailing ".l"/".r" and strip
    const base = meshName.replace(/\.[lr]$/i, "");
    if (nameToLabelId && nameToLabelId[base] != null) return nameToLabelId[base];
    return undefined;
  };

  useEffect(() => {
    scene.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (!mesh.isMesh) return;

      // ensure we have a standard material we can tint
      const mat = mesh.material as THREE.MeshStandardMaterial;
      if (!mat || !(mat instanceof THREE.MeshStandardMaterial)) {
        // replace with standard material if needed
        (mesh as any).material = new THREE.MeshStandardMaterial({ color: "#888888" });
      }

      const lid = resolveLabelId(mesh.name);
      const w = (lid != null && weightsByLabel[lid] != null) ? weightsByLabel[lid] : 0;

      const baseColor = new THREE.Color(0x808080);

      // Highlight by weight: hue from gray → warm; emissive to glow slightly
      const highlight = new THREE.Color().setHSL(
        0.02 + 0.1 * w,   // subtle hue shift with weight
        0.8 * w,          // more saturated when higher
        0.45 + 0.25 * w   // brighter when higher
      );

      const material = (mesh.material as THREE.MeshStandardMaterial);
      if (w > 0) {
        material.color.copy(highlight);
        material.emissive.copy(highlight);
        material.emissiveIntensity = maxEmissiveIntensity * w;
        material.transparent = true;
        material.opacity = 0.95;
      } else {
        // faded background
        material.color.copy(baseColor);
        material.emissive.setRGB(0, 0, 0);
        material.emissiveIntensity = 0;
        material.transparent = fadeOthers;
        material.opacity = fadeOthers ? 0.15 : 0.95;
      }
      material.needsUpdate = true;
    });
  }, [scene, weightsByLabel, maxEmissiveIntensity, fadeOthers]);

  return (
    <>
      <primitive object={scene} />
      <Environment preset="city" />
      <directionalLight position={[3, 5, 2]} intensity={1.1} />
      <ambientLight intensity={0.5} />
      <gridHelper args={[10, 10, "#222", "#222"]} position={[0, -2.5, 0]} />
    </>
  );
}

export default function BrainViewer(props: Props) {
  return (
    <div className="w-full h-[520px] rounded-xl overflow-hidden border border-gray-200">
      <Canvas camera={{ position: [0, 0.5, 3.5], fov: 50 }}>
        <React.Suspense
          fallback={
            <Html center>
              <div className="px-3 py-2 rounded bg-white shadow text-gray-700 text-sm">
                Loading brain.glb…
              </div>
            </Html>
          }
        >
          <Scene {...props} />
        </React.Suspense>
        <OrbitControls makeDefault />
      </Canvas>
    </div>
  );
}

useGLTF.preload("/static/brain/brain.glb");
