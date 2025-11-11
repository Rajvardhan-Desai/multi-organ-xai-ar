"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, useGLTF, Bounds, Environment } from "@react-three/drei";
import * as THREE from "three";

type Mapping = { segments: Record<string, string[]> };

type Props = {
  /** AHA16 scores: "1".."16" -> 0..1 */
  segmentScores: Record<string, number>;
  /** Top-K segments to highlight (sorted by score desc). Default: 8 */
  topK?: number;
  /** Minimum score to highlight (0..1). Default: 0.25 */
  threshold?: number;
  /** GLB URL */
  src?: string;
  /** Mapping URL (JSON with { segments: { "1": ["Node name", ...], ... }}) */
  mappingUrl?: string;
  /** Base fade opacity for non-highlighted meshes */
  opacity?: number;
};

const DEFAULT_GLB = "/static/heart/heart.glb";
const DEFAULT_MAPPING = "/static/heart/mapping.json";

/** Normalize a name for fuzzy comparisons */
const norm = (s: string) =>
  s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").replace(/\s+/g, " ").trim();

/** Generate a few common export variants (strip .t/.j etc.) */
const variants = (raw: string) => {
  const base = raw.trim();
  const out = new Set<string>([
    base,
    base.replace(/\.(t|j)$/i, ""),
    base.replace(/\s*\(mesh\)\s*$/i, ""),
    base.replace(/\.(t|j)$/i, "").replace(/\s*\(mesh\)\s*$/i, ""),
  ]);
  return [...out];
};

function HeartScene({
  segmentScores,
  topK,
  threshold,
  src,
  mapping,
  opacity,
}: Required<Omit<Props, "mappingUrl">> & { mapping: Mapping | null }) {
  const groupRef = useRef<THREE.Group>(null!);
  const gltf = useGLTF(src);
  const [nameIndex, setNameIndex] = useState<Map<string, THREE.Object3D>>(new Map());
  const [allNames, setAllNames] = useState<string[]>([]);

  // Build a name -> Object3D index for fast lookups.
  useEffect(() => {
    const idx = new Map<string, THREE.Object3D>();
    const names: string[] = [];
    gltf.scene.traverse((obj) => {
      if (!obj.name) return;
      idx.set(obj.name, obj);
      names.push(obj.name);
    });
    setNameIndex(idx);
    setAllNames(names);

    // expose debug dumper
    (window as any).heartDump = () => {
      const sorted = [...names].sort((a, b) => a.localeCompare(b));
      console.groupCollapsed("HeartMV — GLB node names");
      console.log("Total:", sorted.length);
      sorted.forEach((n) => console.log(n));
      console.groupEnd();
    };
  }, [gltf.scene]);

  // compute top-k affected entries from scores (id, score)
  const affected = useMemo(() => {
    const pairs = Object.entries(segmentScores || {})
      .filter(([, s]) => typeof s === "number" && !Number.isNaN(s))
      .sort((a, b) => b[1] - a[1]) // desc
      .slice(0, topK)
      .filter(([, s]) => s >= threshold);
    return pairs.map(([sid, score]) => ({
      id: sid,
      score: Math.max(0, Math.min(1, Number(score))),
      label: `AHA${sid}`,
    }));
  }, [segmentScores, topK, threshold]);

  // Build target map: nodeName -> intensity
  const targets = useMemo(() => {
    const map = new Map<string, number>();
    if (!mapping) return map;

    for (const { id, score } of affected) {
      const entries = mapping.segments?.[id] ?? [];
      for (const raw of entries) {
        const name = String(raw || "").trim();
        if (!name) continue;
        map.set(name, Math.max(score, map.get(name) ?? 0));
      }
    }
    return map;
  }, [affected, mapping]);

  // Helper: apply fade to a mesh/material
  const fadeMat = (mat: THREE.Material) => {
    const m = mat as THREE.MeshStandardMaterial;
    if (!m) return;
    m.transparent = true;
    m.opacity = opacity;
    m.emissive?.setRGB(0, 0, 0);
    m.color?.setRGB(1, 1, 1);
  };

  // Helper: apply highlight to a mesh/material
  const lightMat = (mat: THREE.Material, intensity = 1) => {
    const m = mat as THREE.MeshStandardMaterial;
    if (!m) return;
    const t = Math.max(0, Math.min(1, intensity));
    m.transparent = false;
    m.opacity = 1;
    m.color?.setRGB(1, 0.25, 0.25);
    if ((m as any).emissive) (m as any).emissive.setRGB(0.6 * t, 0.1 * t, 0.1 * t);
  };

  // Helper: highlight a node (including descendants if it's a Group/empty).
  const highlightNode = (node: THREE.Object3D, intensity: number) => {
    node.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (mesh.isMesh) {
        const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        mats.forEach((mtl) => mtl && lightMat(mtl, intensity));
      }
    });
  };

  // Resolve a single target name to a node by exact/variant/fuzzy match.
  const resolveTarget = (raw: string): THREE.Object3D | null => {
    // exact
    const exact = nameIndex.get(raw);
    if (exact) return exact;
    // variants
    for (const v of variants(raw)) {
      const hit = nameIndex.get(v);
      if (hit) return hit;
    }
    // fuzzy
    const t = norm(raw);
    for (const [name, obj] of nameIndex) {
      const n = norm(name);
      if (n === t || n.includes(t) || t.includes(n)) return obj;
    }
    return null;
  };

  // On every relevant change, fade everything and then highlight targets
  useEffect(() => {
    if (!gltf?.scene) return;

    // 1) fade all meshes
    gltf.scene.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (mesh.isMesh) {
        const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        mats.forEach((mtl) => mtl && fadeMat(mtl));
      }
    });

    // 2) highlight resolved targets
    const misses: string[] = [];
    for (const [raw, w] of targets.entries()) {
      const node = resolveTarget(raw);
      if (node) {
        highlightNode(node, w);
      } else {
        misses.push(raw);
      }
    }

    if (misses.length) {
      console.warn(
        `HeartMV: ${misses.length} map target(s) not found in GLB. First few:`,
        misses.slice(0, 20)
      );
      // Tip: run window.heartDump() and paste the exact names into mapping.json
    }
  }, [gltf.scene, targets, nameIndex, opacity]);

  // small idle rotation for a bit of life (optional)
  useFrame((_, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.05;
    }
  });

  return (
    <group ref={groupRef} dispose={null}>
      {/* Render loaded glTF */}
      <primitive object={gltf.scene} />
    </group>
  );
}

export default function HeartMV({
  segmentScores,
  topK = 8,
  threshold = 0.25,
  src = DEFAULT_GLB,
  mappingUrl = DEFAULT_MAPPING,
  opacity = 0.15,
}: Props) {
  const [mapping, setMapping] = useState<Mapping | null>(null);

  useEffect(() => {
    let on = true;
    (async () => {
      try {
        const r = await fetch(mappingUrl);
        const m = (await r.json()) as Mapping;
        if (!on) return;
        setMapping(m);
      } catch (e) {
        console.error("HeartMV: failed to load mapping.json", e);
      }
    })();
    return () => {
      on = false;
    };
  }, [mappingUrl]);

  return (
    <div style={{ width: "100%", height: 520 }}>
      <Canvas camera={{ position: [0, 0.2, 1.6], fov: 45 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[2, 2, 2]} intensity={0.9} />
        <Environment preset="city" />
        <Bounds fit clip observe margin={1.1}>
          <HeartScene
            segmentScores={segmentScores}
            topK={topK}
            threshold={threshold}
            src={src!}
            mapping={mapping}
            opacity={opacity}
          />
        </Bounds>
        <OrbitControls enableDamping makeDefault />
      </Canvas>
    </div>
  );
}

// Helpful for static analysis tools
useGLTF.preload(DEFAULT_GLB);
