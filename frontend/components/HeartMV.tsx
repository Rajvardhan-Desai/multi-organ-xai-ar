"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, useGLTF, Bounds, Environment } from "@react-three/drei";
import * as THREE from "three";

type Mapping = { segments: Record<string, string[]> };

type Props = {
  segmentScores: Record<string, number>;
  topK?: number;
  threshold?: number;
  src?: string;
  mappingUrl?: string;
  opacity?: number; // non-highlight fade
};

const DEFAULT_GLB = "/static/heart/heart.glb";
const DEFAULT_MAPPING = "/static/heart/mapping.json";

// ---------- name helpers ----------
const norm = (s: string) =>
  s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").replace(/\s+/g, " ").trim();

const variants = (raw: string) => {
  const base = raw.trim();
  const out = new Set<string>([
    base,
    base.replace(/\.(t|j|g)$/i, ""), // strip .t/.j/.g
    base.replace(/\s*\(mesh\)\s*$/i, ""),
    base.replace(/\.(t|j|g)$/i, "").replace(/\s*\(mesh\)\s*$/i, ""),
  ]);
  return [...out];
};

// ---------- material helpers ----------
function cloneMat(mat: THREE.Material) {
  const c = mat.clone ? mat.clone() : mat;
  (c as any).__heartMV = true;
  return c;
}

// Per-mesh material instancing (run once)
function instanceAllMeshMaterials(root: THREE.Object3D) {
  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh) return;
    if (Array.isArray(mesh.material)) {
      mesh.material = mesh.material.map((m) =>
        m && !(m as any).__heartMV ? cloneMat(m) : m
      );
    } else if (mesh.material && !(mesh.material as any).__heartMV) {
      mesh.material = cloneMat(mesh.material);
    }
  });
}

function setFade(mat: THREE.Material, opacity: number) {
  const m = mat as THREE.MeshStandardMaterial;
  if (!m) return;
  m.transparent = true;
  m.opacity = opacity;
  if ((m as any).emissive) (m as any).emissive.setRGB(0, 0, 0);
  if (m.color) m.color.setRGB(1, 1, 1);
}

function setHighlight(mat: THREE.Material, intensity: number) {
  const m = mat as THREE.MeshStandardMaterial;
  if (!m) return;
  const t = Math.max(0, Math.min(1, intensity));
  m.transparent = false;
  m.opacity = 1;
  if (m.color) m.color.setRGB(1, 0.25, 0.25);
  if ((m as any).emissive) (m as any).emissive.setRGB(0.6 * t, 0.1 * t, 0.1 * t);
}

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

  // Build index + instance materials (once)
  useEffect(() => {
    if (!gltf?.scene) return;

    instanceAllMeshMaterials(gltf.scene);

    const idx = new Map<string, THREE.Object3D>();
    const names: string[] = [];
    gltf.scene.traverse((obj) => {
      if (!obj.name) return;
      idx.set(obj.name, obj);
      names.push(obj.name);
    });
    setNameIndex(idx);

    // debug helper
    (window as any).heartDump = () => {
      const sorted = [...names].sort((a, b) => a.localeCompare(b));
      console.groupCollapsed("HeartMV — GLB node names");
      console.log("Total:", sorted.length);
      sorted.forEach((n) => console.log(n));
      console.groupEnd();
    };
  }, [gltf.scene]);

  // top-K segments
  const affected = useMemo(() => {
    const pairs = Object.entries(segmentScores || {})
      .filter(([, s]) => typeof s === "number" && !Number.isNaN(s))
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK)
      .filter(([, s]) => s >= threshold);
    return pairs.map(([sid, score]) => ({
      id: sid,
      score: Math.max(0, Math.min(1, Number(score))),
    }));
  }, [segmentScores, topK, threshold]);

  // target map: rawName -> intensity
  const targets = useMemo(() => {
    const map = new Map<string, number>();
    if (!mapping) return map;
    for (const { id, score } of affected) {
      const items = mapping.segments?.[id] ?? [];
      for (const raw of items) {
        const nm = (raw || "").trim();
        if (!nm) continue;
        map.set(nm, Math.max(score, map.get(nm) ?? 0));
      }
    }
    return map;
  }, [affected, mapping]);

  const highlightNode = (node: THREE.Object3D, intensity: number) => {
    node.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (!mesh.isMesh) return;
      const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      mats.forEach((mtl) => mtl && setHighlight(mtl, intensity));
    });
  };

  const resolveTarget = (raw: string): THREE.Object3D | null => {
    // exact
    const e = nameIndex.get(raw);
    if (e) return e;
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

  // fade all then highlight
  useEffect(() => {
    if (!gltf?.scene) return;

    gltf.scene.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (!mesh.isMesh) return;
      const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      mats.forEach((mtl) => mtl && setFade(mtl, opacity));
    });

    const misses: string[] = [];
    for (const [raw, w] of targets.entries()) {
      const node = resolveTarget(raw);
      if (node) highlightNode(node, w);
      else misses.push(raw);
    }
    if (misses.length) {
      console.warn(
        `HeartMV: ${misses.length} map target(s) not found in GLB. First few:`,
        misses.slice(0, 20)
      );
    }
  }, [gltf.scene, targets, nameIndex, opacity]);

  // gentle rotation
  useFrame((_, d) => {
    if (groupRef.current) groupRef.current.rotation.y += d * 0.05;
  });

  return (
    <group ref={groupRef} dispose={null}>
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
    let alive = true;
    (async () => {
      try {
        const r = await fetch(mappingUrl);
        const m = (await r.json()) as Mapping;
        if (alive) setMapping(m);
      } catch (e) {
        console.error("HeartMV: failed to load mapping.json", e);
      }
    })();
    return () => {
      alive = false;
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
            src={src}
            mapping={mapping}
            opacity={opacity}
          />
        </Bounds>
        <OrbitControls enableDamping makeDefault />
      </Canvas>
    </div>
  );
}

useGLTF.preload(DEFAULT_GLB);
