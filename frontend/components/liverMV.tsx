"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, ThreeEvent, useFrame } from "@react-three/fiber";
import { Bounds, Environment, OrbitControls, useGLTF } from "@react-three/drei";
import * as THREE from "three";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

// mapping.json shape (same idea as brain/heart):
// {
//   "1": [{ "target": "Segment_1" }],
//   "2": [{ "target": "Segment_2" }],
//   ...
// }
type MappingEntry = { target: string };
type Mapping = Record<string, MappingEntry[]>;

type LiverMVProps = {
  /** Per-region scores from backend, e.g. { "1": 0.8, "2": 0.4, ... } */
  regionScores: Record<string, number>;
  /** How many highest-score regions to highlight */
  topK?: number;
  /** Minimum score to highlight (0..1) */
  threshold?: number;
  /** Optional overall prediction label (e.g. "F3", "Severe fibrosis") */
  overallPrediction?: string;
  /** Optional GLB path; defaults to /static/liver/liver.glb */
  src?: string;
  /** Optional mapping.json path; defaults to /static/liver/mapping.json */
  mappingUrl?: string;
};

const DEFAULT_GLB = "/static/liver/liver.glb";
const DEFAULT_MAPPING = "/static/liver/mapping.json";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const norm = (s: string) =>
  s
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .replace(/\s+([lrg])\b/g, "$1")
    .replace(/\s+/g, " ")
    .trim();

const variants = (raw: string) => {
  const base = raw.trim();
  const out = new Set<string>([
    base,
    base.replace(/\.(t|j|g)$/i, ""),
    base.replace(/\s*\(mesh\)\s*$/i, ""),
    base
      .replace(/\.(t|j|g)$/i, "")
      .replace(/\s*\(mesh\)\s*$/i, ""),
  ]);
  return [...out];
};

function cloneMat(mat: THREE.Material) {
  const c = mat.clone ? mat.clone() : mat;
  (c as any).__liverMV = true;
  return c;
}

function instanceAllMeshMaterials(root: THREE.Object3D) {
  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh) return;
    if (Array.isArray(mesh.material)) {
      mesh.material = mesh.material.map((m) =>
        m && !(m as any).__liverMV ? cloneMat(m) : m
      );
    } else if (mesh.material && !(mesh.material as any).__liverMV) {
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
}

function setHighlight(mat: THREE.Material, intensity: number) {
  const m = mat as THREE.MeshStandardMaterial;
  if (!m) return;
  const t = Math.max(0, Math.min(1, intensity));
  m.transparent = false;
  m.opacity = 1;
  if (m.color) m.color.setRGB(0.8, 0.1, 0.1);
  if ((m as any).emissive)
    (m as any).emissive.setRGB(0.7 * t, 0.15 * t, 0.15 * t);
}

// ---------------------------------------------------------------------------
// LiverScene: shared scene for R3F Canvas
// ---------------------------------------------------------------------------

type Selection = {
  nodeName: string;
  regionId?: string;
  score?: number;
};

type LiverSceneProps = {
  src: string;
  mapping: Mapping | null;
  regionScores: Record<string, number>;
  topK: number;
  threshold: number;
  opacity: number;
  onSelect: (sel: Selection | null) => void;
};

function LiverScene({
  src,
  mapping,
  regionScores,
  topK,
  threshold,
  opacity,
  onSelect,
}: LiverSceneProps) {
  const groupRef = useRef<THREE.Group>(null!);
  const gltf = useGLTF(src);

  const [nameIndex, setNameIndex] = useState<Map<string, THREE.Object3D>>(
    new Map()
  );

  // region index for quick lookup
  const regionIndex = useMemo(() => {
    const m = new Map<string, number>();
    Object.entries(regionScores).forEach(([k, v]) => {
      m.set(String(k), Number(v) || 0);
    });
    return m;
  }, [regionScores]);

  // Build name index + clone materials once
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

    // Debug helper: list liver GLB node names
    (window as any).liverDump = () => {
      const sorted = [...names].sort((a, b) => a.localeCompare(b));
      console.groupCollapsed("LiverMV — GLB node names");
      console.log("Total:", sorted.length);
      sorted.forEach((n) => console.log(n));
      console.groupEnd();
    };
  }, [gltf.scene]);

  const resolveTarget = (raw: string): THREE.Object3D | null => {
    const e = nameIndex.get(raw);
    if (e) return e;
    for (const v of variants(raw)) {
      const hit = nameIndex.get(v);
      if (hit) return hit;
    }
    const t = norm(raw);
    for (const [name, obj] of nameIndex) {
      const n = norm(name);
      if (n === t || n.includes(t) || t.includes(n)) return obj;
    }
    return null;
  };

  // Choose which regions to actually highlight
  const topRegions = useMemo(() => {
    return Object.entries(regionScores)
      .map(([id, sc]) => ({ id, score: Number(sc) || 0 }))
      .filter((r) => r.score >= threshold)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }, [regionScores, topK, threshold]);

  // target node -> intensity
  const targets = useMemo(() => {
    const map = new Map<string, number>();
    if (!mapping) return map;
    for (const { id, score } of topRegions) {
      const entries = mapping[id] || [];
      for (const { target } of entries) {
        const nm = (target || "").trim();
        if (!nm) continue;
        map.set(nm, Math.max(score, map.get(nm) ?? 0));
      }
    }
    return map;
  }, [topRegions, mapping]);

  const highlightNode = (node: THREE.Object3D, intensity: number) => {
    node.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (!mesh.isMesh) return;
      const mats = Array.isArray(mesh.material)
        ? mesh.material
        : [mesh.material];
      mats.forEach((mtl) => mtl && setHighlight(mtl, intensity));
    });
  };

  // Fade all then highlight selected targets
  useEffect(() => {
    if (!gltf?.scene) return;

    gltf.scene.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (!mesh.isMesh) return;
      const mats = Array.isArray(mesh.material)
        ? mesh.material
        : [mesh.material];
      mats.forEach((mtl) => mtl && setFade(mtl, opacity));
    });

    const misses: string[] = [];
    for (const [raw, w] of targets.entries()) {
      const node = resolveTarget(raw);
      if (node) highlightNode(node, w);
      else misses.push(raw);
    }
    if (misses.length) {
      if (process.env.NODE_ENV === "development") {
        console.warn(
          `LiverMV: ${misses.length} map target(s) not found in GLB. First few:`,
          misses.slice(0, 20)
        );
      } else {
        console.debug(
          `LiverMV: ${misses.length} map target(s) not found in GLB.`
        );
      }
    }
  }, [gltf.scene, targets, nameIndex, opacity]);

  // gentle rotation
  useFrame((_, d) => {
    if (groupRef.current) groupRef.current.rotation.y += d * 0.05;
  });

  const handlePointerDown = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    const obj = e.object as THREE.Object3D;
    let cur: THREE.Object3D | null = obj;
    while (cur && !cur.name) cur = cur.parent;
    const name = (cur?.name || obj.name || "").trim();
    if (!name) {
      onSelect(null);
      return;
    }

    // Try to find which region ID this node is mapped to (if any)
    let bestRegionId: string | undefined;
    let bestScore = -1;
    const nKey = norm(name);

    if (mapping) {
      for (const [id, entries] of Object.entries(mapping)) {
        for (const { target } of entries) {
          const tKey = norm(target || "");
          if (!tKey) continue;
          if (tKey === nKey || tKey.includes(nKey) || nKey.includes(tKey)) {
            const sc = regionIndex.get(id) ?? 0;
            if (sc > bestScore) {
              bestScore = sc;
              bestRegionId = id;
            }
          }
        }
      }
    }

    if (bestRegionId) {
      onSelect({
        nodeName: name,
        regionId: bestRegionId,
        score: regionIndex.get(bestRegionId),
      });
    } else {
      onSelect({ nodeName: name });
    }
  };

  return (
    <group
      ref={groupRef}
      dispose={null}
      onPointerDown={handlePointerDown}
    >
      <primitive object={gltf.scene} />
    </group>
  );
}

// ---------------------------------------------------------------------------
// Public component
// ---------------------------------------------------------------------------

export default function LiverMV({
  regionScores,
  topK = 8,
  threshold = 0.25,
  overallPrediction,
  src = DEFAULT_GLB,
  mappingUrl = DEFAULT_MAPPING,
}: LiverMVProps) {
  const [mapping, setMapping] = useState<Mapping | null>(null);
  const [selected, setSelected] = useState<Selection | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const r = await fetch(mappingUrl);
        const m = (await r.json()) as Mapping;
        if (alive) setMapping(m);
      } catch (e) {
        console.error("LiverMV: failed to load mapping.json", e);
      }
    })();
    return () => {
      alive = false;
    };
  }, [mappingUrl]);

  return (
    <div style={{ position: "relative", width: "100%", height: 520 }}>
      {/* Overlay info */}
      <div
        style={{
          position: "absolute",
          left: 12,
          top: 12,
          zIndex: 10,
          background: "rgba(255,255,255,0.85)",
          backdropFilter: "blur(2px)",
          border: "1px solid rgba(0,0,0,0.1)",
          borderRadius: 12,
          padding: "10px 12px",
          fontSize: 12,
          lineHeight: 1.35,
          pointerEvents: "none",
        }}
      >
        {selected ? (
          <>
            <div>
              <b>Node:</b> {selected.nodeName}
            </div>
            {selected.regionId && (
              <div>
                <b>Region ID:</b> {selected.regionId}
                {typeof selected.score === "number" && (
                  <>
                    &nbsp;•&nbsp;<b>Score:</b>{" "}
                    {Number(selected.score).toFixed(4)}
                  </>
                )}
              </div>
            )}
            {overallPrediction && (
              <div>
                <b>Prediction:</b> {overallPrediction}
              </div>
            )}
            {!selected.regionId && (
              <div style={{ opacity: 0.7 }}>
                Not mapped to a region ID.
              </div>
            )}
            <div style={{ opacity: 0.6, marginTop: 2 }}>(click another part)</div>
          </>
        ) : (
          <>
            <div>
              <b>Tip:</b> click a highlighted area
            </div>
            {overallPrediction && (
              <div>
                <b>Prediction:</b> {overallPrediction}
              </div>
            )}
          </>
        )}
      </div>

      <Canvas camera={{ position: [0, 0.2, 1.6], fov: 45 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[2, 2, 2]} intensity={0.9} />
        <Environment preset="city" />
        <Bounds fit clip observe margin={1.1}>
          <LiverScene
            src={src}
            mapping={mapping}
            regionScores={regionScores}
            topK={topK}
            threshold={threshold}
            opacity={0.2}
            onSelect={setSelected}
          />
        </Bounds>
        <OrbitControls enableDamping makeDefault />
      </Canvas>
    </div>
  );
}

// Preload GLB
useGLTF.preload(DEFAULT_GLB);
