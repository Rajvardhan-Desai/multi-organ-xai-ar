"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, ThreeEvent } from "@react-three/fiber";
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
  /** Optional overall prediction to display in the info panel */
  overallPrediction?: string;
};

const DEFAULT_GLB = "/static/heart/heart.glb";
const DEFAULT_MAPPING = "/static/heart/mapping.json";

// ---- name helpers ----------------------------------------------------------
const norm = (s: string) =>
  s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").replace(/\s+/g, " ").trim();

const variants = (raw: string) => {
  const base = raw.trim();
  const out = new Set<string>([
    base,
    base.replace(/\.(t|j|g)$/i, ""),
    base.replace(/\s*\(mesh\)\s*$/i, ""),
    base.replace(/\.(t|j|g)$/i, "").replace(/\s*\(mesh\)\s*$/i, ""),
  ]);
  return [...out];
};

// ---- material helpers (clone per mesh to avoid global tinting) -------------
function cloneMat(mat: THREE.Material) {
  const c = mat.clone ? mat.clone() : mat;
  (c as any).__heartMV = true;
  return c;
}
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

// ---- internal selection type ----------------------------------------------
type Selection = {
  nodeName: string;
  segmentId?: string;   // "1".."16" if found
  score?: number;       // 0..1
};

// ---- Scene -----------------------------------------------------------------
function HeartScene({
  segmentScores,
  topK,
  threshold,
  src,
  mapping,
  opacity,
  onSelect,
}: Required<Omit<Props, "mappingUrl" | "overallPrediction">> & {
  mapping: Mapping | null;
  onSelect: (sel: Selection | null) => void;
}) {
  const groupRef = useRef<THREE.Group>(null!);
  const gltf = useGLTF(src);

  const [nameIndex, setNameIndex] = useState<Map<string, THREE.Object3D>>(new Map());
  // hitIndex: normalizedMeshName -> set of segment IDs that include it
  const [hitIndex, setHitIndex] = useState<Map<string, Set<string>>>(new Map());

  // Build name index + ensure independent materials (once)
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

    (window as any).heartDump = () => {
      const sorted = [...names].sort((a, b) => a.localeCompare(b));
      console.groupCollapsed("HeartMV — GLB node names");
      console.log("Total:", sorted.length);
      sorted.forEach((n) => console.log(n));
      console.groupEnd();
    };
  }, [gltf.scene]);

  // resolve mapping raw target -> actual node in GLB
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

  // Precompute a "hit index": for each mapped segment, add all descendant mesh names
  useEffect(() => {
    if (!mapping || nameIndex.size === 0) return;

    const hi = new Map<string, Set<string>>();

    const add = (meshName: string, sid: string) => {
      const key = norm(meshName);
      if (!hi.has(key)) hi.set(key, new Set());
      hi.get(key)!.add(sid);
    };

    for (const [sid, arr] of Object.entries(mapping.segments || {})) {
      for (const raw of arr || []) {
        const node = resolveTarget(raw);
        if (!node) continue;
        node.traverse((obj) => {
          const mesh = obj as THREE.Mesh;
          if (mesh.isMesh && mesh.name) add(mesh.name, sid);
        });
        // also index the group node name itself
        if (node.name) add(node.name, sid);
      }
    }
    setHitIndex(hi);
  }, [mapping, nameIndex]);

  // compute top-k segments (id, score) from scores
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

  // targets to highlight: rawName -> intensity
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

  // fade all then highlight targets
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
        misses.slice(0, 12)
      );
    }
  }, [gltf.scene, targets, nameIndex, opacity]);

  // click -> report node + segment + score
  const handlePointerDown = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    const obj = e.object as THREE.Object3D;
    // climb to the closest named ancestor just in case
    let cur: THREE.Object3D | null = obj;
    while (cur && !cur.name) cur = cur.parent;
    const name = (cur?.name || obj.name || "").trim();
    if (!name) {
      onSelect(null);
      return;
    }

    const key = norm(name);
    const sids = hitIndex.get(key);
    if (sids && sids.size) {
      // pick the highest-scored segment among all mapped sids for this mesh
      let bestSid: string | undefined;
      let bestScore = -1;
      for (const sid of sids) {
        const sc = Number(segmentScores[sid] ?? 0);
        if (sc > bestScore) {
          bestScore = sc;
          bestSid = sid;
        }
      }
      onSelect({
        nodeName: name,
        segmentId: bestSid,
        score: bestScore >= 0 ? bestScore : undefined,
      });
    } else {
      onSelect({ nodeName: name });
    }
  };

  // gentle rotation
  useFrame((_, d) => {
    if (groupRef.current) groupRef.current.rotation.y += d * 0.05;
  });

  return (
    <group ref={groupRef} dispose={null} onPointerDown={handlePointerDown}>
      <primitive object={gltf.scene} />
    </group>
  );
}

// ---- Wrapper with overlay panel -------------------------------------------
export default function HeartMV({
  segmentScores,
  topK = 8,
  threshold = 0.25,
  src = DEFAULT_GLB,
  mappingUrl = DEFAULT_MAPPING,
  opacity = 0.15,
  overallPrediction,
}: Props) {
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
        console.error("HeartMV: failed to load mapping.json", e);
      }
    })();
    return () => {
      alive = false;
    };
  }, [mappingUrl]);

  return (
    <div style={{ position: "relative", width: "100%", height: 520 }}>
      {/* overlay info */}
      <div
        style={{
          position: "absolute",
          left: 12,
          top: 12,
          zIndex: 10,
          background: "rgba(255,255,255,0.8)",
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
            <div><b>Node:</b> {selected.nodeName}</div>
            {selected.segmentId && (
              <div>
                <b>Segment:</b> AHA{selected.segmentId} &nbsp;•&nbsp;{" "}
                <b>Score:</b> {(selected.score ?? 0).toFixed(3)}
              </div>
            )}
            {overallPrediction && (
              <div>
                <b>Disease:</b> {overallPrediction}
              </div>
            )}
            {!selected.segmentId && (
              <div style={{ opacity: 0.75 }}>Not mapped to an AHA segment.</div>
            )}
            <div style={{ opacity: 0.6, marginTop: 2 }}>(click another part)</div>
          </>
        ) : (
          <>
            <div><b>Tip:</b> click a highlighted area</div>
            {overallPrediction && (
              <div><b>Disease:</b> {overallPrediction}</div>
            )}
          </>
        )}
      </div>

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
            onSelect={setSelected}
          />
        </Bounds>
        <OrbitControls enableDamping makeDefault />
      </Canvas>
    </div>
  );
}

useGLTF.preload(DEFAULT_GLB);
