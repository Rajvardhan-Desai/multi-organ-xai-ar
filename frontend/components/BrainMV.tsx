"use client";

import React, {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Canvas,
  useFrame,
  ThreeEvent,
} from "@react-three/fiber";
import {
  OrbitControls,
  useGLTF,
  Bounds,
  Environment,
} from "@react-three/drei";
import { XR, createXRStore } from "@react-three/xr";
import * as THREE from "three";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type TopRegion = { label_id: number; label_name: string; score: number };

// mapping.json shape:
// {
//   "1": [{ "target": "Third ventricle" }],
//   "2": [{ "target": "Fourth ventricle" }],
//   ...
// }
type MappingEntry = { target: string };
type Mapping = Record<string, MappingEntry[]>;

type Props = {
  /** Top regions from the model (already sorted or not, we'll sort) */
  affected?: TopRegion[];
  /** Limit number of regions to highlight */
  topK?: number;
  /** Minimum score to highlight (0..1) */
  threshold?: number;
  /** GLB URL */
  src?: string;
  /** Mapping URL */
  mappingUrl?: string;
  /** Base fade opacity for non-highlighted meshes */
  opacity?: number;
  /** Optional overall prediction (disease) to display in info panel */
  overallPrediction?: string;
};

const DEFAULT_GLB = "/static/brain/brain.glb";
const DEFAULT_MAPPING = "/static/brain/mapping.json";

// ---- name helpers ----------------------------------------------------------
const norm = (s: string) =>
  s
    .toLowerCase()
    // Turn non-alphanumeric into spaces
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    // Join trailing single-letter suffixes (l/r/g)
    // e.g. "telencephalon r" -> "telencephalonr"
    //      "hemisphere g"   -> "hemisphereg"
    .replace(/\s+([lrg])\b/g, "$1")
    // Normalize remaining spaces
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

// ---- material helpers (clone per mesh to avoid global tinting) -------------
function cloneMat(mat: THREE.Material) {
  const c = mat.clone ? mat.clone() : mat;
  (c as any).__brainMV = true;
  return c;
}

function instanceAllMeshMaterials(root: THREE.Object3D) {
  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh) return;
    if (Array.isArray(mesh.material)) {
      mesh.material = mesh.material.map((m) =>
        m && !(m as any).__brainMV ? cloneMat(m) : m
      );
    } else if (mesh.material && !(mesh.material as any).__brainMV) {
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
  if ((m as any).emissive)
    (m as any).emissive.setRGB(0.6 * t, 0.1 * t, 0.1 * t);
}

// ---- internal selection type ----------------------------------------------
type Selection = {
  nodeName: string;
  labelId?: number;
  labelName?: string;
  score?: number;
};

// ---------------------------------------------------------------------------
// BrainScene (shared between normal 3D + AR)
// ---------------------------------------------------------------------------
function BrainScene({
  affected,
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

  const [nameIndex, setNameIndex] = useState<
    Map<string, THREE.Object3D>
  >(new Map());
  // hitIndex: normalizedMeshName -> set of label ids that include it (as string)
  const [hitIndex, setHitIndex] = useState<
    Map<string, Set<string>>
  >(new Map());

  // index from label_id -> region for fast lookup
  const regionIndex = useMemo(() => {
    const m = new Map<string, TopRegion>();
    affected.forEach((r) => m.set(String(r.label_id), r));
    return m;
  }, [affected]);

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

    (window as any).brainDump = () => {
      const sorted = [...names].sort((a, b) =>
        a.localeCompare(b)
      );
      console.groupCollapsed("BrainMV — GLB node names");
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

  // Precompute a "hit index": for each mapped label, add all descendant mesh names
  useEffect(() => {
    if (!mapping || nameIndex.size === 0) return;

    const hi = new Map<string, Set<string>>();

    const add = (meshName: string, labelId: string) => {
      const key = norm(meshName);
      if (!hi.has(key)) hi.set(key, new Set());
      hi.get(key)!.add(labelId);
    };

    for (const [labelId, entries] of Object.entries(mapping)) {
      (entries || []).forEach(({ target }) => {
        const node = resolveTarget(target);
        if (!node) return;
        node.traverse((obj) => {
          const mesh = obj as THREE.Mesh;
          if (!mesh.isMesh) return;
          if (mesh.name) add(mesh.name, labelId);
        });
        if (node.name) add(node.name, labelId);
      });
    }
    setHitIndex(hi);
  }, [mapping, nameIndex]);

  // compute top-k regions to actually highlight
  const topRegions = useMemo(() => {
    return [...affected]
      .filter(
        (r) =>
          typeof r.score === "number" &&
          !Number.isNaN(r.score)
      )
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .filter((r) => r.score >= threshold);
  }, [affected, topK, threshold]);

  // targets to highlight: rawName -> intensity
  const targets = useMemo(() => {
    const map = new Map<string, number>();
    if (!mapping) return map;
    for (const r of topRegions) {
      const entries = mapping[String(r.label_id)] || [];
      for (const { target } of entries) {
        const nm = (target || "").trim();
        if (!nm) continue;
        map.set(nm, Math.max(r.score, map.get(nm) ?? 0));
      }
    }
    return map;
  }, [topRegions, mapping]);

  const highlightNode = (
    node: THREE.Object3D,
    intensity: number
  ) => {
    node.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (!mesh.isMesh) return;
      const mats = Array.isArray(mesh.material)
        ? mesh.material
        : [mesh.material];
      mats.forEach((mtl) => mtl && setHighlight(mtl, intensity));
    });
  };

  // fade all then highlight targets
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
          `BrainMV: ${misses.length} map target(s) not found in GLB. First few:`,
          misses.slice(0, 20)
        );
      } else {
        console.debug(
          `BrainMV: ${misses.length} map target(s) not found in GLB.`
        );
      }
    }
  }, [gltf.scene, targets, nameIndex, opacity]);

  // click / tap -> report node + label + score
  const handlePointerDown = (
    e: ThreeEvent<PointerEvent>
  ) => {
    e.stopPropagation();
    const obj = e.object as THREE.Object3D;
    let cur: THREE.Object3D | null = obj;
    while (cur && !cur.name) cur = cur.parent;
    const name = (cur?.name || obj.name || "").trim();
    if (!name) {
      onSelect(null);
      return;
    }

    const key = norm(name);
    const ids = hitIndex.get(key);
    if (ids && ids.size) {
      let bestId: string | undefined;
      let bestScore = -1;
      for (const id of ids) {
        const region = regionIndex.get(id);
        const sc = Number(region?.score ?? 0);
        if (sc > bestScore) {
          bestScore = sc;
          bestId = id;
        }
      }
      const bestRegion = bestId
        ? regionIndex.get(bestId)
        : undefined;
      onSelect({
        nodeName: name,
        labelId:
          bestRegion?.label_id ??
          (bestId ? Number(bestId) : undefined),
        labelName: bestRegion?.label_name,
        score: bestRegion?.score,
      });
    } else {
      onSelect({ nodeName: name });
    }
  };

  // gentle rotation
  useFrame((_, d) => {
    if (groupRef.current)
      groupRef.current.rotation.y += d * 0.05;
  });

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
// Normal 3D viewer (BrainMV)
// ---------------------------------------------------------------------------
export function BrainMV({
  affected = [],
  topK = 8,
  threshold = 0.25,
  src = DEFAULT_GLB,
  mappingUrl = DEFAULT_MAPPING,
  opacity = 0.15,
  overallPrediction,
}: Props) {
  const [mapping, setMapping] = useState<Mapping | null>(
    null
  );
  const [selected, setSelected] = useState<Selection | null>(
    null
  );

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const r = await fetch(mappingUrl);
        const m = (await r.json()) as Mapping;
        if (alive) setMapping(m);
      } catch (e) {
        console.error(
          "BrainMV: failed to load mapping.json",
          e
        );
      }
    })();
    return () => {
      alive = false;
    };
  }, [mappingUrl]);

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        height: 520,
      }}
    >
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
            <div>
              <b>Node:</b> {selected.nodeName}
            </div>
            {selected.labelId && (
              <div>
                <b>Region:</b> #{selected.labelId}{" "}
                {selected.labelName
                  ? `— ${selected.labelName}`
                  : ""}
                {typeof selected.score === "number" && (
                  <>
                    &nbsp;•&nbsp;<b>Score:</b>{" "}
                    {Number(selected.score).toFixed(6)}
                  </>
                )}
              </div>
            )}
            {overallPrediction && (
              <div>
                <b>Disease:</b> {overallPrediction}
              </div>
            )}
            {!selected.labelId && (
              <div style={{ opacity: 0.75 }}>
                Not mapped to a region ID.
              </div>
            )}
            <div
              style={{
                opacity: 0.6,
                marginTop: 2,
              }}
            >
              (click another part)
            </div>
          </>
        ) : (
          <>
            <div>
              <b>Tip:</b> click a highlighted area
            </div>
            {overallPrediction && (
              <div>
                <b>Disease:</b> {overallPrediction}
              </div>
            )}
          </>
        )}
      </div>

      <Canvas camera={{ position: [0, 0.2, 1.6], fov: 45 }}>
        <ambientLight intensity={0.6} />
        <directionalLight
          position={[2, 2, 2]}
          intensity={0.9}
        />
        <Environment preset="city" />
        <Bounds fit clip observe margin={1.1}>
          <BrainScene
            affected={affected}
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

// ---------------------------------------------------------------------------
// AR viewer (BrainMV_AR)
// ---------------------------------------------------------------------------

const xrStore = createXRStore();

export function BrainMV_AR({
  affected = [],
  topK = 8,
  threshold = 0.25,
  src = DEFAULT_GLB,
  mappingUrl = DEFAULT_MAPPING,
  opacity = 0.15,
  overallPrediction,
}: Props) {
  const [mapping, setMapping] = useState<Mapping | null>(
    null
  );
  const [selected, setSelected] = useState<Selection | null>(
    null
  );

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const r = await fetch(mappingUrl);
        const m = (await r.json()) as Mapping;
        if (alive) setMapping(m);
      } catch (e) {
        console.error(
          "BrainMV_AR: failed to load mapping.json",
          e
        );
      }
    })();
    return () => {
      alive = false;
    };
  }, [mappingUrl]);

  const enterAR = async () => {
    try {
      await xrStore.enterAR(); // immersive-ar session (must be from a user gesture)
    } catch (e) {
      console.error("Failed to enter AR", e);
    }
  };

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        height: "100vh", // full-screen AR
        overflow: "hidden",
      }}
    >
      {/* AR start button (required user gesture by WebXR) */}
      <button
        type="button"
        onClick={enterAR}
        style={{
          position: "absolute",
          zIndex: 30,
          left: 12,
          bottom: 12,
          padding: "8px 12px",
          borderRadius: 999,
          border: "none",
          fontSize: 13,
          background: "rgba(0,0,0,0.75)",
          color: "white",
          cursor: "pointer",
        }}
      >
        Enter AR
      </button>

      {/* overlay info (DOM overlay) */}
      <div
        style={{
          position: "absolute",
          left: 12,
          top: 12,
          zIndex: 20,
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
            <div>
              <b>Node:</b> {selected.nodeName}
            </div>
            {selected.labelId && (
              <div>
                <b>Region:</b> #{selected.labelId}{" "}
                {selected.labelName
                  ? `— ${selected.labelName}`
                  : ""}
                {typeof selected.score === "number" && (
                  <>
                    &nbsp;•&nbsp;<b>Score:</b>{" "}
                    {Number(selected.score).toFixed(6)}
                  </>
                )}
              </div>
            )}
            {overallPrediction && (
              <div>
                <b>Disease:</b> {overallPrediction}
              </div>
            )}
            {!selected.labelId && (
              <div style={{ opacity: 0.75 }}>
                Not mapped to a region ID.
              </div>
            )}
            <div
              style={{
                opacity: 0.6,
                marginTop: 2,
              }}
            >
              (tap another part)
            </div>
          </>
        ) : (
          <>
            <div>
              <b>Tip:</b> tap a highlighted area
            </div>
            {overallPrediction && (
              <div>
                <b>Disease:</b> {overallPrediction}
              </div>
            )}
          </>
        )}
      </div>

      {/* AR Canvas (no OrbitControls / Environment) */}
      <Canvas camera={{ fov: 45 }} gl={{ alpha: true }}>
        <XR store={xrStore}>
          <ambientLight intensity={0.6} />
          <directionalLight
            position={[2, 2, 2]}
            intensity={0.9}
          />
          <Bounds fit clip observe margin={1.1}>
            <BrainScene
              affected={affected}
              topK={topK}
              threshold={threshold}
              src={src}
              mapping={mapping}
              opacity={opacity}
              onSelect={setSelected}
            />
          </Bounds>
        </XR>
      </Canvas>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Smart wrapper: detects AR support and shows toggle
// ---------------------------------------------------------------------------

type Mode = "3d" | "ar";

/**
 * Default export:
 * - If device has no WebXR AR -> just shows BrainMV (3D)
 * - If AR is supported -> shows a 3D / AR toggle in the top-right
 */
export default function BrainMV_WithARToggle(props: Props) {
  const [arSupported, setArSupported] = useState(false);
  const [mode, setMode] = useState<Mode>("3d");

  // Detect WebXR AR support
  useEffect(() => {
    if (typeof window === "undefined") return;

    const nav = navigator as any;
    if (!nav.xr || !nav.xr.isSessionSupported) {
      setArSupported(false);
      return;
    }

    nav.xr
      .isSessionSupported("immersive-ar")
      .then((supported: boolean) => {
        setArSupported(!!supported);
      })
      .catch(() => setArSupported(false));
  }, []);

  const showAR = arSupported && mode === "ar";

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      {/* Only show toggle if AR is actually supported */}
      {arSupported && (
        <div
          style={{
            position: "absolute",
            right: 12,
            top: 12,
            zIndex: 50,
            display: "flex",
            gap: 4,
          }}
        >
          <button
            type="button"
            onClick={() => setMode("3d")}
            disabled={mode === "3d"}
            style={{
              padding: "6px 10px",
              fontSize: 12,
              borderRadius: 999,
              border: "1px solid rgba(0,0,0,0.2)",
              background:
                mode === "3d" ? "rgba(0,0,0,0.85)" : "white",
              color: mode === "3d" ? "white" : "black",
              cursor: mode === "3d" ? "default" : "pointer",
            }}
          >
            3D
          </button>
          <button
            type="button"
            onClick={() => setMode("ar")}
            disabled={mode === "ar"}
            style={{
              padding: "6px 10px",
              fontSize: 12,
              borderRadius: 999,
              border: "1px solid rgba(0,0,0,0.2)",
              background:
                mode === "ar" ? "rgba(0,0,0,0.85)" : "white",
              color: mode === "ar" ? "white" : "black",
              cursor: mode === "ar" ? "default" : "pointer",
            }}
          >
            AR
          </button>
        </div>
      )}

      {/* Main content */}
      {showAR ? (
        <BrainMV_AR {...props} />
      ) : (
        <BrainMV {...props} />
      )}
    </div>
  );
}

// Preload GLB for both viewers
useGLTF.preload(DEFAULT_GLB);
