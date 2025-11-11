"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";

// allow the custom element in TSX
declare global {
  namespace JSX {
    interface IntrinsicElements {
      "model-viewer": React.DetailedHTMLProps<
        React.HTMLAttributes<HTMLElement>,
        HTMLElement
      > & {
        src?: string;
        ar?: boolean;
        "camera-controls"?: boolean;
        "interaction-prompt"?: string;
        exposure?: string | number;
        "shadow-intensity"?: string | number;
        "environment-image"?: string;
      };
    }
  }
}

type MappingEntry = { target: string };
type BrainStyleMapping = Record<string, MappingEntry[]>;
type HeartStyleMapping = { segments: Record<string, string[]> };
type AnyMapping = BrainStyleMapping | HeartStyleMapping;

type Props = {
  /** AHA16 scores: "1".."16" -> 0..1 */
  segmentScores: Record<string, number>;
  /** Top-K segments to highlight (sorted by score desc). Default: 8 */
  topK?: number;
  /** Minimum score to highlight (0..1). Default: 0.25 */
  threshold?: number;
  /** Override GLB URL */
  glbUrl?: string;
  /** Override mapping URL */
  mappingUrl?: string;
};

const GLB_URL = "/static/heart/heart.glb";
const MAPPING_URL = "/static/heart/mapping.json";

export default function HeartMV({
  segmentScores,
  topK = 8,
  threshold = 0.25,
  glbUrl = GLB_URL,
  mappingUrl = MAPPING_URL,
}: Props) {
  const mvRef = useRef<any>(null);
  const [rawMapping, setRawMapping] = useState<AnyMapping | null>(null);
  const [mapping, setMapping] = useState<BrainStyleMapping | null>(null);
  const [ready, setReady] = useState(false);

  // load the web component only in browser
  useEffect(() => {
    if (typeof window === "undefined") return;
    (async () => {
      if (!customElements.get("model-viewer")) {
        await import("@google/model-viewer");
      }
    })();
  }, []);

  // load mapping.json
  useEffect(() => {
    let on = true;
    (async () => {
      try {
        const r = await fetch(mappingUrl);
        const m = (await r.json()) as AnyMapping;
        if (!on) return;
        setRawMapping(m);
      } catch (e) {
        console.error("HeartMV: failed to load mapping.json", e);
      }
    })();
    return () => {
      on = false;
    };
  }, [mappingUrl]);

  // normalize mapping to Brain-style shape
  useEffect(() => {
    if (!rawMapping) return;
    // If it already looks like Brain-style: keep it
    if (!("segments" in rawMapping)) {
      setMapping(rawMapping as BrainStyleMapping);
      return;
    }
    // Convert Heart-style segments -> Brain-style entries
    const heart = rawMapping as HeartStyleMapping;
    const out: BrainStyleMapping = {};
    Object.entries(heart.segments || {}).forEach(([sid, arr]) => {
      out[sid] = (arr || []).map((name) => ({ target: String(name) }));
    });
    setMapping(out);
  }, [rawMapping]);

  // model loaded?
  useEffect(() => {
    const el = mvRef.current;
    if (!el) return;
    const onLoad = () => setReady(true);
    el.addEventListener("load", onLoad);
    return () => el.removeEventListener("load", onLoad);
  }, []);

  // compute top-k affected entries from scores
  const affected = useMemo(() => {
    const pairs = Object.entries(segmentScores || {})
      .filter(([, s]) => typeof s === "number" && !Number.isNaN(s))
      .sort((a, b) => b[1] - a[1]) // desc by score
      .slice(0, topK)
      .filter(([, s]) => s >= threshold);
    // Convert to brain-like list with label ids/names
    return pairs.map(([sid, score]) => ({
      label_id: Number(sid),
      label_name: `AHA${sid}`,
      score,
    }));
  }, [segmentScores, topK, threshold]);

  // highlight logic
  useEffect(() => {
    if (!ready || !mapping || !mvRef.current) return;

    const mv: any = mvRef.current;
    const model = mv.model;
    if (!model) return;

    // helper: set material to faded
    const fadeMat = (mat: any) => {
      try {
        mat.setAlphaMode?.("BLEND"); // <-- critical so alpha below is respected
        const base = mat?.pbrMetallicRoughness?.baseColorFactor ?? [1, 1, 1, 1];
        mat.pbrMetallicRoughness?.setBaseColorFactor?.([base[0], base[1], base[2], 0.15]);
        mat.setEmissiveFactor?.([0, 0, 0]);
      } catch {}
    };

    // helper: set material to highlighted (reddish glow)
    const lightMat = (mat: any, intensity = 1.0) => {
      try {
        const t = Math.max(0, Math.min(1, intensity));
        const baseR = 1.0, baseG = 0.25, baseB = 0.25;
        const emiR = 0.6 * t, emiG = 0.1 * t, emiB = 0.1 * t;

        mat.setAlphaMode?.("OPAQUE");
        mat.pbrMetallicRoughness?.setBaseColorFactor?.([baseR, baseG, baseB, 1]);
        mat.setEmissiveFactor?.([emiR, emiG, emiB]);
      } catch {}
    };

    // 1) fade all materials by default
    if (Array.isArray(model.materials)) {
      model.materials.forEach(fadeMat);
    }

    // 2) build a set of targets to highlight from affected segments
    const targets = new Map<string, number>(); // name -> weight (score)
    affected.forEach(({ label_id, score }) => {
      const entries = mapping[String(label_id)] || [];
      entries.forEach((e) => {
        const name = e.target?.trim();
        if (!name) return;
        const prev = targets.get(name) ?? 0;
        targets.set(name, Math.max(prev, score)); // keep max intensity per mesh
      });
    });

    // 3) try to match by material name first (exact)
    const found = new Set<string>();
    if (Array.isArray(model.materials)) {
      for (const mat of model.materials) {
        const name = (mat?.name || "").trim();
        if (!name) continue;
        if (targets.has(name)) {
          lightMat(mat, targets.get(name)!);
          found.add(name);
        }
      }
    }

    // 4) best-effort fallback: substring/normalized match
    const norm = (s: string) =>
      s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").replace(/\s+/g, " ").trim();

    const targetNorms = [...targets.entries()].map(([raw, w]) => ({
      raw,
      w,
      n: norm(raw),
    }));

    if (Array.isArray(model.materials)) {
      for (const mat of model.materials) {
        const name = (mat?.name || "").trim();
        if (!name || found.has(name)) continue;
        const nName = norm(name);
        const hit = targetNorms.find((t) => t.n && (nName.includes(t.n) || t.n.includes(nName)));
        if (hit) {
          lightMat(mat, hit.w);
          found.add(hit.raw);
        }
      }
    }

    // 5) log any misses so you can extend mapping.json
    const misses = [...targets.keys()].filter((t) => !found.has(t));
    if (misses.length) {
      console.warn(
        `HeartMV: ${misses.length} map target(s) not found in materials. First few:`,
        misses.slice(0, 20)
      );
    }
  }, [ready, mapping, affected]);

  return (
    <model-viewer
      ref={mvRef}
      src={glbUrl}
      ar
      camera-controls
      interaction-prompt="none"
      style={{ width: "100%", height: 520, background: "transparent" }}
      exposure="1"
      shadow-intensity="0"
      environment-image="neutral"
    />
  );
}
