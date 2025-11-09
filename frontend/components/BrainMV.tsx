"use client";

import React, { useEffect, useRef, useState } from "react";

// allow the custom element in TSX
declare global {
  namespace JSX {
    interface IntrinsicElements {
      "model-viewer": React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement> & {
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

type TopRegion = { label_id: number; label_name: string; score: number };
type MappingEntry = { target: string };
type Mapping = Record<string, MappingEntry[]>;

const GLB_URL = "/static/brain/brain.glb";
const MAPPING_URL = "/static/brain/mapping.json";

export default function BrainMV({ affected = [] as TopRegion[] }) {
  const mvRef = useRef<any>(null);
  const [mapping, setMapping] = useState<Mapping | null>(null);
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
        const r = await fetch(MAPPING_URL);
        const m = (await r.json()) as Mapping;
        if (on) setMapping(m);
      } catch (e) {
        console.error("Failed to load mapping.json", e);
      }
    })();
    return () => {
      on = false;
    };
  }, []);

  // model loaded?
  useEffect(() => {
    const el = mvRef.current;
    if (!el) return;
    const onLoad = () => setReady(true);
    el.addEventListener("load", onLoad);
    return () => el.removeEventListener("load", onLoad);
  }, []);

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

    // helper: set material to highlighted
    const lightMat = (mat: any) => {
      try {
        mat.setAlphaMode?.("OPAQUE"); // fully visible
        mat.pbrMetallicRoughness?.setBaseColorFactor?.([1, 0.25, 0.25, 1]); // reddish
        mat.setEmissiveFactor?.([0.6, 0.1, 0.1]); // glow a bit
      } catch {}
    };

    // 1) fade all materials by default
    if (Array.isArray(model.materials)) {
      model.materials.forEach(fadeMat);
    }

    // 2) build a set of targets to highlight from affected regions
    const targets = new Set<string>();
    affected.forEach(({ label_id }) => {
      const entries = mapping[String(label_id)] || [];
      entries.forEach((e) => targets.add(e.target));
    });

    // 3) try to match by material name first
    const found = new Set<string>();
    if (Array.isArray(model.materials)) {
      for (const mat of model.materials) {
        const name = (mat?.name || "").trim();
        if (!name) continue;
        if (targets.has(name)) {
          lightMat(mat);
          found.add(name);
        }
      }
    }

    // 4) best-effort fallback: substring/normalized match
    const norm = (s: string) =>
      s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").replace(/\s+/g, " ").trim();

    const allTargetNorms = [...targets].map((t) => ({ raw: t, n: norm(t) }));
    if (Array.isArray(model.materials)) {
      for (const mat of model.materials) {
        const name = (mat?.name || "").trim();
        if (!name) continue;
        if (found.has(name)) continue;
        const nName = norm(name);
        const hit = allTargetNorms.find((t) => t.n && (nName.includes(t.n) || t.n.includes(nName)));
        if (hit) {
          lightMat(mat);
          found.add(hit.raw);
        }
      }
    }

    // 5) log any misses so you can extend mapping.json
    const misses = [...targets].filter((t) => !found.has(t));
    if (misses.length) {
      console.warn(
        `model-viewer: ${misses.length} map target(s) not found in materials. First few:`,
        misses.slice(0, 20)
      );
    }
  }, [ready, mapping, affected]);

  return (
    <model-viewer
      ref={mvRef}
      src={GLB_URL}
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
