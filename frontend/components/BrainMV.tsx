"use client";

import "@google/model-viewer";
import { useEffect, useMemo, useRef, useState } from "react";

/** Type guards for model-viewer’s scene-graph API (v4). */
type ModelViewerElement = HTMLElement & {
  model?: {
    // not all builds expose these at the same time, so guard every call
    getNodeByName?: (name: string) => any | null;
    setNodeProperties?: (
      node: any,
      props: Partial<{ visible: boolean; opacity: number }>
    ) => void;
    traverse?: (fn: (node: any) => void) => void;
  };
};

type MappingEntry = { target: string; side?: "L" | "R" };
type Mapping = Record<number, MappingEntry[]>;

type SingleDetail = { label_id: number; label_name: string };
type ManyDetail = { label_ids?: number[]; label_names?: string[]; exclusive?: boolean };

export default function BrainMV({
  src = "/static/brain/brain.glb",
  mapping = {},
  dimOpacity = 0.12,
}: {
  src?: string;
  mapping?: Mapping;
  dimOpacity?: number;
}) {
  const mvRef = useRef<ModelViewerElement | null>(null);
  const [ready, setReady] = useState(false);

  // Debounce updates (keeps Lit from warning about cascading updates)
  const schedule = useMemo(() => {
    let raf: number | null = null;
    return (fn: () => void) => {
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        raf = null;
        fn();
      });
    };
  }, []);

  // cache the full scene’s nodes (best effort)
  const allNodes = useRef<any[]>([]);
  const brightNodes = useRef<any[]>([]);
  const pendingSingle = useRef<SingleDetail | null>(null);
  const pendingMany = useRef<ManyDetail | null>(null);

  const getNode = (name: string) => mvRef.current?.model?.getNodeByName?.(name) ?? null;
  const setNode = (node: any, props: Partial<{ visible: boolean; opacity: number }>) =>
    mvRef.current?.model?.setNodeProperties?.(node, props);

  const captureAllNodes = () => {
    const m = mvRef.current?.model;
    allNodes.current = [];
    if (!m) return;
    if (typeof m.traverse === "function") {
      try {
        m.traverse((node: any) => allNodes.current.push(node));
      } catch {
        // ignore
      }
    }
    // If traverse was missing or returned nothing, we’ll lazily grow the cache
    // as we resolve nodes by name from mapping (below).
  };

  const dimAll = () => {
    if (!mvRef.current?.model || !setNode) return;
    if (allNodes.current.length === 0) captureAllNodes();
    try {
      allNodes.current.forEach((n) => setNode!(n, { opacity: dimOpacity }));
    } catch {
      /* noop */
    }
  };

  const restoreAll = () => {
    if (!mvRef.current?.model || !setNode) return;
    if (allNodes.current.length === 0) captureAllNodes();
    try {
      allNodes.current.forEach((n) => setNode!(n, { opacity: 1 }));
    } catch {
      /* noop */
    }
  };

  const highlightByNames = (names: string[], exclusive: boolean) => {
    const m = mvRef.current?.model;
    if (!m || !setNode) return;

    // Ensure we know the whole scene if we're doing exclusive mode
    if (exclusive) dimAll();

    // Clear previous “bright” nodes
    try {
      brightNodes.current.forEach((n) => setNode!(n, { opacity: exclusive ? 1 : 1 }));
    } catch {}
    brightNodes.current = [];

    // Activate targets
    const found: any[] = [];
    names.forEach((nm) => {
      const node = getNode(nm);
      if (node) {
        found.push(node);
        // keep the mesh fully visible
        try {
          setNode!(node, { opacity: 1 });
        } catch {}
        // If we didn’t have traverse, we at least collect seen nodes for later restore
        if (!mvRef.current?.model?.traverse && !allNodes.current.includes(node)) {
          allNodes.current.push(node);
        }
      }
    });

    brightNodes.current = found;

    if (!exclusive && allNodes.current.length === 0) {
      // If not dimming others and we still have no cache, try to capture now
      captureAllNodes();
    }
  };

  /** Convert label ids/names to node name targets via mapping & string fallbacks. */
  const labelsToNodeNames = (ids?: number[], namesIn?: string[]) => {
    const out: string[] = [];
    if (ids && ids.length) {
      ids.forEach((id) => {
        const entries = mapping[id] || [];
        entries.forEach((e) => out.push(e.target));
      });
    }
    if (namesIn && namesIn.length) {
      namesIn.forEach((label) => {
        out.push(label);
        // simple L/R variants
        const isLeft = /\bleft\b/i.test(label) || /\.l\b/i.test(label);
        const isRight = /\bright\b/i.test(label) || /\.r\b/i.test(label);
        if (isLeft || isRight) {
          const base = label.replace(/\b(left|right)\b/gi, "").trim();
          out.push(
            `${base}.l`,
            `${base}.L`,
            `${base} (Left)`,
            `${base} (left)`,
            `${base}.r`,
            `${base}.R`,
            `${base} (Right)`,
            `${base} (right)`
          );
        }
      });
    }
    return Array.from(new Set(out));
  };

  // on load: mark ready, cache scene, flush any pending highlights
  useEffect(() => {
    const onLoad = () => {
      setReady(true);
      captureAllNodes();
      // announce for other components
      (window as any).__brainReady = true;
      window.dispatchEvent(new CustomEvent("brain-ready"));

      // Flush any queued highlight calls
      if (pendingMany.current) {
        const p = pendingMany.current;
        pendingMany.current = null;
        const names = labelsToNodeNames(p.label_ids, p.label_names);
        schedule(() => highlightByNames(names, !!p.exclusive));
      } else if (pendingSingle.current) {
        const s = pendingSingle.current;
        pendingSingle.current = null;
        const names = labelsToNodeNames([s.label_id], [s.label_name]);
        schedule(() => highlightByNames(names, false));
      }
    };
    const el = mvRef.current;
    el?.addEventListener("load", onLoad as any);
    return () => el?.removeEventListener("load", onLoad as any);
  }, [schedule]);

  // listen for single-ROI highlight
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent<SingleDetail>).detail;
      if (!detail) return;
      if (!mvRef.current?.model?.getNodeByName || !mvRef.current?.model?.setNodeProperties) {
        pendingSingle.current = detail;
        return;
      }
      const names = labelsToNodeNames([detail.label_id], [detail.label_name]);
      schedule(() => highlightByNames(names, false));
    };
    window.addEventListener("highlight-roi", handler as any);
    return () => window.removeEventListener("highlight-roi", handler as any);
  }, [schedule, mapping]);

  // listen for many-ROI highlight (exclusive fade others)
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent<ManyDetail>).detail;
      if (!detail) return;
      const haveApi =
        !!mvRef.current?.model?.getNodeByName && !!mvRef.current?.model?.setNodeProperties;
      if (!haveApi) {
        pendingMany.current = detail;
        return;
      }
      const names = labelsToNodeNames(detail.label_ids, detail.label_names);
      schedule(() => highlightByNames(names, !!detail.exclusive));
    };
    window.addEventListener("highlight-rois", handler as any);
    return () => window.removeEventListener("highlight-rois", handler as any);
  }, [schedule, mapping]);

  const clearAll = () => {
    // full reset to opaque everything
    restoreAll();
  };

  return (
    <div className="w-full border rounded-xl overflow-hidden bg-white">
      <div className="flex items-center justify-between px-3 py-2 border-b bg-slate-50">
        <div className="text-sm text-slate-600">
          {ready ? "3D brain ready" : "Loading 3D brain…"}
        </div>
        <button
          className="text-xs px-2 py-1 rounded bg-slate-200 hover:bg-slate-300"
          onClick={clearAll}
          type="button"
        >
          Clear highlight
        </button>
      </div>

      {/* @ts-ignore */}
      <model-viewer
        ref={mvRef as any}
        src={src}
        ar
        camera-controls
        exposure="1.0"
        shadow-intensity="0"
        style={{ width: "100%", height: 560, background: "#0b1020" }}
      ></model-viewer>
    </div>
  );
}
