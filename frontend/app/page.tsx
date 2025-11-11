"use client";

import React, { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";

// Dynamically load viewers on the client only
const BrainMV = dynamic(() => import("../components/BrainMV"), { ssr: false });
const HeartMV = dynamic(() => import("../components/HeartMV"), { ssr: false });

const BACKEND = process.env.NEXT_PUBLIC_BACKEND ?? "http://127.0.0.1:8000";

type RegistryPayload = { organs: { organ: string; diseases: string[] }[] };

type TopRegion = { label_id: number; label_name: string; score: number };
type InferResponse = {
  // shared
  prediction: string;
  proba: Record<string, number>;
  used_features?: string[];

  // brain-legacy
  icv_mm3?: number;
  top_regions?: TopRegion[];
  xai?: { method: string; top_regions?: TopRegion[] };

  // heart
  segment_scores?: Record<string, number>; // "1".."16" -> 0..1
};

export default function Page() {
  const [orgs, setOrgs] = useState<{ organ: string; diseases: string[] }[]>([]);
  const [organ, setOrgan] = useState("");
  const [disease, setDisease] = useState("");

  // Inputs
  const [file, setFile] = useState<File | null>(null);     // brain single file
  const [edFile, setEdFile] = useState<File | null>(null); // heart ED mask
  const [esFile, setEsFile] = useState<File | null>(null); // heart ES mask
  const [xai, setXai] = useState(true);

  // UI state
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<InferResponse | null>(null);

  // Fetch registry (with fallback)
  useEffect(() => {
    let ok = true;
    (async () => {
      try {
        const r = await fetch(`${BACKEND}/registry`);
        const data: RegistryPayload = await r.json();
        const rows = Array.isArray(data?.organs) ? data.organs : [];
        const normalized = rows
          .filter((x) => x && typeof x.organ === "string" && Array.isArray(x.diseases))
          .map((x) => ({ organ: x.organ, diseases: x.diseases }));

        const fallback =
          normalized.length > 0
            ? normalized
            : [
                { organ: "brain", diseases: ["alzheimer"] },
                { organ: "heart", diseases: ["cardiomyopathy"] },
              ];

        if (!ok) return;
        setOrgs(fallback);
        if (!organ && fallback.length) setOrgan(fallback[0].organ);
      } catch {
        if (!ok) return;
        const fallback = [
          { organ: "brain", diseases: ["alzheimer"] },
          { organ: "heart", diseases: ["cardiomyopathy"] },
        ];
        setOrgs(fallback);
        if (!organ) setOrgan("brain");
      }
    })();
    return () => {
      ok = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const diseases = useMemo(() => {
    const row = orgs.find((x) => x.organ === organ);
    return row?.diseases ?? [];
  }, [orgs, organ]);

  useEffect(() => {
    if (!diseases.includes(disease)) setDisease(diseases[0] ?? "");
    // reset inputs when organ changes
    setResult(null);
    setError(null);
    setFile(null);
    setEdFile(null);
    setEsFile(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [diseases.length, organ]);

  const onInfer = async () => {
    setBusy(true);
    setError(null);
    setResult(null);

    try {
      const qs = new URLSearchParams({
        organ,
        disease,
        xai: xai ? "1" : "0",
      });

      const fd = new FormData();

      if (organ === "heart") {
        // Prefer ED+ES masks; allow fallback: single ED used as both
        if (!edFile && !file) {
          throw new Error("Please choose ED mask (.nii/.nii.gz) for heart.");
        }
        if (edFile) fd.append("ed_file", edFile);
        if (esFile) fd.append("es_file", esFile ?? edFile ?? (file as File));
        if (!edFile && file) {
          // fallback to legacy single-file param accepted by backend
          fd.append("file", file);
        }
      } else {
        // brain (single segmentation file)
        if (!file) throw new Error("Please choose a segmentation file (.nii/.nii.gz).");
        fd.append("file", file);
      }

      const url = `${BACKEND}/infer?${qs.toString()}`;
      const r = await fetch(url, { method: "POST", body: fd });
      const data = await r.json();
      if (!r.ok) throw new Error(data?.detail || "Inference failed");
      setResult(data as InferResponse);
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setBusy(false);
    }
  };

  const affectedBrain: TopRegion[] = useMemo(() => {
    if (!result) return [];
    return result?.xai?.top_regions?.length
      ? (result.xai.top_regions as TopRegion[])
      : result.top_regions ?? [];
  }, [result]);

  const segScoresHeart = useMemo(
    () => (result?.segment_scores ? result.segment_scores : {}),
    [result]
  );

  return (
    <main className="min-h-screen p-6 flex flex-col gap-6 bg-white">
      <h1 className="text-2xl font-semibold">EX-AI-AR — Multi-organ</h1>

      {/* Organ / disease / XAI */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">Organ</label>
          <select
            className="border rounded px-3 py-2 w-full"
            value={organ}
            onChange={(e) => setOrgan(e.target.value)}
          >
            {orgs.map((o) => (
              <option key={o.organ} value={o.organ}>
                {o.organ}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Disease</label>
          <select
            className="border rounded px-3 py-2 w-full"
            value={disease}
            onChange={(e) => setDisease(e.target.value)}
          >
            {diseases.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-end">
          <label className="inline-flex items-center gap-2">
            <input
              type="checkbox"
              checked={xai}
              onChange={(e) => setXai(e.target.checked)}
            />
            <span>Explainable AI</span>
          </label>
        </div>
      </div>

      {/* Inputs */}
      {organ === "heart" ? (
        <div className="flex flex-col lg:flex-row items-start gap-4">
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium">ED mask (.nii/.nii.gz)</label>
            <input
              type="file"
              accept=".nii,.nii.gz"
              onChange={(e) => setEdFile(e.target.files?.[0] ?? null)}
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium">ES mask (.nii/.nii.gz)</label>
            <input
              type="file"
              accept=".nii,.nii.gz"
              onChange={(e) => setEsFile(e.target.files?.[0] ?? null)}
            />
            <div className="text-xs text-gray-500">
              Optional — if omitted, ED will be reused (reduced EF quality).
            </div>
          </div>
          <button
            className="bg-black text-white rounded px-4 py-2 disabled:opacity-50"
            onClick={onInfer}
            disabled={busy || (!edFile && !file)}
          >
            {busy ? "Running…" : "Run Inference"}
          </button>
        </div>
      ) : (
        <div className="flex flex-col md:flex-row items-start gap-3">
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium">Segmentation (.nii/.nii.gz)</label>
            <input
              type="file"
              accept=".nii,.nii.gz"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </div>
          <button
            className="bg-black text-white rounded px-4 py-2 disabled:opacity-50"
            onClick={onInfer}
            disabled={busy || !file}
          >
            {busy ? "Running…" : "Run Inference"}
          </button>
        </div>
      )}

      {/* Errors */}
      {error && (
        <div className="text-red-600 text-sm border border-red-200 bg-red-50 p-3 rounded">
          {error}
        </div>
      )}

      {/* Viewer */}
      <div className="border rounded-lg p-3">
        {organ === "heart" ? (
          <HeartMV
  segmentScores={result?.segment_scores ?? {}}
  topK={8}
  threshold={0.25}
/>
        ) : (
          <BrainMV affected={(affectedBrain ?? []).slice(0, 3)} />
        )}
      </div>

      {/* Results */}
      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h2 className="text-lg font-semibold mb-2">Prediction</h2>
            <pre className="text-sm bg-gray-100 p-3 rounded overflow-auto">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>

          <div>
            <h2 className="text-lg font-semibold mb-2">
              {organ === "heart" ? "AHA16 Segment Scores" : "Top Regions"}
            </h2>
            {organ === "heart" ? (
              <ul className="text-sm list-disc pl-6">
                {Object.entries(segScoresHeart)
                  .sort(([a], [b]) => Number(a) - Number(b))
                  .map(([sid, sc]) => (
                    <li key={sid}>
                      <b>SEG {sid}</b> — score {(Number(sc) || 0).toFixed(3)}
                    </li>
                  ))}
              </ul>
            ) : (
              <ul className="text-sm list-disc pl-6">
                {(affectedBrain ?? []).map((r) => (
                  <li key={r.label_id}>
                    <b>#{r.label_id}</b> {r.label_name} — score {r.score.toFixed(6)}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </main>
  );
}
