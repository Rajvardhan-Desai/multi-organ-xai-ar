"use client";

import React, { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";

// Dynamically load the viewer on the client only
const BrainMV = dynamic(() => import("../components/BrainMV"), { ssr: false });

const BACKEND = process.env.NEXT_PUBLIC_BACKEND ?? "http://127.0.0.1:8000";

type RegistryPayload = { organs: { organ: string; diseases: string[] }[] };
type TopRegion = { label_id: number; label_name: string; score: number };
type InferResponse = {
  prediction: string;
  proba: Record<string, number>;
  icv_mm3: number;
  used_features: string[];
  top_regions?: TopRegion[];
  xai?: { method: string; top_regions?: TopRegion[] };
};

export default function Page() {
  const [orgs, setOrgs] = useState<{ organ: string; diseases: string[] }[]>([]);
  const [organ, setOrgan] = useState("");
  const [disease, setDisease] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [xai, setXai] = useState(true);

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
          .filter(
            (x) => x && typeof x.organ === "string" && Array.isArray(x.diseases)
          )
          .map((x) => ({ organ: x.organ, diseases: x.diseases }));

        const fallback =
          normalized.length > 0
            ? normalized
            : [{ organ: "brain", diseases: ["alzheimer"] }];

        if (!ok) return;
        setOrgs(fallback);
        if (!organ && fallback.length) setOrgan(fallback[0].organ);
      } catch {
        if (!ok) return;
        const fallback = [{ organ: "brain", diseases: ["alzheimer"] }];
        setOrgs(fallback);
        if (!organ) setOrgan("brain");
      }
    })();
    return () => {
      ok = false;
    };
  }, []); // load once

  const diseases = useMemo(() => {
    const row = orgs.find((x) => x.organ === organ);
    return row?.diseases ?? [];
  }, [orgs, organ]);

  useEffect(() => {
    if (!diseases.includes(disease)) setDisease(diseases[0] ?? "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [diseases.length, organ]);

  const onInfer = async () => {
    if (!file || !organ || !disease) {
      setError("Please select organ, disease and choose a segmentation file.");
      return;
    }
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const url = `${BACKEND}/infer?organ=${encodeURIComponent(
        organ
      )}&disease=${encodeURIComponent(disease)}&xai=${xai ? "1" : "0"}`;
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

  const affected: TopRegion[] = useMemo(() => {
    if (!result) return [];
    return result?.xai?.top_regions?.length
      ? (result.xai.top_regions as TopRegion[])
      : result.top_regions ?? [];
  }, [result]);

  return (
    <main className="min-h-screen p-6 flex flex-col gap-6 bg-white">
      <h1 className="text-2xl font-semibold">Model Registry</h1>

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
            <span>Explainable AI (feature importance)</span>
          </label>
        </div>
      </div>

      <div className="flex flex-col md:flex-row items-start gap-3">
        <label className="text-sm font-medium">Segmentation (.nii.gz)</label>
        <input
          type="file"
          accept=".nii,.nii.gz"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <button
          className="bg-black text-white rounded px-4 py-2 disabled:opacity-50"
          onClick={onInfer}
          disabled={busy || !file}
        >
          {busy ? "Running…" : "Run Inference"}
        </button>
      </div>

      {error && (
        <div className="text-red-600 text-sm border border-red-200 bg-red-50 p-3 rounded">
          {error}
        </div>
      )}

      <div className="border rounded-lg p-3">
        <BrainMV
          affected={(
            result?.xai?.top_regions ??
            result?.top_regions ??
            []
          ).slice(0, 3)}
        />
      </div>

      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h2 className="text-lg font-semibold mb-2">Prediction</h2>
            <pre className="text-sm bg-gray-100 p-3 rounded overflow-auto">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
          <div>
            <h2 className="text-lg font-semibold mb-2">Top Regions</h2>
            <ul className="text-sm list-disc pl-6">
              {affected.map((r) => (
                <li key={r.label_id}>
                  <b>#{r.label_id}</b> {r.label_name} — score{" "}
                  {r.score.toFixed(6)}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </main>
  );
}
