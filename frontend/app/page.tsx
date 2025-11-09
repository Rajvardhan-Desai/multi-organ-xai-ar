"use client";

import { useEffect, useMemo, useState } from "react";

type Registry = { organs: { organ: string; diseases: string[] }[] };
type Proba = Record<string, number>;
type PredictionResponse = {
  prediction: string;
  proba: Proba;
  icv_mm3: number;
  used_features: string[];
  top_regions: { label_id: number; label_name: string; score: number }[];
  xai: null | { method: string; top_regions: { label_id: number; label_name: string; score: number }[] };
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "") || "http://localhost:8000";

export default function Page() {
  const [orgs, setOrgs] = useState<Registry["organs"]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const [organ, setOrgan] = useState<string>("");
  const [disease, setDisease] = useState<string>("");

  const [file, setFile] = useState<File | null>(null);
  const [xai, setXai] = useState<boolean>(true);

  const [pred, setPred] = useState<PredictionResponse | null>(null);
  const [inferring, setInferring] = useState(false);
  const [inferErr, setInferErr] = useState<string | null>(null);

  // Fetch registry on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setErr(null);
      try {
        const res = await fetch(`${API_BASE}/registry`, { cache: "no-store" });
        if (!res.ok) throw new Error(`GET /registry ${res.status}`);
        const json = (await res.json()) as unknown;

        const parsed: Registry = {
          organs: Array.isArray((json as any)?.organs) ? (json as any).organs : [],
        };

        if (!cancelled) {
          setOrgs(parsed.organs);
          // Auto-select first available organ/disease
          if (parsed.organs.length > 0) {
            setOrgan(parsed.organs[0].organ);
            const firstDiseases = parsed.organs[0].diseases || [];
            if (firstDiseases.length > 0) setDisease(firstDiseases[0]);
          }
        }
      } catch (e: any) {
        if (!cancelled) setErr(e?.message || "Failed to load registry");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Keep disease in sync when organ changes
  useEffect(() => {
    const entry = orgs.find((o) => o.organ === organ);
    const ds = entry?.diseases ?? [];
    if (ds.length > 0) {
      if (!ds.includes(disease)) setDisease(ds[0]);
    } else {
      setDisease("");
    }
  }, [organ, orgs]); // eslint-disable-line react-hooks/exhaustive-deps

  const diseasesForSelected = useMemo(() => {
    return orgs.find((o) => o.organ === organ)?.diseases ?? [];
  }, [orgs, organ]);

  const onInfer = async () => {
    setInferErr(null);
    setPred(null);
    if (!file) {
      setInferErr("Please choose a .nii.gz segmentation file.");
      return;
    }
    if (!organ || !disease) {
      setInferErr("Please select an organ and disease.");
      return;
    }
    setInferring(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const url = `${API_BASE}/infer?organ=${encodeURIComponent(organ)}&disease=${encodeURIComponent(
        disease
      )}&xai=${xai ? "1" : "0"}`;
      const res = await fetch(url, { method: "POST", body: form });
      const json = await res.json();
      if (!res.ok) {
        throw new Error(json?.detail || "Inference failed");
      }
      setPred(json as PredictionResponse);
    } catch (e: any) {
      setInferErr(e?.message || "Inference failed");
    } finally {
      setInferring(false);
    }
  };

  return (
    <main className="p-6 max-w-5xl mx-auto">
      <h1 className="text-2xl font-semibold mb-4">Model Registry</h1>

      {/* Registry state */}
      {loading && <div className="mb-4 text-sm text-gray-500">Loading registry…</div>}
      {err && (
        <div className="mb-4 text-sm text-red-600">
          {err} &nbsp; <code className="text-xs">API_BASE={API_BASE}</code>
        </div>
      )}
      {!loading && !err && orgs.length === 0 && (
        <div className="mb-4 text-sm text-amber-600">No models registered.</div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
        {/* Organ */}
        <div>
          <label className="block text-sm mb-1">Organ</label>
          <select
            className="border rounded px-3 py-2 w-full"
            value={organ}
            onChange={(e) => setOrgan(e.target.value)}
            disabled={orgs.length === 0}
          >
            {orgs.map((o) => (
              <option key={o.organ} value={o.organ}>
                {o.organ}
              </option>
            ))}
          </select>
        </div>

        {/* Disease */}
        <div>
          <label className="block text-sm mb-1">Disease</label>
          <select
            className="border rounded px-3 py-2 w-full"
            value={disease}
            onChange={(e) => setDisease(e.target.value)}
            disabled={diseasesForSelected.length === 0}
          >
            {diseasesForSelected.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </div>

        {/* XAI toggle */}
        <div className="flex items-center gap-2">
          <input id="xai" type="checkbox" checked={xai} onChange={(e) => setXai(e.target.checked)} />
          <label htmlFor="xai" className="text-sm">
            Explainable AI (feature importance)
          </label>
        </div>
      </div>

      {/* File input */}
      <div className="mt-6">
        <label className="block text-sm mb-1">Segmentation (.nii.gz)</label>
        <input
          type="file"
          accept=".nii,.nii.gz"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="border rounded px-3 py-2 w-full"
        />
      </div>

      {/* Infer */}
      <div className="mt-4 flex gap-3">
        <button
          onClick={onInfer}
          disabled={!file || !organ || !disease || inferring}
          className="bg-black text-white rounded px-4 py-2 disabled:opacity-50"
        >
          {inferring ? "Running…" : "Run Inference"}
        </button>
        {inferErr && <span className="text-sm text-red-600">{inferErr}</span>}
      </div>

      {/* Result */}
      {pred && (
        <section className="mt-8">
          <h2 className="text-xl font-semibold mb-2">Prediction</h2>
          <pre className="text-xs bg-gray-100 p-3 rounded overflow-auto">
{JSON.stringify(pred, null, 2)}
          </pre>
        </section>
      )}
    </main>
  );
}
