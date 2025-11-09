'use client';

import dynamic from 'next/dynamic';
import { useCallback, useEffect, useMemo, useState } from 'react';

const BrainMV = dynamic(() => import('../components/BrainMV'), { ssr: false });

type Registry = { organs: { organ: string; diseases: string[] }[] };
type TopRegion = { label_id: number; label_name: string; score: number };
type InferResp = {
  prediction: string;
  proba: Record<string, number>;
  icv_mm3: number;
  used_features: string[];
  top_regions: TopRegion[];
  xai?: { method?: string; top_regions?: TopRegion[] | any };
};

type MappingEntry = { target: string; side?: 'L' | 'R' };
type Mapping = Record<number, MappingEntry[]>;

const API_URL = process.env.NEXT_PUBLIC_API || 'http://localhost:8000';

export default function Page() {
  const [orgs, setOrgs] = useState<Registry['organs']>([]);
  const [organ, setOrgan] = useState('');
  const [disease, setDisease] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [useXai, setUseXai] = useState(true);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<InferResp | null>(null);

  const [mapping, setMapping] = useState<Mapping>({}); // optional mapping for GLB node names

  // Load registry
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`${API_URL}/registry`);
        const json: Registry = await r.json();
        if (cancelled) return;

        const list = Array.isArray(json?.organs) ? json.organs : [];
        setOrgs(list);

        // sensible defaults
        if (list.length) {
          setOrgan((prev) => prev || list[0].organ);
          const firstDis = list[0].diseases?.[0] || '';
          setDisease((prev) => prev || firstDis);
        }
      } catch (e: any) {
        setError(`Failed to load registry: ${e?.message || e}`);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Keep disease in sync when organ changes
  const diseases = useMemo(() => {
    const o = orgs.find((x) => x.organ === organ);
    return o?.diseases || [];
  }, [orgs, organ]);

  useEffect(() => {
    if (diseases.length && !disease) setDisease(diseases[0]);
    if (diseases.length && !diseases.includes(disease)) setDisease(diseases[0]);
  }, [diseases, disease]);

  // Optional: load mapping file if you placed one at /public/static/mapping.json
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch('/static/mapping.json', { cache: 'no-store' });
        if (!res.ok) return; // file optional
        const m = (await res.json()) as Mapping;
        if (!cancelled) setMapping(m || {});
      } catch {
        // ignore — mapping is optional
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // After inference, auto-highlight all affected regions and fade others
  const spotlight = useCallback((resp: InferResp | null) => {
    if (!resp) return;
    const src = resp.top_regions?.length ? resp.top_regions : resp.xai?.top_regions || [];
    const ids = Array.isArray(src) ? src.map((r: any) => r.label_id).filter(Boolean) : [];

    const names = Array.isArray(src) ? src.map((r: any) => r.label_name).filter(Boolean) : [];

    const fire = () =>
      window.dispatchEvent(
        new CustomEvent('highlight-rois', {
          detail: {
            label_ids: ids,
            label_names: names,
            exclusive: true, // dim all others
          },
        })
      );

    // If viewer has already loaded:
    if ((window as any).__brainReady) fire();
    else window.addEventListener('brain-ready', () => fire(), { once: true });
  }, []);

  // Submit for inference
  const onInfer = useCallback(async () => {
    setError(null);
    setResult(null);
    if (!file) {
      setError('Please choose a .nii.gz segmentation file.');
      return;
    }
    if (!organ || !disease) {
      setError('Select organ and disease first.');
      return;
    }

    try {
      setLoading(true);
      const fd = new FormData();
      fd.append('file', file, file.name);

      const res = await fetch(
        `${API_URL}/infer?organ=${encodeURIComponent(organ)}&disease=${encodeURIComponent(
          disease
        )}&xai=${useXai ? 1 : 0}`,
        { method: 'POST', body: fd }
      );

      if (!res.ok) {
        const msg = await res.text();
        throw new Error(`Backend error ${res.status}: ${msg}`);
      }
      const json = (await res.json()) as InferResp;
      setResult(json);
      // kick the 3D highlight
      spotlight(json);
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }, [file, organ, disease, useXai, spotlight]);

  return (
    <main className="min-h-screen bg-slate-100 text-slate-900">
      <div className="max-w-6xl mx-auto px-4 py-6 space-y-6">
        <h1 className="text-2xl font-semibold">Model Registry</h1>

        {/* Controls */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Organ */}
          <div className="flex flex-col gap-1">
            <label className="text-sm text-slate-600">Organ</label>
            <select
              value={organ}
              onChange={(e) => setOrgan(e.target.value)}
              className="border rounded px-3 py-2 bg-white"
            >
              {orgs.map((o) => (
                <option key={o.organ} value={o.organ}>
                  {o.organ}
                </option>
              ))}
            </select>
          </div>

          {/* Disease */}
          <div className="flex flex-col gap-1">
            <label className="text-sm text-slate-600">Disease</label>
            <select
              value={disease}
              onChange={(e) => setDisease(e.target.value)}
              className="border rounded px-3 py-2 bg-white"
            >
              {diseases.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </div>

          {/* XAI */}
          <div className="flex items-end">
            <label className="inline-flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={useXai}
                onChange={(e) => setUseXai(e.target.checked)}
              />
              Explainable AI (feature importance)
            </label>
          </div>
        </div>

        {/* Upload + Infer */}
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex-1">
            <div className="text-sm text-slate-600 mb-1">Segmentation (.nii.gz)</div>
            <input
              type="file"
              accept=".nii,.nii.gz"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="block w-full"
            />
          </label>

          <button
            onClick={onInfer}
            disabled={loading || !file || !organ || !disease}
            className="px-4 py-2 rounded bg-indigo-600 text-white disabled:bg-indigo-300"
          >
            {loading ? 'Running…' : 'Run Inference'}
          </button>

          {file && (
            <div className="text-xs text-slate-600 truncate max-w-[40ch]">{file.name}</div>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="px-3 py-2 rounded bg-red-50 text-red-700 border border-red-200">
            {error}
          </div>
        )}

        {/* 3D viewer */}
        <BrainMV mapping={mapping} />

        {/* JSON result */}
        <div className="bg-white border rounded-xl p-4">
          <div className="text-sm font-medium mb-2">Prediction</div>
          <pre className="text-xs overflow-auto">
            {result ? JSON.stringify(result, null, 2) : '—'}
          </pre>
        </div>
      </div>
    </main>
  );
}
