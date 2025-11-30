"use client";

import React, { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";


const BrainMV = dynamic(() => import("../components/BrainMV"), { ssr: false });
const LiverMV = dynamic(() => import("../components/liverMV"), { ssr: false });
const HeartMV = dynamic(() => import("../components/HeartMV"), { ssr: false });

const BACKEND = process.env.NEXT_PUBLIC_BACKEND ?? "http://127.0.0.1:8000";

type RegistryPayload = { organs: { organ: string; diseases: string[] }[] };

type TopRegion = { label_id: number; label_name: string; score: number };
type InferResponse = {
  prediction: string;
  proba: Record<string, number>;
  used_features?: string[];
  icv_mm3?: number;
  top_regions?: TopRegion[];
  xai?: { method: string; top_regions?: TopRegion[] };
  segment_scores?: Record<string, number>;
};

export default function Page() {
  const [orgs, setOrgs] = useState<{ organ: string; diseases: string[] }[]>([]);
  const [organ, setOrgan] = useState("");
  const [disease, setDisease] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [edFile, setEdFile] = useState<File | null>(null);
  const [esFile, setEsFile] = useState<File | null>(null);
  const [xai, setXai] = useState(true);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<InferResponse | null>(null);

  // ---------------------------------------------------------------------------
  // Load /registry once and set fallback for local dev
  // ---------------------------------------------------------------------------
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
            : [
                { organ: "brain", diseases: ["alzheimer"] },
                { organ: "heart", diseases: ["cardiomyopathy"] },
                { organ: "liver", diseases: ["fibrosis"] },
              ];

        if (!ok) return;
        setOrgs(fallback);
        if (!organ && fallback.length) setOrgan(fallback[0].organ);
      } catch {
        if (!ok) return;
        const fallback = [
          { organ: "brain", diseases: ["alzheimer"] },
          { organ: "heart", diseases: ["cardiomyopathy"] },
          { organ: "liver", diseases: ["fibrosis"] },
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

  // Reset state when organ (or its disease list) changes
  useEffect(() => {
    if (!diseases.includes(disease)) setDisease(diseases[0] ?? "");
    setResult(null);
    setError(null);
    setFile(null);
    setEdFile(null);
    setEsFile(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [diseases.length, organ]);

  // ---------------------------------------------------------------------------
  // Inference
  // ---------------------------------------------------------------------------
  const onInfer = async () => {
    setBusy(true);
    setError(null);
    setResult(null);

    try {
      const qs = new URLSearchParams({ organ, disease, xai: xai ? "1" : "0" });
      const fd = new FormData();

      if (organ === "heart") {
        if (!edFile && !file)
          throw new Error("Please choose ED mask (.nii/.nii.gz) for heart.");
        if (edFile) fd.append("ed_file", edFile);
        if (esFile) fd.append("es_file", esFile ?? edFile ?? (file as File));
        if (!edFile && file) fd.append("file", file);
      } else {
        if (!file)
          throw new Error(
            organ === "brain"
              ? "Please choose a segmentation file (.nii/.nii.gz)."
              : "Please choose a CT/NIfTI volume (.nii/.nii.gz)."
          );
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

  // ---------------------------------------------------------------------------
  // Derived data
  // ---------------------------------------------------------------------------
  const affectedBrain: TopRegion[] = useMemo(() => {
    if (!result) return [];
    return result?.xai?.top_regions?.length
      ? (result.xai.top_regions as TopRegion[])
      : result.top_regions ?? [];
  }, [result]);

  const segmentScores = useMemo(
    () => (result?.segment_scores ? result.segment_scores : {}),
    [result]
  );

  // Prediction probabilities chart
  const probChartData = useMemo(() => {
    if (!result?.proba) return [];
    return Object.entries(result.proba)
      .map(([label, value]) => ({
        label,
        percent: (value || 0) * 100,
      }))
      .sort((a, b) => b.percent - a.percent);
  }, [result]);

  // Top regions / segments chart
  const topRegionChartData = useMemo(() => {
    if (organ === "heart" || organ === "liver") {
      const entries = Object.entries(segmentScores || {});
      return entries
        .map(([sid, sc]) => ({
          name: `SEG ${sid}`,
          id: sid,
          score: Number(sc) || 0,
        }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 10);
    }

    const regions = affectedBrain || [];
    return regions
      .map((r) => ({
        name: `#${r.label_id}`,
        label: r.label_name,
        score: r.score,
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 10);
  }, [organ, affectedBrain, segmentScores]);

  const confidenceDescription = useMemo(() => {
    if (!result?.prediction || !result?.proba) return "";
    const p = result.proba[result.prediction] ?? 0;
    if (p >= 0.85) return "high confidence";
    if (p >= 0.7) return "moderate confidence";
    if (p >= 0.55) return "borderline confidence";
    return "low confidence";
  }, [result]);

  // ---------------------------------------------------------------------------
  // Narrative disease analysis
  // ---------------------------------------------------------------------------
  const diseaseAnalysis = useMemo(() => {
    if (!result) return "";
    const pred = result.prediction;
    const proba = result.proba || {};
    const predProb = proba[pred] ?? 0;

    if (organ === "brain" && disease.toLowerCase() === "alzheimer") {
      const cnProb = proba["CN"];
      const adProb = proba["AD"];
      return [
        `For the selected task (brain – Alzheimer), the model predicts **${pred}** with a probability of ${(
          predProb * 100
        ).toFixed(1)}%.`,
        cnProb !== undefined && adProb !== undefined
          ? `The relative probabilities are CN: ${(cnProb * 100).toFixed(
              1
            )}%, AD: ${(adProb * 100).toFixed(1)}%. This suggests a ${
              cnProb > adProb ? "slightly higher" : "slightly lower"
            } likelihood of being classified as cognitively normal compared to Alzheimer’s disease.`
          : "",
        affectedBrain.length
          ? `The most influential regions for this prediction include **${affectedBrain
              .slice(0, 3)
              .map((r) => r.label_name)
              .join(", ")}**, according to the explainability scores.`
          : "",
        "These outputs are intended for research/decision-support only and are **not** a standalone clinical diagnosis.",
      ]
        .filter(Boolean)
        .join(" ");
    }

    if (organ === "heart") {
      const segCount = Object.keys(segmentScores).length;
      return [
        `For the selected task (heart – ${
          disease || "unknown disease"
        }), the model predicts **${pred}** with a probability of ${(
          predProb * 100
        ).toFixed(1)}%.`,
        segCount
          ? `A total of ${segCount} AHA16 segments have associated scores, indicating which myocardial regions contributed more strongly to this prediction.`
          : "",
        "Interpretation of these findings should be done together with clinical and imaging information.",
      ]
        .filter(Boolean)
        .join(" ");
    }

    if (organ === "liver") {
      const segCount = Object.keys(segmentScores).length;
      return [
        `For the selected task (liver – ${
          disease || "fibrosis"
        }), the model predicts **${pred}** with a probability of ${(
          predProb * 100
        ).toFixed(1)}%.`,
        segCount
          ? `Segment-level scores over pseudo-Couinaud regions (I–VIII) highlight which parts of the liver contributed most strongly to the prediction.`
          : "",
        "These findings should be interpreted in the context of the full CT study and clinical history rather than in isolation.",
      ]
        .filter(Boolean)
        .join(" ");
    }

    return `The model predicts **${pred}** with a probability of ${(
      predProb * 100
    ).toFixed(
      1
    )}%. This should be interpreted in the context of additional clinical and imaging information.`;
  }, [result, organ, disease, affectedBrain, segmentScores]);

  // ---------------------------------------------------------------------------
  // Predictive analysis narrative
  // ---------------------------------------------------------------------------
  const predictiveAnalysis = useMemo(() => {
    if (!result) return "";
    const pred = result.prediction;
    const proba = result.proba || {};
    const entries = Object.entries(proba).sort((a, b) => b[1] - a[1]);
    const alt = entries.find(([label]) => label !== pred);
    const predProb = proba[pred] ?? 0;

    const base = `The model’s decision boundary currently favors **${pred}** (${(
      predProb * 100
    ).toFixed(1)}% probability).`;

    const altText = alt
      ? ` The next most likely class is **${alt[0]}** with a probability of ${(
          alt[1] * 100
        ).toFixed(
          1
        )}%, indicating that small changes in input data or model parameters could potentially shift the classification in borderline cases.`
      : "";

    let xaiText: string;
    if (organ === "heart" || organ === "liver") {
      xaiText =
        " Segment-level scores highlight which anatomical regions most strongly influenced the decision, supporting regional risk and pattern analysis.";
    } else {
      xaiText =
        " Region-level explainability scores highlight which brain structures most strongly influenced the decision, which can help relate the prediction to known disease patterns.";
    }

    const caution =
      " These predictions are probabilistic and should not be used as the sole basis for treatment decisions.";

    return base + altText + " " + xaiText + caution;
  }, [result, organ]);

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <main className="min-h-screen p-6 flex flex-col gap-6 bg-white">
      <h1 className="text-2xl font-semibold">EX-AI-AR Multi-organ</h1>

      {/* Controls */}
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

        <div className="flex items-end justify-between gap-3">
          <label className="inline-flex items-center gap-2">
            <input
              type="checkbox"
              checked={xai}
              onChange={(e) => setXai(e.target.checked)}
            />
            <span>Explainable AI</span>
          </label>

          {organ === "heart" && (
            <button
              type="button"
              className="text-xs border px-2 py-1 rounded"
              onClick={() => (window as any).heartDump?.()}
              title="Logs all GLB names to console"
            >
              Debug: Dump GLB names
            </button>
          )}
        </div>
      </div>

      {/* Results */}
      {result && (
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Column 1: Prediction + prob chart + raw JSON */}
          <div className="space-y-3 min-w-0">
            <h2 className="text-lg font-semibold">Prediction</h2>

            <div className="text-sm space-y-1">
              <div>
                Overall prediction:{" "}
                <span className="font-semibold">{result.prediction}</span>{" "}
                <span className="text-xs text-gray-500">
                  ({confidenceDescription || "confidence not available"})
                </span>
              </div>
            </div>

            {/* Class probabilities chart (fixed size to avoid width=-1 errors) */}
            {probChartData.length > 0 && (
              <div className="border rounded p-3 bg-gray-50 overflow-auto">
                <div className="text-xs font-medium text-gray-600 mb-2">
                  Class probabilities
                </div>
                <BarChart width={360} height={190} data={probChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="label" />
                  <YAxis unit="%" />
                  <Tooltip
                    formatter={(value: any) => `${Number(value).toFixed(2)} %`}
                  />
                  <Bar dataKey="percent" />
                </BarChart>
              </div>
            )}

            {/* Raw JSON for debugging */}
            <pre className="text-xs bg-gray-100 p-3 rounded overflow-auto">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>

          {/* Column 2: Top regions / segments + chart */}
          <div className="space-y-3 min-w-0">
            <h2 className="text-lg font-semibold">
              {organ === "heart"
                ? "AHA16 Segment Scores"
                : organ === "liver"
                ? "Liver Segment Scores"
                : "Top Regions"}
            </h2>

            {/* Top regions / segments chart (fixed size) */}
            {topRegionChartData.length > 0 && (
              <div className="border rounded p-3 bg-gray-50 overflow-auto">
                <div className="text-xs font-medium text-gray-600 mb-2">
                  {organ === "heart"
                    ? "Top segments by score"
                    : organ === "liver"
                    ? "Top liver segments by score"
                    : "Top brain regions by XAI score"}
                </div>
                <BarChart
                  width={380}
                  height={220}
                  data={topRegionChartData}
                  layout="vertical"
                  margin={{ left: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis
                    type="category"
                    dataKey="name"
                    width={70}
                    tick={{ fontSize: 10 }}
                  />
                  <Tooltip
                    formatter={(value: any) =>
                      Number(value).toFixed(4).toString()
                    }
                    labelFormatter={(label: any, payload: any) => {
                      const item = payload?.[0]?.payload as any;
                      if (organ === "heart" || organ === "liver") {
                        return label;
                      }
                      return `${label} – ${item?.label ?? ""}`;
                    }}
                  />
                  <Bar dataKey="score" />
                </BarChart>
              </div>
            )}

            {/* Textual list */}
            {organ === "brain" ? (
              <ul className="text-sm list-disc pl-6">
                {(affectedBrain ?? []).map((r) => (
                  <li key={r.label_id}>
                    <b>#{r.label_id}</b> {r.label_name} — score{" "}
                    {r.score.toFixed(6)}
                  </li>
                ))}
              </ul>
            ) : (
              <ul className="text-sm list-disc pl-6">
                {Object.entries(result?.segment_scores ?? {})
                  .sort(([a], [b]) => Number(a) - Number(b))
                  .map(([sid, sc]) => (
                    <li key={sid}>
                      <b>SEG {sid}</b> — score {(Number(sc) || 0).toFixed(3)}
                    </li>
                  ))}
              </ul>
            )}
          </div>

          {/* Column 3: Disease + predictive analysis */}
          <div className="space-y-4 min-w-0">
            <div>
              <h2 className="text-lg font-semibold mb-2">Disease Analysis</h2>
              <div className="text-sm bg-gray-50 border rounded p-3">
                {diseaseAnalysis || "No analysis available for this result."}
              </div>
            </div>

            <div>
              <h2 className="text-lg font-semibold mb-2">
                Predictive Analysis
              </h2>
              <div className="text-sm bg-gray-50 border rounded p-3">
                {predictiveAnalysis ||
                  "No predictive analysis available for this result."}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* File inputs + Run button */}
      {organ === "heart" ? (
        <div className="flex flex-col lg:flex-row items-start gap-4">
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium">
              ED mask (.nii/.nii.gz)
            </label>
            <input
              type="file"
              accept=".nii,.nii.gz"
              onChange={(e) => setEdFile(e.target.files?.[0] ?? null)}
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium">
              ES mask (.nii/.nii.gz)
            </label>
            <input
              type="file"
              accept=".nii,.nii.gz"
              onChange={(e) => setEsFile(e.target.files?.[0] ?? null)}
            />
            <div className="text-xs text-gray-500">
              Optional — if omitted, ED will be reused.
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
            <label className="text-sm font-medium">
              {organ === "brain"
                ? "Segmentation (.nii/.nii.gz)"
                : "CT volume (.nii/.nii.gz)"}
            </label>
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

      {/* Error */}
      {error && (
        <div className="text-red-600 text-sm border border-red-200 bg-red-50 p-3 rounded">
          {error}
        </div>
      )}

      {/* 3D model view */}
      <div className="border rounded-lg p-3">
        {organ === "liver" ? (
          <LiverMV
            regionScores={result?.segment_scores ?? {}}
            topK={8}
            threshold={0.2}
            overallPrediction={result?.prediction}
          />
        ) : organ === "heart" ? (
          <HeartMV
            segmentScores={result?.segment_scores ?? {}}
            topK={8}
            threshold={0.2}
            overallPrediction={result?.prediction || undefined}
          />
        ) : (
          <BrainMV
            affected={affectedBrain ?? []}
            topK={8}
            threshold={0.1}
            opacity={0.25}
            overallPrediction={result?.prediction || undefined}
          />
        )}
      </div>
    </main>
  );
}
