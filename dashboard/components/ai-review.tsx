"use client";

import { useEffect, useState } from "react";
import { Brain, RefreshCw } from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
type Review = { text: string; model: string; created_at: string; symbol: string; candle_time: string | null; truncated: boolean };
type Status = { configured: boolean; model: string; running: boolean; latest: Review | null };

export function AIReview() {
  const [status, setStatus] = useState<Status | null>(null);
  const [review, setReview] = useState<Review | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API}/api/ai-review`, { signal: controller.signal, cache: "no-store" })
      .then(async r => { if (!r.ok) throw new Error("AI review status unavailable."); return r.json() as Promise<Status>; })
      .then(s => { setStatus(s); setReview(s.latest); })
      .catch(e => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, []);
  async function generate() {
    setBusy(true); setError("");
    try {
      const response = await fetch(`${API}/api/ai-review`, { method: "POST", signal: AbortSignal.timeout(70000) });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || "Review failed.");
      setReview(body);
    } catch (e) { setError(e instanceof Error ? e.message : "Review failed."); }
    finally { setBusy(false); }
  }
  return <section className="panel history" aria-label="AI trading review">
    <div className="panelHead"><div><span className="eyebrow"><Brain size={15} /> OPENROUTER</span><h2>Explain my trading</h2></div>
      <button className="navButton" onClick={() => void generate()} disabled={busy || status?.configured === false}>
        <RefreshCw size={15} className={busy ? "spin" : ""} />{busy ? "Reviewing…" : "Generate review"}
      </button>
    </div>
    <p className="reason">Sends the current signal, risk summary and up to 10 recent paper trades to OpenRouter and its model provider. Reviews explain recorded data; they cannot place trades or change risk controls.</p>
    <p className="reason" style={{ fontSize: 12 }}>Model: {status?.model || "Loading…"} · Runs only when requested · One request per minute</p>
    {status?.configured === false && <p role="status">Configure OPENROUTER_API_KEY in the backend environment and restart the backend.</p>}
    {error && <p role="alert" style={{ color: "#ffb19e" }}>{error}</p>}
    {review ? <div>
      <p className="reason" style={{ fontSize: 12 }}>{review.symbol} · Reviewed {new Date(review.created_at).toLocaleString()} · {review.model}<br />Signal candle: {review.candle_time ? new Date(review.candle_time).toLocaleString() : "Unavailable"}</p>
      <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.8, overflowWrap: "anywhere" }}>{review.text}</div>
      {review.truncated && <p className="reason">This response reached its length limit and may be incomplete.</p>}
      <p className="reason" style={{ fontSize: 12 }}>AI commentary can be inaccurate. This review describes the snapshot above, not subsequent market changes.</p>
    </div> : <p className="reason">No review generated yet. Your strategy and local prediction model continue operating independently.</p>}
  </section>;
}
