"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ArrowLeft, RefreshCw, Activity, ArrowUpRight, ArrowDownRight } from "lucide-react";
import styles from "./pairs-dashboard.module.css";

type Pair = {
  symbol: string; digits: number; active: boolean; status: "fresh" | "stale" | "unavailable";
  message: string; feed?: string; price?: number; candle_time?: string; day_start?: string;
  change_percent?: number; change_pips?: number; trend?: string; atr_pips?: number; rsi?: number;
  series?: { time: string; price: number }[];
  decision?: { action: string; reason: string; candle_time: string } | null;
  paper?: { closed: number; wins: number; open: number; realized_pnl: number } | null;
};
type Overview = { as_of: string; interval: string; strategy: string; journal_available: boolean; pairs: Pair[] };
const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const number = (value?: number, digits = 2) => value == null ? "—" : value.toFixed(digits);
const signed = (value?: number, digits = 2) => value == null ? "—" : `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
const stamp = (value?: string) => value ? new Date(value).toLocaleString() : "—";

function Chart({ pair }: { pair: Pair }) {
  const [hover, setHover] = useState<number | null>(null);
  const series = pair.series || [];
  if (series.length < 2) return <div className={styles.empty}>Chart unavailable for this pair.</div>;
  const lo = Math.min(...series.map(p => p.price)), hi = Math.max(...series.map(p => p.price));
  const pad = Math.max((hi - lo) * .1, 10 ** -pair.digits);
  const y = (p: number) => 190 - (p - lo + pad) / (hi - lo + pad * 2) * 170;
  const x = (i: number) => 20 + i / (series.length - 1) * 740;
  const points = series.map((p, i) => `${x(i)},${y(p.price)}`).join(" ");
  const index = hover ?? series.length - 1;
  return <div>
    <div className={styles.chartReadout}><strong>{number(series[index].price, pair.digits)}</strong><span>{stamp(series[index].time)} · candle close</span></div>
    <svg className={styles.chart} viewBox="0 0 780 220" role="img" aria-label={`${pair.symbol} recent closing prices`}
      onPointerMove={e => { const r = e.currentTarget.getBoundingClientRect(); setHover(Math.max(0, Math.min(series.length - 1, Math.round(((e.clientX - r.left) / r.width * 780 - 20) / 740 * (series.length - 1))))); }}
      onPointerLeave={() => setHover(null)}>
      {[40, 90, 140, 190].map(v => <line key={v} x1="20" x2="760" y1={v} y2={v} stroke="#24364c" strokeDasharray="4 6" />)}
      <polygon points={`20,205 ${points} 760,205`} fill="#39cbb015" />
      <polyline points={points} fill="none" stroke="#54ddc2" strokeWidth="2.5" />
      <line x1={x(index)} x2={x(index)} y1="15" y2="205" stroke="#91a4bd" strokeDasharray="3 4" />
      <circle cx={x(index)} cy={y(series[index].price)} r="4" fill="#fff" />
      <text x="24" y="15" fill="#91a4bd" fontSize="11">High {number(hi, pair.digits)}</text>
      <text x="24" y="218" fill="#91a4bd" fontSize="11">Low {number(lo, pair.digits)}</text>
    </svg>
    <div className={styles.chartEnds}><span>{stamp(series[0].time)}</span><span>{stamp(series.at(-1)?.time)}</span></div>
  </div>;
}

export function PairsDashboard() {
  const [data, setData] = useState<Overview | null>(null);
  const [selected, setSelected] = useState("EUR/USD");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [now, setNow] = useState(Date.now());
  const request = useRef<AbortController | null>(null);
  async function refresh() {
    if (request.current) return;
    const controller = new AbortController(); request.current = controller; setBusy(true);
    const timeout = setTimeout(() => controller.abort(), 30000);
    try {
      const response = await fetch(`${API}/api/pairs`, { cache: "no-store", signal: controller.signal });
      if (!response.ok) throw new Error(`Pair monitor unavailable (${response.status})`);
      const next: Overview = await response.json();
      if (!controller.signal.aborted && request.current === controller) { setData(next); setError(""); setNow(Date.now()); }
    } catch (e) {
      if (request.current === controller) setError(e instanceof Error && e.name !== "AbortError" ? e.message : "Refresh timed out. Displayed prices may be old.");
    } finally { clearTimeout(timeout); if (request.current === controller) { request.current = null; setBusy(false); } }
  }
  useEffect(() => {
    void refresh();
    const interval = setInterval(() => { if (!document.hidden) void refresh(); }, 300000);
    const clock = setInterval(() => setNow(Date.now()), 15000);
    return () => { clearInterval(interval); clearInterval(clock); request.current?.abort(); request.current = null; };
  }, []);
  const expired = !!data && now - new Date(data.as_of).getTime() > 360000;
  const unreliable = !!error || expired;
  const pair = data?.pairs.find(p => p.symbol === selected);
  return <main className={styles.page}>
    <nav className={styles.nav}><Link href="/"><ArrowLeft size={16} /> Paper trading</Link><span><Activity size={16} /> MARKET WATCH</span><button onClick={() => void refresh()} disabled={busy}><RefreshCw size={16} className={busy ? "spin" : ""} />{busy ? "Refreshing…" : "Refresh"}</button></nav>
    <section className={styles.hero}><div><span className={styles.eyebrow}>FOREX / PAIR COMPARISON</span><h1>Three pairs.<br />One clear view.</h1><p>Follow price movement and compare recorded paper results.</p></div><div className={styles.update}><b>{data?.interval || "15min"} candles</b><span>Refreshes every 5 minutes</span><span>Last refresh: {stamp(data?.as_of)}</span></div></section>
    {(error || expired) && <div className={styles.warning} role="alert">{error || "This view has not refreshed recently. Refresh before using these prices."}</div>}
    {!data && <div className={styles.empty} role="status">{busy ? "Loading pair prices…" : "No prices loaded. Use Refresh to try again."}</div>}
    <section className={styles.cards} aria-label="Select a currency pair">{data?.pairs.map(p => <button key={p.symbol} className={`${styles.card} ${selected === p.symbol ? styles.selected : ""}`} onClick={() => setSelected(p.symbol)} aria-pressed={selected === p.symbol}>
      <div className={styles.cardTop}><b>{p.symbol}</b><span>{p.active ? "Paper engine" : "Monitoring only"}</span></div>
      <strong className={styles.price}>{number(p.price, p.digits)}</strong>
      <div className={(p.change_percent ?? 0) >= 0 ? styles.positive : styles.negative}>{(p.change_percent ?? 0) >= 0 ? <ArrowUpRight size={18} /> : <ArrowDownRight size={18} />}{signed(p.change_percent)}% <small>({signed(p.change_pips, 1)} pips)</small></div>
      <p>{unreliable ? "Refresh needed" : p.status === "fresh" ? `${p.feed === "synthetic" ? "Synthetic preview" : "Price available"}` : p.status === "stale" ? "Stale price" : "Data unavailable"}</p>
      <small>{p.day_start ? `Since ${stamp(p.day_start)}` : p.message}</small>
    </button>)}</section>
    {pair && <section className={styles.detail}>
      <article className={styles.panel}><div className={styles.panelTop}><h2>{pair.symbol} price history</h2><span>Last {pair.series?.length || 0} candles</span></div>
        {(pair.status !== "fresh" || unreliable) && <p className={styles.warning}>Historical view · {unreliable ? "Refresh needed" : pair.message}</p>}
        <Chart key={`${pair.symbol}:${data?.as_of}`} pair={pair} /><p className={styles.note}>Closing prices, including the latest forming candle. Times use your device timezone.</p>
      </article>
      <aside className={styles.panel}><h2>Market context</h2><p className={styles.note}>Indicators describe the market; they are not entry approvals.</p><dl className={styles.facts}><div><dt>EMA trend</dt><dd>{pair.trend || "—"}</dd></div><div><dt>RSI · closed candle</dt><dd>{number(pair.rsi, 1)}</dd></div><div><dt>ATR · typical range</dt><dd>{number(pair.atr_pips, 1)} pips</dd></div></dl>
        <div className={styles.decision}><b>{unreliable || pair.status !== "fresh" ? "Decision unavailable" : pair.decision?.action || "Monitoring only"}</b><p>{unreliable || pair.status !== "fresh" ? "A fresh view is needed." : pair.decision?.reason || "No trading decision is assigned to this view. Other pairs need their own validated strategy and execution settings."}</p>{pair.decision && <small>Decision candle: {stamp(pair.decision.candle_time)}</small>}</div>
      </aside>
    </section>}
    <section className={styles.panel}><div className={styles.panelTop}><h2>Recorded paper performance</h2><span>Actual journal records · USD</span></div><p className={styles.note}>Price movement above is not strategy profit. A pair without recorded trades has no performance result yet.</p>
      <div className={styles.tableWrap}><table><thead><tr><th>Pair</th><th>Closed trades</th><th>Win rate</th><th>Realized P&amp;L</th><th>Open trades</th></tr></thead><tbody>{data?.pairs.map(p => <tr key={p.symbol}><th>{p.symbol}</th><td>{p.paper?.closed ?? "Not tested"}</td><td>{p.paper?.closed ? `${(p.paper.wins / p.paper.closed * 100).toFixed(1)}%` : "—"}</td><td>{p.paper ? `$${p.paper.realized_pnl.toFixed(2)}` : "—"}</td><td>{p.paper?.open ?? "—"}</td></tr>)}</tbody></table></div>
      {data && !data.journal_available && <p role="alert" className={styles.warning}>Paper journal is temporarily unavailable.</p>}
    </section>
    <p className={styles.note}>Monitoring adds no orders. The existing paper engine continues to manage its configured pair. Data errors can affect one pair while the others remain available.</p>
  </main>;
}
