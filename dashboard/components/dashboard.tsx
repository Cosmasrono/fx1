"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Brain,
  CheckCircle2,
  Clock,
  Compass,
  Layers,
  Lock,
  Radio,
  RefreshCw,
  ShieldCheck,
  Target,
  Wallet,
  Zap,
} from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import logo from "@/public/nebtech-logo.png";
import type { Backtest, ModelStatus, Signal, State } from "@/lib/types";
import { AIReview } from "@/components/ai-review";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const money = (n: number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(n);
const price = (n?: number | null) => (n !== undefined && n !== null ? n.toFixed(5) : "—");
const pct = (n?: number | null, digits = 0) =>
  n !== undefined && n !== null ? `${(n * 100).toFixed(digits)}%` : "—";
const rMultiple = (n?: number | null, digits = 2) =>
  n !== undefined && n !== null ? `${n >= 0 ? "+" : "−"}${Math.abs(n).toFixed(digits)}R` : "—";
const shortDate = (s?: string) => (s ? new Date(s).toLocaleDateString() : "—");

export function Dashboard() {
  const [data, setData] = useState<State | null>(null);
  const [backtest, setBacktest] = useState<Backtest | null>(null);
  const [loading, setLoading] = useState(true);
  const [backtesting, setBacktesting] = useState(false);
  const [startingTraining, setStartingTraining] = useState(false);
  const [error, setError] = useState("");
  const [connected, setConnected] = useState(false);
  const [checkedAt, setCheckedAt] = useState(0);

  const load = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/state`, { cache: "no-store", signal: AbortSignal.timeout(4500) });
      if (!r.ok) throw new Error(`API request failed (${r.status})`);
      setData(await r.json());
      setConnected(true);
      setCheckedAt(Date.now());
      setError("");
    } catch (e) {
      setConnected(false);
      setError(
        e instanceof TypeError
          ? `Cannot reach the API at ${API}`
          : e instanceof Error
          ? e.message
          : "Connection failed"
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, [load]);

  async function scan() {
    setLoading(true);
    try {
      const r = await fetch(`${API}/api/scan`, { method: "POST" });
      if (!r.ok) {
        const x = await r.json();
        throw new Error(x.detail || "Scan failed");
      }
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Scan failed");
    } finally {
      setLoading(false);
    }
  }

  async function runBacktest() {
    setBacktesting(true);
    try {
      const r = await fetch(`${API}/api/backtest`, { method: "POST" });
      if (!r.ok) {
        const x = await r.json();
        throw new Error(x.detail || "Backtest failed");
      }
      setBacktest(await r.json());
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Backtest failed");
    } finally {
      setBacktesting(false);
    }
  }

  // Training runs in the background on the API; the 5-second state poll
  // carries its progress, so this only needs to start it.
  async function trainModel() {
    setStartingTraining(true);
    try {
      const r = await fetch(`${API}/api/model/train`, { method: "POST" });
      if (!r.ok) {
        const x = await r.json().catch(() => ({}));
        throw new Error(x.detail || "Model training failed to start");
      }
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Model training failed to start");
    } finally {
      setStartingTraining(false);
    }
  }

  const sig = data?.latest_signal;
  const trade = data?.open_trade;
  const openTrades = data?.open_trades ?? (trade ? [trade] : []);
  const maxConcurrent = data?.max_concurrent_trades ?? 3;
  const hasCapacity = openTrades.length < maxConcurrent;
  const bias = sig?.market_bias || "NEUTRAL";
  const strategy = (data?.strategy || "smc").toLowerCase();
  const isSmc = strategy === "smc";
  const isJudas = strategy === "judas";
  const setupScore = sig?.setup_score ?? (sig?.confidence ? Math.round(sig.confidence * 100) : 0);
  const scoreBreakdown = sig?.score_breakdown || {};
  const entryThreshold = data?.configuration?.entry_threshold ?? 70;
  const higherTimeframeOn = data?.configuration?.higher_timeframe_filter === true;
  const higherTimeframeAligned = sig?.indicators.weekly_trend === bias && sig?.indicators.monthly_trend === bias;
  const riskGate = data?.risk_gate ?? sig?.risk_gate;
  const signalTime = sig ? new Date(sig.candle_time).getTime() : NaN;
  const staleSignal = !!sig && (!Number.isFinite(signalTime) || signalTime > checkedAt ||
    checkedAt - signalTime > (data?.configuration?.signal_max_age_seconds ?? 3600) * 1000);
  const unavailable = !connected || !!data?.last_error || staleSignal;
  const isReadyToEnter = !unavailable && riskGate?.allowed !== false && setupScore >= entryThreshold && (sig?.action === "BUY" || sig?.action === "SELL");
  const entryStatus = unavailable && data ? "UNAVAILABLE" : isReadyToEnter ? `${sig?.action} APPROVED` : "WAITING";
  const entryReason = !connected ? "Waiting for a current connection to the engine."
    : data?.last_error ? `Engine update failed: ${data.last_error}`
    : staleSignal ? "The last signal is out of date. Waiting for fresh market data."
    : riskGate?.allowed === false ? riskGate.reason
    : sig?.reason || "Waiting for the first completed market scan.";

  return (
    <main>
      <header>
        <div>
          <Image className="brandLogo" src={logo} alt="NebTech Innovation" priority />
          <span className="brandDivider" />
          <span className="logoPulse" />
          <strong>EUR/USD</strong>
          <small>{data?.interval || "15min"} paper trader</small>
        </div>
        <div className="headerRight">
          <Link className="navButton" href="/pairs"><Layers size={15} /> Compare pairs</Link>
          <Link className="navButton" href="/eurusd">
            <BarChart3 size={15} /> EUR/USD Chart
          </Link>
          <span className="demo">
            <ShieldCheck size={15} /> PAPER ONLY
          </span>
          <span className="feed">
            <span /> {data?.feed || "offline"}
          </span>
          <span className="feed">
            {isJudas ? "ICT JUDAS SWING" : isSmc ? "SMC (Smart Money)" : "CLASSIC CONFLUENCE"}
          </span>
          <span className={`feed ${data?.news?.blocking ? "red" : ""}`}>
            NEWS {data?.news?.status || "DISABLED"}
          </span>
          <button onClick={runBacktest} disabled={backtesting}>
            <Activity size={16} className={backtesting ? "spin" : ""} />{" "}
            {backtesting ? "Testing…" : "Backtest"}
          </button>
          <button onClick={scan} disabled={loading}>
            <RefreshCw size={16} className={loading ? "spin" : ""} /> Scan now
          </button>
        </div>
      </header>

      <section className="wrap">
        <div className="hero">
          <div>
            <p className="overline">AUTOMATED REAL-TIME TRADING ENGINE</p>
            <h1>
              EUR/USD <span>{data?.interval || "15min"}</span>
            </h1>
            <p>
              {isJudas
                ? "ICT Judas swing engine: marks the Asian session range, waits for London to sweep one side, then trades the close back inside toward the opposite side."
                : isSmc
                ? "Smart Money Concepts engine tracking real-time liquidity sweeps, structure shifts (CHOCH), displacement legs, and Fair Value Gap retracements."
                : "Classic trend confluence engine evaluating EMA 20/50 alignment, ADX trend strength, MACD, RSI, and momentum."}
            </p>
          </div>
          <div className={`signal entrySignal ${isReadyToEnter ? sig?.action.toLowerCase() : "wait"}`} role="status">
            <small>LAST EVALUATED ENTRY</small>
            <strong>{entryStatus}</strong>
            <span>
              {sig
                ? `${bias} market bias · ${setupScore}/${entryThreshold} setup points`
                : "Starting engine…"}
            </span>
            <p className="entryReason">{entryReason}</p>
            {sig && <span>Signal candle: {new Date(sig.candle_time).toLocaleString()}</span>}
          </div>
        </div>

        {error && (
          <div className="error">
            <AlertTriangle />
            <div>
              <strong>Connection or data error</strong>
              <p>{error}. Make sure FastAPI is running and Twelve Data is reachable.</p>
            </div>
          </div>
        )}

        <div className="metrics">
          <Metric
            icon={<Wallet />}
            label="Balance"
            value={data ? money(data.account.balance) : "—"}
            sub={data?.configuration ? `Configured starting balance ${money(data.configuration.starting_balance)}` : "Paper account"}
          />
          <Metric
            icon={<Activity />}
            label="Equity"
            value={data ? money(data.account.equity) : "—"}
            sub="Includes floating paper P/L"
          />
          <Metric
            icon={<Target />}
            label="Win rate"
            value={data ? `${data.stats.win_rate}%` : "—"}
            sub={`${data?.stats.closed || 0} closed demo trades`}
          />
          <Metric
            icon={<Clock />}
            label="Last candle"
            value={
              sig
                ? new Date(sig.candle_time).toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })
                : "—"
            }
            sub={
              sig
                ? new Date(sig.candle_time).toLocaleDateString()
                : "Waiting for market data"
            }
          />
        </div>

        {/* REAL-TIME CONFIRMATION TRACKER */}
        <section className="confirmationsPanel">
          <div className="confirmationsHeader">
            <div>
              <small className="overline">LIVE CONFIRMATION TRACKER</small>
              <h2 style={{ margin: "4px 0 0", fontSize: "24px" }}>
                {isJudas
                  ? "ICT Judas Swing Sequence"
                  : isSmc
                  ? "SMC Institutional Sequence"
                  : "Classic Confluence Checklist"}
              </h2>
            </div>

            <div className="scoreTracker">
              <div className="scoreLabels">
                <span>
                  Setup Score: <strong>{setupScore} / 100 pts</strong>
                </span>
                <span
                  style={{
                    color: isReadyToEnter ? "var(--green)" : "var(--muted)",
                    fontWeight: 800,
                  }}
                >
                  {setupScore >= entryThreshold
                    ? `✓ ${entryThreshold} PTS THRESHOLD MET`
                    : `Needs ${entryThreshold - setupScore} more pts to enter`}
                </span>
              </div>
              <div className="scoreBarOuter">
                <div
                  className={`scoreBarFill ${setupScore >= entryThreshold ? "ready" : "waiting"}`}
                  style={{ width: `${Math.min(100, setupScore)}%` }}
                />
                <div className="scoreThresholdLine" style={{ left: `${Math.min(100, entryThreshold)}%` }} title={`${entryThreshold} pt entry threshold`} />
              </div>
            </div>
          </div>

          <p className="reason" style={{ margin: "0 0 16px" }}>
            <b className={bias === "BULLISH" ? "green" : bias === "BEARISH" ? "red" : ""}>
              {bias} BIAS.
            </b>{" "}
            {sig?.reason || "Waiting for market scan…"}
          </p>

          {/* Strategy Rules Grid */}
          <div className="confirmGrid">
            {isJudas ? (
              <>
                <ConfirmCard
                  title="1. Tight Asian Range"
                  points={20}
                  passed={(scoreBreakdown.tight_range || 0) === 20}
                  detail={
                    sig?.indicators.asian_high
                      ? `${price(sig.indicators.asian_low)} – ${price(sig.indicators.asian_high)} (${
                          sig.indicators.range_pips?.toFixed(1) ?? "—"
                        } pips, max 30)`
                      : "Building the 00:00–06:00 UTC range"
                  }
                />
                <ConfirmCard
                  title="2. London Sweep"
                  points={30}
                  passed={(scoreBreakdown.liquidity_sweep || 0) === 30}
                  detail={
                    sig?.indicators.swept_level
                      ? `Ran ${price(sig.indicators.swept_level)} by ${
                          sig.indicators.sweep_depth_pips?.toFixed(1) ?? "—"
                        } pips`
                      : "Waiting for London (06:00–10:00 UTC) to run the Asian high or low"
                  }
                />
                <ConfirmCard
                  title="3. Close Back Inside"
                  points={30}
                  passed={(scoreBreakdown.reclaim || 0) === 30}
                  detail={
                    (scoreBreakdown.reclaim || 0) === 30
                      ? "False move confirmed: candle closed back inside the range"
                      : "Waiting for a close back inside the range"
                  }
                />
                <ConfirmCard
                  title="4. Rejection Candle"
                  points={20}
                  passed={(scoreBreakdown.rejection || 0) === 20}
                  detail="Entry candle body of at least 0.3 ATR in the reversal direction"
                />
              </>
            ) : isSmc ? (
              <>
                <ConfirmCard
                  title="1. Liquidity Sweep"
                  points={20}
                  passed={(scoreBreakdown.liquidity_sweep || 0) === 20}
                  detail={
                    sig?.indicators.swept_level
                      ? `Swept ${price(sig.indicators.swept_level)} swing level`
                      : "Scanning for sweep of prior swing high/low"
                  }
                />
                <ConfirmCard
                  title="2. Structure Shift (CHOCH)"
                  points={20}
                  passed={(scoreBreakdown.structure_shift || 0) === 20}
                  detail={
                    sig?.indicators.smc_structure
                      ? `Character shift (${sig.indicators.smc_structure}) confirmed`
                      : "Waiting for close beyond opposing swing"
                  }
                />
                <ConfirmCard
                  title="3. Displacement Leg"
                  points={20}
                  passed={(scoreBreakdown.displacement || 0) === 20}
                  detail={
                    (scoreBreakdown.displacement || 0) === 20
                      ? "Energetic body >= 0.6 ATR confirmed"
                      : `Waiting for strong impulse candle (>= ${price(
                          (sig?.indicators.atr || 0.0005) * 0.6
                        )})`
                  }
                />
                <ConfirmCard
                  title="4. Point of Interest (POI)"
                  points={20}
                  passed={(scoreBreakdown.point_of_interest || 0) === 20}
                  detail={
                    sig?.indicators.poi
                      ? `Retraced into ${sig.indicators.poi}`
                      : "Price not yet inside Fair Value Gap / Order Block"
                  }
                />
                <ConfirmCard
                  title="5. Dealing Range Zone"
                  points={10}
                  passed={(scoreBreakdown.premium_discount || 0) === 10}
                  detail={
                    sig?.indicators.equilibrium
                      ? bias === "BULLISH"
                        ? `In Discount (< ${price(sig.indicators.equilibrium)})`
                        : `In Premium (> ${price(sig.indicators.equilibrium)})`
                      : "Evaluating 50% equilibrium"
                  }
                />
                <ConfirmCard
                  title="6. Active Killzone"
                  points={10}
                  passed={(scoreBreakdown.killzone || 0) === 10}
                  detail={
                    sig?.indicators.killzone && sig.indicators.killzone !== "OUTSIDE"
                      ? `${sig.indicators.killzone} (Active liquidity window)`
                      : "Outside Killzone (London: 07-10h / NY: 12-15h UTC)"
                  }
                />
              </>
            ) : (
              <>
                <ConfirmCard
                  title="1. EMA 20/50 Trend"
                  points={15}
                  passed={(scoreBreakdown.ema_trend || 0) === 15}
                  detail={`EMA20: ${price(sig?.indicators.ema20)} vs EMA50: ${price(
                    sig?.indicators.ema50
                  )}`}
                />
                <ConfirmCard
                  title="2. ADX Trend Strength"
                  points={15}
                  passed={(scoreBreakdown.adx || 0) === 15}
                  detail={`ADX ${sig?.indicators.adx?.toFixed(1) || "—"} (Min 20.0)`}
                />
                <ConfirmCard
                  title="3. MACD Momentum"
                  points={10}
                  passed={(scoreBreakdown.macd || 0) === 10}
                  detail={`MACD: ${sig?.indicators.macd?.toFixed(6) || "—"}`}
                />
                <ConfirmCard
                  title="4. RSI Confluence"
                  points={10}
                  passed={(scoreBreakdown.rsi || 0) === 10}
                  detail={`RSI: ${sig?.indicators.rsi?.toFixed(1) || "—"} (52-70 Buy, 30-48 Sell)`}
                />
                <ConfirmCard
                  title="5. 10-Bar Momentum"
                  points={10}
                  passed={(scoreBreakdown.momentum || 0) === 10}
                  detail={`10-Bar: ${(((sig?.indicators.momentum_10 || 0) * 100)).toFixed(3)}%`}
                />
                <ConfirmCard
                  title="6. Market Structure"
                  points={15}
                  passed={(scoreBreakdown.market_structure || 0) === 15}
                  detail={`Structure: ${sig?.indicators.structure || "MIXED"}`}
                />
              </>
            )}
          </div>

          {/* Safety & Risk Gates */}
          <div style={{ marginTop: "20px" }}>
            <small className="overline">SAFETY & MACRO RISK GATES</small>
            <div className="gatesGrid">
              <div className="gateItem">
                <div>
                  <small>Higher-Timeframe Trend</small>
                  <span>
                    W: {sig?.indicators.weekly_trend || "—"} · M:{" "}
                    {sig?.indicators.monthly_trend || "—"}
                  </span>
                </div>
                <span
                  className={`confirmPill ${
                    higherTimeframeOn && higherTimeframeAligned
                      ? "pass"
                      : "pending"
                  }`}
                >
                  {!higherTimeframeOn ? "DISABLED" : higherTimeframeAligned ? "ALIGNED" : "NOT ALIGNED"}
                </span>
              </div>

              <div className="gateItem">
                <div>
                  <small>News Blackout Guard</small>
                  <span>{data?.news?.event ? data.news.event.name : "Economic Calendar"}</span>
                </div>
                <span
                  className={`confirmPill ${
                    data?.news?.blocking ? "block" : "pass"
                  }`}
                >
                  {data?.news?.status || "CLEAR"}
                </span>
              </div>

              <div className="gateItem">
                <div>
                  <small>Circuit Breakers</small>
                  <span>{data?.configuration?.max_consecutive_losses ?? 3}-Loss & Drawdown Check</span>
                </div>
                <span
                  className={`confirmPill ${
                    riskGate?.allowed !== false ? "pass" : "block"
                  }`}
                >
                  {riskGate?.allowed !== false ? "CLEAR" : "PAUSED"}
                </span>
              </div>

              <div className="gateItem">
                <div>
                  <small>Position Slots</small>
                  <span>{openTrades.length} / {maxConcurrent} Active Trades</span>
                </div>
                <span className={`confirmPill ${hasCapacity ? "pass" : "block"}`}>
                  {hasCapacity ? `${maxConcurrent - openTrades.length} SLOTS OPEN` : "CAP REACHED"}
                </span>
              </div>
            </div>
          </div>
        </section>

        {riskGate?.allowed === false && (
          <p className="reason">{riskGate.reason}{riskGate.resume_at ? ` Rechecked after ${new Date(riskGate.resume_at).toLocaleString()}; other limits still apply.` : ""}</p>
        )}

        <PredictionPanel
          model={data?.prediction_model}
          signal={sig}
          onTrain={trainModel}
          starting={startingTraining}
        />

        <div className="grid">
          <section className="panel">
            <div className="panelHead">
              <div>
                <small>CURRENT PRICE ACTION</small>
                <h2>{price(sig?.price)}</h2>
              </div>
              <span className="symbol">EUR / USD</span>
            </div>
            {sig ? (
              <>
                <div className="setupSummary">
                  <span>
                    Market bias{" "}
                    <b className={bias === "BULLISH" ? "green" : bias === "BEARISH" ? "red" : ""}>
                      {bias}
                    </b>
                  </span>
                  <span>
                    Last entry decision <b>{entryStatus}</b>
                  </span>
                  <span>
                    Setup score <b>{setupScore}/100</b>
                  </span>
                  <span>
                    Session <b>{sig.indicators.session || "—"}</b>
                  </span>
                  <span>
                    Structure <b>{sig.indicators.structure || "—"}</b>
                  </span>
                  <span>
                    ADX <b>{sig.indicators.adx?.toFixed(1) || "—"}</b>
                  </span>
                  {sig.indicators.killzone && (
                    <>
                      <span>
                        Killzone <b>{sig.indicators.killzone}</b>
                      </span>
                      <span>
                        POI <b>{sig.indicators.poi || "none"}</b>
                      </span>
                      <span>
                        Shift <b>{sig.indicators.smc_structure || "none"}</b>
                      </span>
                      <span>
                        Swept <b>{price(sig.indicators.swept_level)}</b>
                      </span>
                    </>
                  )}
                </div>
                <div className="indicators">
                  <Indicator label="EMA 20" value={price(sig.indicators.ema20)} />
                  <Indicator label="EMA 50" value={price(sig.indicators.ema50)} />
                  <Indicator label="RSI 14" value={sig.indicators.rsi.toFixed(1)} />
                  <Indicator label="ATR 14" value={price(sig.indicators.atr)} />
                  <Indicator label="MACD" value={sig.indicators.macd.toFixed(6)} />
                  <Indicator
                    label="Momentum 10"
                    value={`${(sig.indicators.momentum_10 * 100).toFixed(3)}%`}
                  />
                </div>
              </>
            ) : (
              <p className="empty">Waiting for the first scan.</p>
            )}
          </section>

          <section className="panel">
            <div className="panelHead">
              <div>
                <small>ACTIVE POSITIONS ({openTrades.length} / {maxConcurrent})</small>
                <h2>
                  {openTrades.length === 0
                    ? "No open trades"
                    : openTrades.length === 1
                    ? `${openTrades[0].side} #${openTrades[0].id}`
                    : `${openTrades.length} Concurrent ${openTrades[0].side} Positions`}
                </h2>
              </div>
              {openTrades.length > 0 &&
                (openTrades[0].side === "BUY" ? (
                  <ArrowUpRight className="green" />
                ) : (
                  <ArrowDownRight className="red" />
                ))}
            </div>
            {openTrades.length > 0 ? (
              <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                {openTrades.map((pos, idx) => (
                  <div
                    key={pos.id}
                    className="position"
                    style={{
                      borderBottom:
                        idx < openTrades.length - 1 ? "1px solid var(--border, rgba(255,255,255,0.08))" : "none",
                      paddingBottom: idx < openTrades.length - 1 ? "14px" : "0",
                    }}
                  >
                    <div>
                      <small>Position #{pos.id}</small>
                      <b className={pos.side === "BUY" ? "green" : "red"}>{pos.side}</b>
                    </div>
                    <div>
                      <small>Entry</small>
                      <b>{price(pos.entry)}</b>
                    </div>
                    <div>
                      <small>Stop loss</small>
                      <b className="red">{price(pos.stop_loss)}</b>
                    </div>
                    <div>
                      <small>Take profit</small>
                      <b className="green">{price(pos.take_profit)}</b>
                    </div>
                    <div>
                      <small>Risk</small>
                      <b>{money(pos.risk_amount)}</b>
                    </div>
                    <div>
                      <small>Units</small>
                      <b>{pos.units.toLocaleString()}</b>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="emptyState">
                <ShieldCheck size={36} />
                <h3>Waiting for {entryThreshold}/100 Confirmation</h3>
                <p>{data?.configuration?.target_description || "Waiting for execution settings."}</p>
              </div>
            )}
          </section>
        </div>

        <section className="panel history">
          <div className="panelHead">
            <div>
              <small>PAPER JOURNAL</small>
              <h2>Recent trades</h2>
            </div>
          </div>
          <div className="table">
            <div className="tr row">
              <b>Trade</b>
              <b>Entry</b>
              <b>Stop / Target</b>
              <b>Status</b>
              <b>Model</b>
              <b>P/L</b>
            </div>
            {data?.trades.length ? (
              data.trades.map((t) => (
                <div className="row" key={t.id}>
                  <span>
                    <b className={t.side === "BUY" ? "green" : "red"}>{t.side}</b>
                    <small> #{t.id}</small>
                  </span>
                  <span>{price(t.entry)}</span>
                  <span>
                    {price(t.stop_loss)} / {price(t.take_profit)}
                  </span>
                  <span className="status">{t.status}</span>
                  <span title="Model's target-before-stop probability at entry">
                    {pct(t.signal_snapshot?.prediction?.target_before_stop_probability)}
                  </span>
                  <b className={t.pnl > 0 ? "green" : t.pnl < 0 ? "red" : ""}>
                    {money(t.pnl || 0)}
                  </b>
                </div>
              ))
            ) : (
              <p className="empty">No paper trades yet.</p>
            )}
          </div>
        </section>

        {backtest && (
          <section className="panel backtest">
            <div className="panelHead">
              <div>
                <small>HISTORICAL BACKTEST</small>
                <h2>
                  {backtest.bars} candles · {backtest.feed}
                </h2>
              </div>
              <span className="symbol">
                Costs: {backtest.assumptions.spread_pips} pip spread +{" "}
                {backtest.assumptions.slippage_pips} pip slippage
              </span>
            </div>
            <p className="reason">
              Historical simulation with shared position limits, loss controls, pattern memory,
              stop/target handling and {backtest.assumptions.risk_percent ?? "configured"}% risk sizing.
            </p>
            {backtest.assumptions.excluded_gates?.length ? <p className="modelMeta">Excluded: {backtest.assumptions.excluded_gates.join("; ")}.</p> : null}
            <div className="indicators">
              <Indicator label="Net P/L" value={money(backtest.metrics.net_pnl)} />
              <Indicator label="Return" value={`${backtest.metrics.return_percent}%`} />
              <Indicator
                label="Max drawdown"
                value={`${backtest.metrics.max_drawdown_percent}%`}
              />
              <Indicator
                label="Closed trades"
                value={String(backtest.metrics.closed_trades)}
              />
              <Indicator label="Win rate" value={`${backtest.metrics.win_rate}%`} />
              <Indicator
                label="Profit factor"
                value={backtest.metrics.profit_factor?.toFixed(2) || "—"}
              />
            </div>
          </section>
        )}

        <AIReview />
        <Link className="chartShortcut" href="/eurusd">
          <BarChart3 size={16} /> Open EUR/USD Chart
        </Link>

        <footer>
          <ShieldCheck size={16} /> Demo execution only. No broker connection and no
          real-money order endpoint.
        </footer>
      </section>
    </main>
  );
}

function Metric({
  icon,
  label,
  value,
  sub,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub: string;
}) {
  return (
    <div className="metric">
      <span className="metricIcon">{icon}</span>
      <small>{label}</small>
      <strong>{value}</strong>
      <p>{sub}</p>
    </div>
  );
}

function Indicator({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <small>{label}</small>
      <b>{value}</b>
    </div>
  );
}

function ConfirmCard({
  title,
  points,
  passed,
  detail,
}: {
  title: string;
  points: number;
  passed: boolean;
  detail: string;
}) {
  return (
    <div className={`confirmCard ${passed ? "passed" : "pending"}`}>
      <div className="confirmCardTop">
        <span className="confirmCardTitle">
          {passed ? (
            <CheckCircle2 size={16} style={{ color: "var(--green)" }} />
          ) : (
            <Clock size={16} style={{ color: "var(--muted)" }} />
          )}
          {title}
        </span>
        <span className={`confirmPill ${passed ? "pass" : "pending"}`}>
          {passed ? `+${points} PTS` : `0 / ${points}`}
        </span>
      </div>
      <div className="confirmDetail">{detail}</div>
    </div>
  );
}

function PredictionPanel({
  model,
  signal,
  onTrain,
  starting,
}: {
  model?: ModelStatus;
  signal?: Signal | null;
  onTrain: () => void;
  starting: boolean;
}) {
  const job = model?.job;
  const running = job?.status === "running";
  const report = model?.available ? model.report : undefined;
  const evidence = report?.out_of_sample;
  const latestFold = evidence?.folds[evidence.folds.length - 1];
  const improvement = evidence?.baseline.avg_r != null && evidence.filtered.avg_r != null
    ? evidence.filtered.avg_r - evidence.baseline.avg_r : null;
  const prediction = signal?.prediction;
  const probability = prediction?.available ? prediction.target_before_stop_probability : undefined;
  const threshold = prediction?.threshold ?? model?.threshold ?? 0;
  const take = prediction?.allowed ?? true;
  const progress = job?.total ? Math.min(100, ((job.done ?? 0) / job.total) * 100) : 4;

  let current: React.ReactNode;
  if (!model) {
    current = <p className="modelText">Waiting for the API…</p>;
  } else if (!model.enabled) {
    current = (
      <p className="modelText">
        The model filter is off. Set <code>PREDICTION_MODEL_ENABLED=true</code> in{" "}
        <code>backend/.env</code> and restart the API.
      </p>
    );
  } else if (!model.available) {
    current = (
      <p className="modelText">
        {model.problem ||
          "No model trained yet. Click Train model: the first run replays five years of candles and takes about 15 minutes."}
      </p>
    );
  } else if (probability !== undefined) {
    current = (
      <>
        <div className="probability">
          <strong className={take ? "green" : "red"}>{pct(probability)}</strong>
          <span className={`confirmPill ${threshold > 0 ? take ? "pass" : "block" : ""}`}>
            {threshold <= 0 ? "INFORMATION ONLY" : take ? "MODEL ALLOWS" : "MODEL BLOCKS"}
          </span>
        </div>
        <div className="probBar">
          <div className={`probFill ${take ? "take" : "skip"}`} style={{ width: pct(probability) }} />
          {threshold > 0 && (
            <div className="probMarker" style={{ left: pct(threshold) }} title={`Threshold ${pct(threshold)}`} />
          )}
        </div>
        <p className="modelText">
          Estimated chance this setup reaches take-profit before stop-loss.{" "}
          {threshold > 0 ? `Entries need at least ${pct(threshold)}.` : "This model never blocks entries."}
          {" "}The engine checks other entry conditions separately.
        </p>
      </>
    );
  } else {
    current = (
      <p className="modelText">
        No BUY/SELL setup reached the model on the last candle. It scores every setup that passes
        the strategy&apos;s checks.
      </p>
    );
  }

  const verdict = !report
    ? null
    : !report.filter_helps
    ? {
        tone: "warn",
        text: model?.threshold_source === "env" && (model.threshold ?? 0) > 0
          ? "The filter did not improve results on unseen data. A manual threshold is nevertheless active."
          : "On unseen data the filter did not improve results, so it never blocks an entry. Treat the probability as information only.",
      }
    : (evidence?.filtered.avg_r ?? 0) < 0
    ? {
        tone: "warn",
        text: "The filter cut the average loss, but the setups it allows still lost money on average in this replay. It reduces losses; it does not make this strategy profitable by itself.",
      }
    : {
        tone: "good",
        text: "Setups the model allows were profitable on unseen data. Keep paper trading to confirm before relying on it.",
      };

  return (
    <section className="panel modelPanel">
      <div className="panelHead">
        <div>
          <small>WIN-PROBABILITY MODEL</small>
          <h2>Will this setup hit TP before SL?</h2>
        </div>
        <button onClick={onTrain} disabled={running || starting}>
          <Brain size={16} /> {running ? "Training…" : model?.available ? "Retrain model" : "Train model"}
        </button>
      </div>

      {report && evidence && (
        <div className="performanceSummary">
          {verdict && <p className={`modelNote ${verdict.tone}`}>{verdict.text}</p>}
          <div className="performanceMetrics">
            <div><small>Average return per setup</small><strong className={(evidence.filtered.avg_r ?? 0) < 0 ? "red" : ""}>{rMultiple(evidence.filtered.avg_r)}</strong><span>With filtering · {rMultiple(evidence.baseline.avg_r)} without</span></div>
            <div><small>Change from filtering</small><strong className={improvement == null ? "" : improvement > 0 ? "green" : improvement < 0 ? "red" : ""}>{rMultiple(improvement, 3)}</strong><span>Difference in average R per setup</span></div>
            <div><small>Latest unseen period</small><strong className={(latestFold?.filtered.avg_r ?? 0) < 0 ? "red" : ""}>{rMultiple(latestFold?.filtered.avg_r)}</strong><span>{latestFold ? `${shortDate(latestFold.start)} – ${shortDate(latestFold.end)} · ${latestFold.filtered.trades.toLocaleString()} retained setups` : "No tested periods"}</span></div>
          </div>
          <p className="modelMeta">Historical validation of the filtering procedure; 1R is the amount risked per setup. {evidence.filtered.trades.toLocaleString()} retained of {evidence.baseline.trades.toLocaleString()} tested setups. {report.validation_scope || "These setup results are not a complete account simulation."}</p>
          <details className="validationPeriods">
            <summary>Compare all {evidence.folds.length} unseen periods</summary>
            <div className="table"><table>
              <caption>Historical results with and without model filtering</caption>
              <thead><tr><th scope="col">Period</th><th scope="col">Without filter</th><th scope="col">With filter</th><th scope="col">Retained / tested</th><th scope="col">Win rate with filter</th></tr></thead>
              <tbody>{evidence.folds.map(fold => <tr key={fold.fold}>
                <th scope="row">{shortDate(fold.start)} – {shortDate(fold.end)}</th>
                <td>{rMultiple(fold.baseline.avg_r)}</td><td>{rMultiple(fold.filtered.avg_r)}</td>
                <td>{fold.filtered.trades.toLocaleString()} / {fold.baseline.trades.toLocaleString()}</td><td>{pct(fold.filtered.win_rate, 1)}</td>
              </tr>)}</tbody>
            </table></div>
          </details>
        </div>
      )}

      {model?.learning && (
        <div className="reason" aria-live="polite">
          <strong>Learning from paper trades</strong>
          <p>{model.learning.trained_trades} completed trades used in the saved model · {model.learning.new_trades} new outcomes awaiting review · {model.learning.closed_trades} total closed trades.</p>
          <p>{model.learning.enabled
            ? `Automatic retraining after ${model.learning.retrain_after} new outcomes, at least ${model.learning.cooldown_hours} hours apart.`
            : "Automatic retraining is off; use Train model to include new outcomes."}</p>
          {report?.paper_learning && <p>Training included {report.paper_learning.wins} targets reached and {report.paper_learning.losses} stopped trades. {Object.values(report.paper_learning.skipped).reduce((a, b) => a + b, 0)} outcomes were excluded because their data or settings could not be used.</p>}
          {signal?.learning && <p>Current setup resembles {signal.learning.loss_matches} losses and {signal.learning.win_matches} wins from the last {signal.learning.lookback_days} days. {signal.learning.blocked ? "The repeated-loss guard blocked this entry." : "The repeated-loss guard allows this setup."}</p>}
          <p className="modelMeta">Both wins and losses are training evidence. Setup scores are rule checks; model probabilities are estimates. A small trade journal is not enough to establish a reliable pattern.</p>
        </div>
      )}

      {running && (
        <div className="trainProgress">
          <div>
            <span>{job?.stage}</span>
            <span>
              {job?.total ? `${(job.done ?? 0).toLocaleString()} / ${job.total.toLocaleString()} candles` : ""}
            </span>
          </div>
          <div className="probBar">
            <div className="probFill" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}
      {job?.status === "error" && <p className="modelNote warn">Training failed: {job.error}</p>}

      <div className="modelBody">
        <div className="modelCard">
          <small>CURRENT SETUP</small>
          {current}
        </div>
        <div className="modelCard">
          <small>TESTED ON DATA IT NEVER SAW</small>
          {report && evidence ? (
            <>
              <div className="evidence">
                <div>
                  <small>Win rate</small>
                  <b>
                    {pct(evidence.baseline.win_rate, 1)} → {pct(evidence.filtered.win_rate, 1)}
                  </b>
                  <span>all setups → setups it allows</span>
                </div>
                <div>
                  <small>Average per trade</small>
                  <b>
                    {rMultiple(evidence.baseline.avg_r)} → {rMultiple(evidence.filtered.avg_r)}
                  </b>
                  <span>1R = the amount risked</span>
                </div>
                <div>
                  <small>Losing setups skipped</small>
                  <b className="green">{evidence.losses_avoided.toLocaleString()}</b>
                  <span>winners given up: {evidence.wins_given_up.toLocaleString()}</span>
                </div>
                <div>
                  <small>Ranking skill (AUC)</small>
                  <b>{evidence.auc?.toFixed(2) ?? "—"}</b>
                  <span>0.50 = coin flip · 1.00 = perfect</span>
                </div>
              </div>
              <p className="modelMeta">
                {evidence.baseline.trades.toLocaleString()} {report.population} across{" "}
                {evidence.folds.length} later periods. Trained {shortDate(report.trained_at)} on{" "}
                {report.samples.toLocaleString()} setups, {shortDate(report.data_start)} –{" "}
                {shortDate(report.data_end)}.
                {model?.threshold_source === "env" ? " Threshold set by PREDICTION_MIN_PROBABILITY." : ""}
                {report.validation_scope ? ` ${report.validation_scope}` : ""}
              </p>
            </>
          ) : (
            <p className="modelText">Train the model to see how it performed on unseen history.</p>
          )}
        </div>
      </div>
    </section>
  );
}
