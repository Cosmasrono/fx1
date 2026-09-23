"use client";

import {useCallback,useEffect,useMemo,useState} from "react";
import Link from "next/link";
import {AlertTriangle,ArrowLeft,RefreshCw,ShieldCheck} from "lucide-react";
import Image from "next/image";
import logo from "@/public/nebtech-logo.png";
import type {Candle,CandleResponse,State,Trade} from "@/lib/types";

const API=process.env.NEXT_PUBLIC_API_URL||"http://localhost:8000";
const WIDTH=1200,HEIGHT=560,PAD={top:28,right:82,bottom:48,left:18};
const price=(value:number)=>value.toFixed(5);
const money=(value:number)=>new Intl.NumberFormat("en-US",{style:"currency",currency:"USD"}).format(value);

function ema(candles:Candle[],period:number){
  const factor=2/(period+1);let current=candles[0]?.close??0;
  return candles.map((c,index)=>current=index===0?c.close:c.close*factor+current*(1-factor));
}

export function MarketChart(){
  const [data,setData]=useState<CandleResponse|null>(null);
  const [state,setState]=useState<State|null>(null);
  const [selected,setSelected]=useState<number|null>(null);
  const [loading,setLoading]=useState(true);
  const [error,setError]=useState("");
  const load=useCallback(async()=>{setLoading(true);try{const [chartResponse,stateResponse]=await Promise.all([fetch(`${API}/api/candles?limit=160`,{cache:"no-store"}),fetch(`${API}/api/state`,{cache:"no-store"})]);if(!chartResponse.ok){const body=await chartResponse.json().catch(()=>({}));throw new Error(body.detail||`Chart request failed (${chartResponse.status})`)}if(!stateResponse.ok)throw new Error(`Trade state request failed (${stateResponse.status})`);setData(await chartResponse.json());setState(await stateResponse.json());setError("")}catch(reason){setError(reason instanceof TypeError?`Cannot reach the API at ${API}`:reason instanceof Error?reason.message:"Chart request failed")}finally{setLoading(false)}},[]);
  useEffect(()=>{load();const id=setInterval(load,60000);return()=>clearInterval(id)},[load]);
  const openPositions = useMemo(() => (
    state?.open_trades && state.open_trades.length > 0
      ? state.open_trades
      : state?.open_trade ? [state.open_trade] : []
  ), [state]);
  const chart=useMemo(()=>{const candles=data?.candles??[];if(!candles.length)return null;const levels=openPositions.flatMap(p=>[p.entry,p.stop_loss,p.take_profit]);const low=Math.min(...candles.map(c=>c.low),...(levels.length?levels:[Infinity])),high=Math.max(...candles.map(c=>c.high),...(levels.length?levels:[-Infinity]));const cleanLow=isFinite(low)?low:1.0,cleanHigh=isFinite(high)?high:1.1;const margin=(cleanHigh-cleanLow)*.06||.0001,min=cleanLow-margin,max=cleanHigh+margin;const plotW=WIDTH-PAD.left-PAD.right,plotH=HEIGHT-PAD.top-PAD.bottom;const x=(index:number)=>PAD.left+(index+.5)*plotW/candles.length;const y=(value:number)=>PAD.top+(max-value)*plotH/(max-min);const line=(values:number[])=>values.map((value,index)=>`${x(index)},${y(value)}`).join(" ");const visibleTrades=(state?.trades??[]).filter(t=>new Date(t.opened_at).getTime()>=new Date(candles[0].datetime).getTime());const tradeIndex=(trade:Trade)=>candles.reduce((best,candle,index)=>Math.abs(new Date(candle.datetime).getTime()-new Date(trade.opened_at).getTime())<Math.abs(new Date(candles[best].datetime).getTime()-new Date(trade.opened_at).getTime())?index:best,0);return{candles,min,max,x,y,plotW,plotH,ema20:line(ema(candles,20)),ema50:line(ema(candles,50)),visibleTrades,tradeIndex}},[data,state,openPositions]);
  const active=chart?.candles[selected??chart.candles.length-1];
  return <main><header><div><Image className="brandLogo" src={logo} alt="NebTech Innovation" priority/><span className="brandDivider"/><span className="logoPulse"/><strong>EUR/USD</strong><small>{data?.interval||"15min"} market chart</small></div><div className="headerRight"><Link className="navButton" href="/"><ArrowLeft size={16}/> Dashboard</Link><span className="demo"><ShieldCheck size={15}/> PAPER ONLY</span><button onClick={load} disabled={loading}><RefreshCw size={16} className={loading?"spin":""}/> Refresh</button></div></header>
  <section className="wrap chartWrap"><div className="chartHero"><div><p className="overline">MARKET VIEW</p><h1>EUR/USD <span>{data?.interval||"15min"}</span></h1><p>Recent price action with EMA 20 and EMA 50 trend overlays.</p></div><div className="chartQuote"><small>SELECTED CLOSE</small><strong>{active?price(active.close):"—"}</strong><span>{active?new Date(active.datetime).toLocaleString():"Waiting for data"}</span></div></div>
  {error&&<div className="error"><AlertTriangle/><div><strong>Chart data unavailable</strong><p>{error}</p></div></div>}
  {openPositions.length>0?<div style={{display:"flex",flexDirection:"column",gap:"10px",marginBottom:"16px"}}>{openPositions.map(pos=>{const stopPips=Math.abs(pos.entry-pos.stop_loss)/.0001;const targetPips=Math.abs(pos.take_profit-pos.entry)/.0001;const targetDollars=targetPips*.0001*pos.units;return <section key={pos.id} className="tradePlan"><div><small>OPEN PAPER TRADE</small><strong className={pos.side==="BUY"?"green":"red"}>{pos.side} #{pos.id}</strong></div><div><small>STOP DISTANCE</small><strong>{stopPips.toFixed(1)} pips</strong><span>Estimated loss <b className="red">-{money(pos.risk_amount)}</b></span></div><div><small>PROFIT TARGET</small><strong>{targetPips.toFixed(1)} pips</strong><span>Estimated gain <b className="green">+{money(targetDollars)}</b></span></div><div><small>RISK / REWARD</small><strong>1 : {(targetDollars/pos.risk_amount).toFixed(2)}</strong><span>{pos.units.toLocaleString()} EUR units</span></div></section>})}</div>:<div className="noTradePlan"><ShieldCheck size={16}/><span>No open entry. Pip and USD targets will appear here when the strategy opens a paper trade.</span></div>}
  <section className="panel chartPanel"><div className="panelHead"><div><small>EUR / USD</small><h2>Price chart</h2></div><div className="chartLegend"><span className="candleKey up"/>Bullish <span className="candleKey down"/>Bearish <span className="lineKey ema20"/>EMA 20 <span className="lineKey ema50"/>EMA 50 <span className="tradeKey"/>Entry</div></div>
  {chart?<div className="chartCanvas"><svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="EUR/USD candlestick chart" onMouseLeave={()=>setSelected(null)}>
    {[0,1,2,3,4].map(step=>{const value=chart.max-(chart.max-chart.min)*step/4,y=PAD.top+chart.plotH*step/4;return <g key={step}><line className="gridLine" x1={PAD.left} x2={WIDTH-PAD.right} y1={y} y2={y}/><text className="axisText" x={WIDTH-PAD.right+10} y={y+4}>{price(value)}</text></g>})}
    <polyline className="emaLine ema50Stroke" points={chart.ema50}/><polyline className="emaLine ema20Stroke" points={chart.ema20}/>
    {openPositions.map(pos=><g key={`levels-${pos.id}`} className="positionLevels"><line className="tradeLevel entryLevel" x1={PAD.left} x2={WIDTH-PAD.right} y1={chart.y(pos.entry)} y2={chart.y(pos.entry)}/><text className="levelLabel entryLabel" x={WIDTH-PAD.right-5} y={chart.y(pos.entry)-5}>ENTRY #{pos.id} {price(pos.entry)}</text><line className="tradeLevel stopLevel" x1={PAD.left} x2={WIDTH-PAD.right} y1={chart.y(pos.stop_loss)} y2={chart.y(pos.stop_loss)}/><text className="levelLabel stopLabel" x={WIDTH-PAD.right-5} y={chart.y(pos.stop_loss)-5}>SL #{pos.id} {price(pos.stop_loss)}</text><line className="tradeLevel targetLevel" x1={PAD.left} x2={WIDTH-PAD.right} y1={chart.y(pos.take_profit)} y2={chart.y(pos.take_profit)}/><text className="levelLabel targetLevel" x={WIDTH-PAD.right-5} y={chart.y(pos.take_profit)-5}>TP #{pos.id} {price(pos.take_profit)}</text></g>)}
    {chart.candles.map((c,index)=>{const x=chart.x(index),up=c.close>=c.open,width=Math.max(2,chart.plotW/chart.candles.length*.62),top=chart.y(Math.max(c.open,c.close)),bottom=chart.y(Math.min(c.open,c.close));return <g key={c.datetime} className="candle" onMouseEnter={()=>setSelected(index)}><rect className="candleHit" x={x-chart.plotW/chart.candles.length/2} y={PAD.top} width={chart.plotW/chart.candles.length} height={chart.plotH}/><line className={up?"wick up":"wick down"} x1={x} x2={x} y1={chart.y(c.high)} y2={chart.y(c.low)}/><rect className={up?"body up":"body down"} x={x-width/2} y={top} width={width} height={Math.max(1,bottom-top)}/></g>})}
    {chart.visibleTrades.map(trade=>{const index=chart.tradeIndex(trade),x=chart.x(index),y=chart.y(trade.entry),buy=trade.side==="BUY";return <g className={`entryMarker ${buy?"buy":"sell"}`} key={`trade-${trade.id}`}><path d={buy?`M ${x} ${y-3} l -7 10 h 14 z`:`M ${x} ${y+3} l -7 -10 h 14 z`}/><text textAnchor="middle" x={x} y={buy?y+20:y-13}>{trade.side} #{trade.id}</text></g>})}
    {[0,.25,.5,.75,1].map((ratio,index)=>{const i=Math.min(chart.candles.length-1,Math.floor((chart.candles.length-1)*ratio)),x=chart.x(i);return <text className="axisText timeAxis" textAnchor={index===0?"start":index===4?"end":"middle"} x={x} y={HEIGHT-16} key={ratio}>{new Date(chart.candles[i].datetime).toLocaleDateString([],{month:"short",day:"numeric"})}</text>})}
  </svg>{active&&<div className="ohlc"><span>{new Date(active.datetime).toLocaleString()}</span><b>O {price(active.open)}</b><b>H {price(active.high)}</b><b>L {price(active.low)}</b><b>C {price(active.close)}</b></div>}</div>:<div className="chartLoading"><RefreshCw className={loading?"spin":""}/><p>{loading?"Loading market candles…":"No candles available."}</p></div>}
  </section><footer><ShieldCheck size={16}/> Chart data supports paper-trading analysis only. It is not broker execution data.</footer></section></main>;
}
