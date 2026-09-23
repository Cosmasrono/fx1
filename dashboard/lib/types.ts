export type RiskGate = {
  allowed: boolean;
  reason: string;
  daily_pnl?: number;
  weekly_pnl?: number;
  resume_at?: string | null;
};

export type PredictionInfo = {
  available: boolean;
  enabled: boolean;
  allowed: boolean;
  target_before_stop_probability?: number;
  threshold?: number;
  reason?: string;
};

export type ModelSlice = {
  trades: number;
  win_rate: number | null;
  avg_r: number | null;
  total_r: number;
};

export type ModelReport = {
  trained_at: string;
  strategy: string;
  data_start: string;
  data_end: string;
  samples: number;
  win_rate: number;
  population: string;
  threshold: number;
  filter_helps: boolean;
  used_latest_candles: boolean;
  historical_samples?: number;
  paper_learning?: { closed_trades: number; used: number; wins: number; losses: number; legacy_verified: number; skipped: Record<string, number> };
  validation_scope?: string;
  out_of_sample: {
    auc: number | null;
    baseline: ModelSlice;
    filtered: ModelSlice;
    losses_avoided: number;
    wins_given_up: number;
    folds: {
      fold: number;
      start: string;
      end: string;
      auc: number | null;
      threshold: number;
      baseline: ModelSlice;
      filtered: ModelSlice;
    }[];
  };
};

export type TrainingJob = {
  status: "idle" | "running" | "done" | "error";
  stage?: string;
  done?: number;
  total?: number;
  started_at?: string;
  finished_at?: string;
  error?: string;
};

export type ModelStatus = {
  enabled: boolean;
  available: boolean;
  problem?: string | null;
  threshold?: number;
  threshold_source?: "env" | "validated";
  report?: ModelReport;
  job: TrainingJob;
  learning?: { enabled: boolean; closed_trades: number; trained_trades: number; new_trades: number; retrain_after: number; cooldown_hours: number };
};

export type Indicators = {
  ema20: number;
  ema50: number;
  rsi: number;
  macd: number;
  macd_signal: number;
  atr: number;
  vol_20: number;
  momentum_10: number;
  adx?: number;
  structure?: string;
  session?: string;
  weekly_trend?: string;
  monthly_trend?: string;
  killzone?: string;
  smc_structure?: string | null;
  poi?: string | null;
  swept_level?: number;
  equilibrium?: number | null;
  target_source?: "LIQUIDITY" | "R_MULTIPLE";
  asian_high?: number;
  asian_low?: number;
  range_pips?: number;
  sweep_depth_pips?: number;
};

export type Signal = {
  action: "BUY" | "SELL" | "HOLD";
  market_bias?: "BULLISH" | "BEARISH" | "NEUTRAL";
  bias_score?: number;
  confidence: number;
  setup_score?: number;
  score_breakdown?: Record<string, number>;
  price: number;
  reason: string;
  candle_time: string;
  indicators: Indicators;
  model?: string;
  levels_hint?: { stop: number; target: number };
  risk_gate?: RiskGate;
  prediction?: PredictionInfo;
  learning?: { similar_matches: number; loss_matches: number; win_matches: number; blocked: boolean; lookback_days: number; average_pnl: number | null };
};

export type Trade = {
  id: number;
  side: "BUY" | "SELL";
  status: "OPEN" | "CLOSED";
  entry: number;
  stop_loss: number;
  take_profit: number;
  units: number;
  risk_amount: number;
  opened_at: string;
  closed_at?: string;
  exit_price?: number;
  pnl: number;
  reason?: string;
  signal_snapshot?: { prediction?: PredictionInfo };
};

export type NewsStatus = {
  enabled: boolean;
  configured: boolean;
  blocking: boolean;
  status: "DISABLED" | "UNCONFIGURED" | "UNAVAILABLE" | "BLACKOUT" | "CLEAR";
  event: { name: string; country?: string; time: string; importance: number } | null;
  checked_at: string;
  error?: string;
};

export type State = {
  mode: string;
  symbol: string;
  interval: string;
  strategy?: string;
  feed: string;
  account: { balance: number; equity: number; updated_at: string };
  open_trade: Trade | null;
  open_trades?: Trade[];
  max_concurrent_trades?: number;
  configuration?: { starting_balance: number; risk_percent: number; entry_threshold: number; higher_timeframe_filter: boolean; max_consecutive_losses: number; loss_cooldown_hours: number; target_description: string; signal_max_age_seconds?: number };
  risk_gate?: RiskGate;
  latest_signal: Signal | null;
  trades: Trade[];
  stats: { closed: number; wins: number; win_rate: number };
  news: NewsStatus;
  prediction_model?: ModelStatus;
  last_error: string | null;
};

export type Candle = {
  datetime: string;
  open: number;
  high: number;
  low: number;
  close: number;
};

export type CandleResponse = {
  symbol: string;
  interval: string;
  feed: string;
  candles: Candle[];
};

type PerformanceSlice = {
  trades: number;
  net_pnl: number;
  win_rate: number;
  profit_factor: number | null;
  expectancy_per_trade: number;
  breakeven_win_rate: number | null;
};

export type Backtest = {
  symbol: string;
  interval: string;
  feed: string;
  bars: number;
  assumptions: {
    strategy?: string;
    spread_pips: number;
    slippage_pips: number;
    higher_timeframe_filter: boolean;
    risk_percent?: number;
    max_concurrent_trades?: number;
    excluded_gates?: string[];
  };
  metrics: {
    starting_balance: number;
    ending_balance: number;
    net_pnl: number;
    return_percent: number;
    closed_trades: number;
    open_trades: number;
    win_rate: number;
    profit_factor: number | null;
    average_win: number;
    average_loss: number;
    max_drawdown_percent: number;
  };
  monthly: { month: string; trades: number; pnl: number }[];
  diagnostics: {
    overall: PerformanceSlice;
    by_side: Record<"BUY" | "SELL", PerformanceSlice>;
    by_hour_utc: (PerformanceSlice & { hour_utc: number })[];
    by_session?: Record<string, PerformanceSlice>;
    max_consecutive_losses: number;
  };
};
