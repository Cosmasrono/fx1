from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openrouter_api_key: SecretStr = Field(default=SecretStr(""), repr=False,
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "api_key"))
    openrouter_model: str = "openrouter/free"
    twelve_data_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("TWELVE_DATA_API_KEY", "Twelve_Data_key"),
    )
    use_synthetic_data: bool = False
    starting_balance: float = 10_000.0
    risk_percent: float = 1.0
    atr_stop_multiplier: float = 1.0
    atr_target_multiplier: float = 1.5
    # Keep paper fills comparable with the backtest. These are EUR/USD
    # assumptions, not live broker quotes.
    spread_pips: float = 0.8
    slippage_pips: float = 0.1
    pip_size: float = 0.0001
    scan_seconds: int = 60
    # Comma-separated browser origins permitted to call the local API. The
    # localhost regex in main.py also covers 127.0.0.1, which is a common way
    # to open a local Next.js dashboard.
    allowed_origins: str = Field(
        default="http://localhost:3000",
        validation_alias=AliasChoices("ALLOWED_ORIGINS", "ALLOWED_ORIGIN"),
    )
    symbol: str = "EUR/USD"
    interval: str = "15min"
    higher_timeframe_filter: bool = True
    weekly_interval: str = "1week"
    monthly_interval: str = "1month"
    higher_timeframe_cache_minutes: int = 60
    adx_min: float = 20.0
    setup_score_min: int = 70
    # "classic" is the EMA/MACD/RSI/momentum confluence model; "smc" is the
    # liquidity-sweep / structure-shift / order-block model in smc.py.
    strategy: str = "classic"
    smc_swing_strength: int = 2
    # How many bars a sweep stays actionable before the setup is considered stale.
    smc_sweep_window: int = 20
    smc_displacement_atr: float = 0.6
    smc_stop_buffer_atr: float = 0.25
    smc_score_min: int = 70
    # Fall back to this R multiple when the next pool is too close. This is
    # a target placement rule, not evidence that price will reach that target.
    smc_min_reward: float = 1.5
    # ICT Asian range / Judas swing (STRATEGY=judas), hours in UTC. The Asian
    # range is built from range_start to range_end; London may sweep it and
    # close back inside from range_end until window_end.
    judas_range_start_hour: int = 0
    judas_range_end_hour: int = 6
    judas_window_end_hour: int = 10
    judas_max_range_pips: float = 30.0
    judas_rejection_atr: float = 0.3
    judas_stop_buffer_atr: float = 0.25
    judas_min_reward: float = 1.5
    judas_score_min: int = 80
    max_daily_loss_percent: float = 2.0
    max_weekly_loss_percent: float = 5.0
    max_consecutive_losses: int = Field(default=3, ge=1)
    loss_cooldown_hours: float = Field(default=24.0, gt=0)
    max_concurrent_trades: int = Field(
        default=3, ge=1,
        validation_alias=AliasChoices("MAX_CONCURRENT_TRADES", "max_concurrent_trades"),
    )
    # Optional high-impact economic-calendar guard. Financial Modeling Prep's
    # free Basic plan is sufficient because the calendar is cached hourly.
    news_filter_enabled: bool = False
    fmp_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("FMP_API_KEY", "FINANCIAL_MODELING_PREP_API_KEY"),
    )
    news_blackout_minutes_before: int = 30
    news_blackout_minutes_after: int = 15
    news_cache_minutes: int = 60
    news_fail_closed: bool = True
    pattern_min_loss_matches: int = 3
    pattern_similarity_threshold: float = 0.85
    pattern_lookback_days: int = Field(default=90, ge=1)
    # Optional local logistic model. It estimates whether a rule-based setup
    # reaches its configured target before its stop; it never creates a trade
    # direction by itself.
    prediction_model_enabled: bool = True
    prediction_auto_retrain: bool = True
    prediction_retrain_new_trades: int = Field(default=5, ge=1)
    prediction_retrain_hours: float = Field(default=6.0, gt=0)
    market_max_age_bars: int = Field(default=3, ge=1)
    # Unset uses the threshold training validated out of sample. Set it only to
    # deliberately override that choice.
    prediction_min_probability: float | None = None
    prediction_min_samples: int = 200
    prediction_max_horizon_bars: int = 96
    # The second path supports existing local projects that placed the key in
    # the dashboard environment file. New setups should use backend/.env.
    model_config = SettingsConfigDict(env_file=(".env", "../dashboard/.env.local"), extra="ignore")


settings = Settings()
