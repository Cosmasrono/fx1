from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
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
    scan_seconds: int = 300
    # Comma-separated browser origins permitted to call the local API. The
    # localhost regex in main.py also covers 127.0.0.1, which is a common way
    # to open a local Next.js dashboard.
    allowed_origins: str = Field(
        default="http://localhost:3000",
        validation_alias=AliasChoices("ALLOWED_ORIGINS", "ALLOWED_ORIGIN"),
    )
    symbol: str = "EUR/USD"
    interval: str = "5min"
    higher_timeframe_filter: bool = True
    weekly_interval: str = "1week"
    monthly_interval: str = "1month"
    higher_timeframe_cache_minutes: int = 60
    adx_min: float = 20.0
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
    # The second path supports existing local projects that placed the key in
    # the dashboard environment file. New setups should use backend/.env.
    model_config = SettingsConfigDict(env_file=(".env", "../dashboard/.env.local"), extra="ignore")


settings = Settings()
