import yaml
import enum
from dataclasses import dataclass, field


# ---------- File groups per asset ----------
@dataclass
class MODE(enum.Enum):
    """Represents the possible states."""
    REALTIME = 'REALTIME'
    DATASET = 'DATASET'
    DEBUG = 'DEBUG'

@dataclass
class AssetFiles:
    vp: str = ""
    deriv: str = ""
    bars: str = ""
    rollingTPO: str = ""
    daily: str = ""
    composite: str = ""
    out_trades: str = ""
    out_summary: str = ""

# ---------- Non-asset files ----------

@dataclass
class BaseFiles:
    db: str = "database/btc_historical.db"
    state: str = "config/last_state.json"


# ---------- Parameters (same as before, default values preserved) ----------

@dataclass
class Params:
    prox_pct: float = 0.002
    max_rr_allowed: float = 15.0
    min_rr_allowed: float = 0.0

    lookahead: dict = field(default_factory=lambda: {"s_days": 3, "a_days": 1, "b_hours": 4})
    bars: dict = field(default_factory=lambda: {"per_day": 96, "per_hour": 4})

    session_weight: dict = field(default_factory=lambda: {
        "Asia": 0.15, "London": 0.05, "NewYork": 0.25, "Close": 0.25
    })

    tier_filters: dict = field(default_factory=lambda: {
        "S+": {"min_rr": 1.0, "min_dist_pct": 0.001,  "max_dist_pct": 0.02, "max_tgt_dist_pct": 0.025},
        "S":  {"min_rr": 1.0, "min_dist_pct": 0.001,  "max_dist_pct": 0.02, "max_tgt_dist_pct": 0.025},
        "A":  {"min_rr": 0.8, "min_dist_pct": 0.0008, "max_dist_pct": 0.03, "max_tgt_dist_pct": 0.020},
        "B":  {"min_rr": 0.5, "min_dist_pct": 0.0005, "max_dist_pct": 0.04, "max_tgt_dist_pct": 0.015},
        "C":  {"min_rr": 0.25,"min_dist_pct": 0.0002, "max_dist_pct": 0.05, "max_tgt_dist_pct": 0.010},
    })

    pts: dict = field(default_factory=lambda: {
        "rotation_trigger": 0.5,
        "breakout_trigger": 1.5,
        "rotation_va_reject": 2.0,
        "candle_hammer": 2.0,
        "candle_shooting_star": 2.0,
        "candle_marubozu": 3.0,
        "candle_doji_penalty": 0.75,
        "candle_engulfing": 2.5,
        "volume_spike": 2.5,
        "vwap_touch": 0.5,
        "ib_touch": 0.25,
        "session_bias": 0.25,
        "opening_type": 0.25,
        "value_migration": 0.25,
        "flow_confirm": 0.5
    })

    tier_score: dict = field(default_factory=lambda: {
        "S": 12,
        "A": 8,
        "B": 4
    })


# ---------- Top-level Config ----------

@dataclass
class Config:
    mode: MODE = MODE.DATASET
    btc_files: AssetFiles = field(default_factory=AssetFiles)
    eth_files: AssetFiles = field(default_factory=AssetFiles)
    sol_files: AssetFiles = field(default_factory=AssetFiles)

    files: BaseFiles = field(default_factory=BaseFiles)
    params: Params = field(default_factory=Params)


# ---------- Loader ----------

def update_dataclass_from_dict(obj, values: dict):
    """
    Recursively assign matching keys from dict onto dataclass.
    """
    for k, v in values.items():
        if hasattr(obj, k):
            setattr(obj, k, v)


def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    cfg = Config()

    # asset-specific blocks
    if "btc_files" in data:
        update_dataclass_from_dict(cfg.btc_files, data["btc_files"])

    if "eth_files" in data:
        update_dataclass_from_dict(cfg.eth_files, data["eth_files"])

    if "sol_files" in data:
        update_dataclass_from_dict(cfg.sol_files, data["sol_files"])

    # generic files
    if "files" in data:
        update_dataclass_from_dict(cfg.files, data["files"])

    # parameters (full override allowed)
    if "params" in data:
        update_dataclass_from_dict(cfg.params, data["params"])

    if data["mode"] == "realtime":
        cfg.mode = MODE.REALTIME
    elif data["mode"] == "dataset":
        cfg.mode = MODE.DATASET
    elif data["mode"] == "debug":
        cfg.mode = MODE.DEBUG

    return cfg
