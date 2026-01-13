import math
import logging

# -------------------------
# Small helpers (pure Python)
# -------------------------

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers (important when called in workers)
    logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(processName)s] %(levelname)s: %(message)s"))

    # File handler
    fh = logging.FileHandler("tier.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(processName)s] %(levelname)s: %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)

def proximity(level, price, pct):
    if level is None or (isinstance(level, float) and math.isnan(level)):
        return False
    return abs(price - level) / level <= pct

def parse_single_prints(sp_str):
    if not sp_str or (isinstance(sp_str, float) and math.isnan(sp_str)) or str(sp_str).strip() == "":
        return []
    try:
        return [float(x) for x in str(sp_str).split(",") if x.strip() != ""]
    except:
        return []

def dedup_levels(levels, min_dist=0.001):
    cleaned = []
    for lvl in sorted(levels):
        if not cleaned or abs(lvl - cleaned[-1]) / lvl > min_dist:
            cleaned.append(lvl)
    return cleaned

def compute_rr(entry, target, stop, direction):
    if direction == "long":
        num = target - entry
        den = entry - stop
    else:
        num = entry - target
        den = stop - entry

    if den == 0:
        return 0

    return num / den

def sign(x):
    if x > 0: return 1
    if x < 0: return -1
    return 0