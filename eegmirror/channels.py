# --------------------------------------------------------
# EEGMirror
# Channel-name -> scalp-proportion (x%, y%) lookup used by the
# Montage-Agnostic Position Embedding (MAPE), see paper Sec 3.1.2.
# --------------------------------------------------------
import re

# Anterior (0, near nasion) -> Posterior (100, near inion) ring positions.
# Extra "transitional" rows used by the extended 10-10/10-5 system (CFC,
# CCP, FTT, TTP, TPP, ...) are placed halfway between their neighbours.
_ROW_Y = {
    'NZ': 0,
    'FP': 10,
    'AF': 20,
    'F': 30,
    'FFT': 35, 'FFC': 35,
    'FT': 40, 'FC': 40, 'CFC': 40,
    'FTT': 45, 'FCC': 45,
    'T': 50, 'C': 50,
    'TTP': 55, 'CCP': 55,
    'TP': 60, 'CP': 60,
    'TPP': 65, 'CPP': 65,
    'P': 70,
    'PPO': 75,
    'PO': 80,
    'POO': 85,
    'O': 90, 'CB': 90,
    'OI': 95,
    'IZ': 100,
    # non-cortical reference sites, approximated onto the same grid
    'M': 50,   # mastoid  ~ T row, most lateral
    'A': 50,   # ear lobe ~ T row, most lateral
}
# Longest prefixes must be tried first so e.g. "FTT" is not mis-parsed as "F".
_ROW_PREFIXES = sorted(_ROW_Y.keys(), key=len, reverse=True)

# A handful of channels that don't follow the regular "<row><index>" /
# "<row>Z" pattern, given explicit (x, y) proportions.
_SPECIAL = {
    'NZ': (50, 0), 'IZ': (50, 100),
    'M1': (0, 50), 'M2': (100, 50),
    'A1': (0, 50), 'A2': (100, 50),
    'CB1': (5, 92), 'CB2': (95, 92),
    'O9': (0, 90), 'O10': (100, 90),
    'T1': (5, 45), 'T2': (95, 45),
    'T3': (0, 50), 'T4': (100, 50),   # legacy names == T7 / T8
    'T5': (0, 70), 'T6': (100, 70),   # legacy names == P7 / P8
}


def _lateral_x(num: int) -> float:
    """Map the trailing 10-10 index to a left(0)-right(100) proportion.

    Convention: 'z' (num=0) is the midline (50). Odd numbers are on the
    left hemisphere, even numbers on the right, growing outward in 10%
    steps: 1/2 -> 40/60, 3/4 -> 30/70, 5/6 -> 20/80, 7/8 -> 10/90,
    9/10 -> 0/100 (most lateral, temporal chain).
    """
    if num == 0:
        return 50.0
    step = (num + 1) // 2 if num % 2 == 1 else num // 2
    x = 50.0 - 10.0 * step if num % 2 == 1 else 50.0 + 10.0 * step
    return float(min(max(x, 0.0), 100.0))


def _parse_single(name: str):
    name = name.upper().strip()
    if name in _SPECIAL:
        return _SPECIAL[name]

    core = name[:-1] if name.endswith('H') and not name.endswith('ZH') else name

    m = re.match(r'^([A-Z]+?)(\d+)$', core)
    if m:
        prefix, num = m.group(1), int(m.group(2))
        x = _lateral_x(num)
    elif core.endswith('Z'):
        prefix = core[:-1]
        x = 50.0
    else:
        prefix = core
        x = 50.0

    for cand in _ROW_PREFIXES:
        if prefix == cand:
            return x, float(_ROW_Y[cand])
    # Unknown row letters (arbitrary/novel channel name): fall back to the
    # vertex. MAPE degrades gracefully -- the channel is still assigned a
    # deterministic (if uninformative) position rather than crashing.
    return x, 50.0


def channel_to_xy(name: str):
    """Return the (x_proportion, y_proportion) in [0, 100] for `name`.

    Bipolar / derivation channels such as "FP1-F7" are placed at the
    midpoint of their two electrodes.
    """
    if '-' in name:
        a, b = name.split('-', 1)
        xa, ya = _parse_single(a)
        xb, yb = _parse_single(b)
        return (xa + xb) / 2.0, (ya + yb) / 2.0
    return _parse_single(name)


def channels_to_xy(names):
    """Vectorized helper: list[str] -> list[(x, y)]."""
    return [channel_to_xy(n) for n in names]
