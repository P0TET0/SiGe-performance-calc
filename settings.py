from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

# Required Windows path from the TODO requirements.
DEFAULT_ND_PICKLE_PATH = Path(
    r"C:\Users\miots\ruruproject\SiGe-performance\SiGe-performance-calc\data\N_D_values.pkl"
)
FALLBACK_ND_PICKLE_PATH = DATA_DIR / "N_D_values.pkl"
T_RANGE_PICKLE_PATH = DATA_DIR / "T_range.pkl"
XI_F_PICKLE_PATH = DATA_DIR / "xi_F_vals.pkl"

# Predefined composition options for the UI.
COMPOSITION_OPTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Fallback settings when temperature-dependent data is unavailable.
XI_F_FALLBACK_MIN = -20.0
XI_F_FALLBACK_MAX = 20.0
XI_F_FALLBACK_POINTS = 100
FALLBACK_TEMPERATURE_K = 300.0
