from __future__ import annotations

import pickle
import re
from functools import lru_cache
import math
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np
from scipy.integrate import quad
from scipy.constants import N_A as N_A_CONST
from scipy.constants import e, h, k, pi

from settings import (
    COMPOSITION_OPTIONS,
    DEFAULT_ND_PICKLE_PATH,
    FALLBACK_ND_PICKLE_PATH,
    FALLBACK_TEMPERATURE_K,
    T_RANGE_PICKLE_PATH,
    XI_F_FALLBACK_MAX,
    XI_F_FALLBACK_MIN,
    XI_F_FALLBACK_POINTS,
    XI_F_PICKLE_PATH,
)

mp.mp.dps = 40

HBAR = mp.mpf(h) / (2 * mp.pi)
KB = mp.mpf(k)
Q = mp.mpf(e)
PI = mp.mpf(pi)
N_A = mp.mpf(N_A_CONST)
M_STAR_E = mp.mpf(1.4 * 9.11e-31)
M_STAR_H = mp.mpf(1.4 * 9.11e-31)
E_G = mp.mpf(0.910022) * Q

S_PARAMETER = mp.mpf(1.5)
BETA = mp.mpf(2.0)
GAMMA = mp.mpf(0.91)
DEFORMATION_POTENTIAL = mp.mpf(2.94 * e)
SMALL = mp.mpf("1e-30")


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def _parse_numeric(value: Any) -> float | None:
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value.strip())
        if match:
            return float(match.group(0))
    return None


def _to_float_list(values: Any) -> list[float]:
    if isinstance(values, np.ndarray):
        flat = values.reshape(-1).tolist()
        return [float(item) for item in flat]
    if isinstance(values, (list, tuple)):
        return [float(item) for item in values]
    raise TypeError(f"Unsupported candidate container: {type(values)}")


def _to_float_array_1d(values: Any) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values.astype(float).reshape(-1)
    if isinstance(values, (list, tuple)):
        return np.asarray([float(item) for item in values], dtype=float)
    return np.asarray([float(values)], dtype=float)


def _resolve_nd_source_path() -> Path:
    if DEFAULT_ND_PICKLE_PATH.exists():
        return DEFAULT_ND_PICKLE_PATH
    if FALLBACK_ND_PICKLE_PATH.exists():
        return FALLBACK_ND_PICKLE_PATH
    raise FileNotFoundError(
        "N_D_values.pkl was not found at the required path:\n"
        f"{DEFAULT_ND_PICKLE_PATH}"
    )


def get_nd_pickle_path() -> Path:
    return _resolve_nd_source_path()


@lru_cache(maxsize=1)
def _load_nd_source() -> Any:
    return _read_pickle(_resolve_nd_source_path())


@lru_cache(maxsize=1)
def _load_temperature_source() -> np.ndarray | None:
    if not T_RANGE_PICKLE_PATH.exists():
        return None
    try:
        return _to_float_array_1d(_read_pickle(T_RANGE_PICKLE_PATH))
    except Exception:
        return None


@lru_cache(maxsize=1)
def _load_xi_f_source() -> Any:
    if not XI_F_PICKLE_PATH.exists():
        return None
    try:
        return _read_pickle(XI_F_PICKLE_PATH)
    except Exception:
        return None


def _match_composition_key(mapping: dict[Any, Any], composition: float) -> Any:
    if composition in mapping:
        return composition
    if str(composition) in mapping:
        return str(composition)
    for fmt in (f"{composition:g}", f"{composition:.1f}", f"{composition:.2f}"):
        if fmt in mapping:
            return fmt

    numeric_keys: list[tuple[float, Any]] = []
    for key in mapping:
        numeric = _parse_numeric(key)
        if numeric is not None:
            numeric_keys.append((numeric, key))
    if numeric_keys:
        return min(numeric_keys, key=lambda item: abs(item[0] - composition))[1]

    return next(iter(mapping))


def get_composition_candidates() -> list[float]:
    raw = _load_nd_source()
    if isinstance(raw, dict):
        numeric_values: list[float] = []
        for key in raw:
            parsed = _parse_numeric(key)
            if parsed is not None:
                numeric_values.append(parsed)
        if numeric_values:
            unique_sorted = sorted(set(numeric_values))
            return unique_sorted
    return list(COMPOSITION_OPTIONS)


def load_nd_candidates(composition: float) -> list[float]:
    raw = _load_nd_source()
    if isinstance(raw, dict):
        key = _match_composition_key(raw, composition)
        return _to_float_list(raw[key])
    if isinstance(raw, (list, tuple, np.ndarray)):
        return _to_float_list(raw)
    raise TypeError(
        "N_D_values.pkl must be list, numpy.ndarray, tuple, or dict."
    )


def _closest_nd_index(nd_value: float, nd_candidates: list[float]) -> int:
    if not nd_candidates:
        raise ValueError("No N_D candidate is available.")
    nd_array = np.asarray(nd_candidates, dtype=float)
    return int(np.abs(nd_array - float(nd_value)).argmin())


def _extract_xi_row(
    xi_source: Any,
    composition: float,
    nd_value: float,
    nd_candidates: list[float],
) -> np.ndarray | None:
    current = xi_source
    if isinstance(xi_source, dict):
        key = _match_composition_key(xi_source, composition)
        current = xi_source[key]

    if current is None:
        return None

    if isinstance(current, np.ndarray):
        array = current.astype(float, copy=False)
    else:
        array = np.asarray(current, dtype=object)

    if array.ndim == 1:
        if array.size == 0:
            return None
        first = array[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            index = _closest_nd_index(nd_value, nd_candidates)
            if index >= array.size:
                return None
            return _to_float_array_1d(array[index])
        return _to_float_array_1d(array)

    if array.ndim >= 2:
        index = _closest_nd_index(nd_value, nd_candidates)
        if index >= array.shape[0]:
            return None
        return _to_float_array_1d(array[index])

    return None


def _resolve_temperature_mode_inputs(
    composition: float,
    nd_value: float,
    nd_candidates: list[float],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    t_values = _load_temperature_source()
    xi_source = _load_xi_f_source()
    if t_values is None or xi_source is None:
        return None, None

    xi_values = _extract_xi_row(xi_source, composition, nd_value, nd_candidates)
    if xi_values is None:
        return None, None

    if len(t_values) != len(xi_values):
        return None, None
    return t_values, xi_values


def _xi_g(t: mp.mpf) -> mp.mpf:
    return E_G / (KB * t)


def _a_cubed(y: mp.mpf) -> mp.mpf:
    return mp.power(2.7155e-10, 3) * (1 - y) + mp.power(2.8288e-10, 3) * y


def _a(y: mp.mpf) -> mp.mpf:
    return mp.power(_a_cubed(y), mp.mpf(1) / 3)


def _m_g(y: mp.mpf) -> mp.mpf:
    return mp.mpf(28.086) * (1 - y) + mp.mpf(72.59) * y


def _m_kg(y: mp.mpf) -> mp.mpf:
    return _m_g(y) * mp.mpf("1e-3")


def _g_elastic(y: mp.mpf) -> mp.mpf:
    return (mp.mpf(1.033) * (1 - y) + mp.mpf(1.017) * y) * mp.mpf("1e-3")


def _theta(y: mp.mpf) -> mp.mpf:
    return mp.mpf("1.48e-8") * _a(y) ** (-mp.mpf(3) / 2) * _m_g(y) ** (
        -mp.mpf(1) / 2
    ) * _g_elastic(y)


def _v_s(y: mp.mpf) -> mp.mpf:
    return (KB / HBAR) * (6 * mp.pi**2) ** (-mp.mpf(1) / 3) * _theta(y) * _a(y)


def _rho_d(y: mp.mpf) -> mp.mpf:
    return _m_kg(y) / (_a_cubed(y) * N_A)


def _tau_n_inv(y: mp.mpf, xi: mp.mpf, t: mp.mpf) -> mp.mpf:
    return _tau_n_prefactor(y, t) * xi**2


def _tau_n_prefactor(y: mp.mpf, t: mp.mpf) -> mp.mpf:
    coeff = ((20 * mp.pi) / 3) * HBAR * N_A * ((6 * mp.pi**2) / 4) ** (mp.mpf(1) / 3)
    ratio = BETA * (1 + (mp.mpf(5) / 9) * BETA) / (1 + BETA)
    return (
        coeff
        * ratio
        * GAMMA**2
        / (_m_kg(y) * _a(y) ** 2)
        * (t / _theta(y)) ** 3
    )


def _tau_u_inv(y: mp.mpf, xi: mp.mpf, t: mp.mpf) -> mp.mpf:
    return _tau_n_inv(y, xi, t) / BETA


def _tau_c(y: mp.mpf, xi: mp.mpf, t: mp.mpf) -> mp.mpf:
    inv_value = _tau_n_inv(y, xi, t) + _tau_u_inv(y, xi, t)
    if abs(inv_value) < SMALL:
        return mp.mpf("inf")
    return 1 / inv_value


def _tau_integrand(x: mp.mpf) -> mp.mpf:
    return x**4 * mp.exp(x) / mp.expm1(x) ** 2


@lru_cache(maxsize=1024)
def _phonon_integrals_cached(y_float: float, t_float: float) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    y = mp.mpf(y_float)
    t = mp.mpf(t_float)
    upper = float(_theta(y) / t)
    if not np.isfinite(upper) or upper <= 0:
        return mp.nan, mp.nan, mp.nan

    tau_n_prefactor = _tau_n_prefactor(y, t)
    if abs(tau_n_prefactor) < SMALL:
        return mp.nan, mp.nan, mp.nan

    lower = 1e-8

    def bose_factor(x: float) -> float:
        exp_x = math.exp(x)
        expm1_x = math.expm1(x)
        return exp_x / (expm1_x * expm1_x)

    def j_integral(power: int) -> mp.mpf:
        integrand = lambda xi: (xi**power) * bose_factor(xi)
        value, _ = quad(integrand, lower, upper, limit=300)
        return mp.mpf(value)

    j2 = j_integral(2)
    j4 = j_integral(4)
    j6 = j_integral(6)

    tau_c_prefactor = 1 / (tau_n_prefactor * (1 + 1 / BETA))
    i1 = tau_c_prefactor * j2
    i2 = tau_c_prefactor * tau_n_prefactor * j4
    i3 = tau_n_prefactor * (1 - (tau_c_prefactor * tau_n_prefactor)) * j6
    return i1, i2, i3


def _tau_dp_e(y: mp.mpf, t: mp.mpf) -> mp.mpf:
    numerator = mp.pi * _rho_d(y) * HBAR**4 * _v_s(y) ** 2
    denominator = mp.sqrt(2) * DEFORMATION_POTENTIAL**2 * (M_STAR_E * KB * t) ** (
        mp.mpf(3) / 2
    )
    return numerator / denominator


def _tau_dp_h(y: mp.mpf, t: mp.mpf) -> mp.mpf:
    numerator = mp.pi * _rho_d(y) * HBAR**4 * _v_s(y) ** 2
    denominator = mp.sqrt(2) * DEFORMATION_POTENTIAL**2 * (M_STAR_H * KB * t) ** (
        mp.mpf(3) / 2
    )
    return numerator / denominator


@lru_cache(maxsize=16384)
def _fermi_dirac_cached(s_float: float, xi_f_float: float) -> mp.mpf:
    s = mp.mpf(s_float)
    xi_f = mp.mpf(xi_f_float)

    def integrand(x: mp.mpf) -> mp.mpf:
        return x**s / (mp.exp(x - xi_f) + 1)

    return mp.quad(integrand, [0, mp.inf])


def fermi_dirac(s: mp.mpf, xi_f: mp.mpf) -> mp.mpf:
    return _fermi_dirac_cached(float(s), float(xi_f))


def _delta(s: mp.mpf, xi_f: mp.mpf) -> mp.mpf:
    num = (s + mp.mpf(5) / 2) * fermi_dirac(s + mp.mpf(3) / 2, xi_f)
    den = (s + mp.mpf(3) / 2) * fermi_dirac(s + mp.mpf(1) / 2, xi_f)
    return num / den


def _delta_capital(s: mp.mpf, xi_f: mp.mpf) -> mp.mpf:
    term = (s + mp.mpf(7) / 2) * fermi_dirac(s + mp.mpf(5) / 2, xi_f)
    den = (s + mp.mpf(3) / 2) * fermi_dirac(s + mp.mpf(1) / 2, xi_f)
    return term / den - _delta(s, xi_f) ** 2


def _sigma1(s: mp.mpf, xi_f: mp.mpf, n_b: mp.mpf, m_star: mp.mpf, tau_value: mp.mpf) -> mp.mpf:
    return (
        (4 * Q**2 * n_b * tau_value) / (3 * mp.sqrt(PI) * m_star)
    ) * (s + mp.mpf(3) / 2) * fermi_dirac(s + mp.mpf(1) / 2, xi_f)


def _alpha1(s: mp.mpf, xi_f: mp.mpf) -> mp.mpf:
    return (KB / Q) * (_delta(s, xi_f) - xi_f)


def _l1(s: mp.mpf, xi_f: mp.mpf) -> mp.mpf:
    return (KB / Q) ** 2 * _delta_capital(s, xi_f)


def _n_b(m_star: mp.mpf, t: mp.mpf) -> mp.mpf:
    return 2 * (m_star * KB * t / (2 * PI * HBAR**2)) ** (mp.mpf(3) / 2)


def _n_c(t: mp.mpf) -> mp.mpf:
    return _n_b(M_STAR_E, t)


def _n_v(t: mp.mpf) -> mp.mpf:
    return _n_b(M_STAR_H, t)


def _alpha2_e(s: mp.mpf, xi_f: mp.mpf) -> mp.mpf:
    return -_alpha1(s, xi_f)


def _alpha2_h(s: mp.mpf, xi_f: mp.mpf, t: mp.mpf) -> mp.mpf:
    return _alpha1(s, -xi_f - _xi_g(t))


def _sigma2_e(s: mp.mpf, xi_f: mp.mpf, t: mp.mpf, y: mp.mpf) -> mp.mpf:
    return _sigma1(s, xi_f, _n_c(t), M_STAR_E, _tau_dp_e(y, t))


def _sigma2_h(s: mp.mpf, xi_f: mp.mpf, t: mp.mpf, y: mp.mpf) -> mp.mpf:
    return _sigma1(s, -xi_f - _xi_g(t), _n_v(t), M_STAR_H, _tau_dp_h(y, t))


def _l2_e(s: mp.mpf, xi_f: mp.mpf) -> mp.mpf:
    return _l1(s, xi_f)


def _l2_h(s: mp.mpf, xi_f: mp.mpf, t: mp.mpf) -> mp.mpf:
    return _l1(s, -xi_f - _xi_g(t))


def _sigma2(s: mp.mpf, xi_f: mp.mpf, t: mp.mpf, y: mp.mpf) -> mp.mpf:
    return _sigma2_e(s, xi_f, t, y) + _sigma2_h(s, xi_f, t, y)


def _alpha2(s: mp.mpf, xi_f: mp.mpf, t: mp.mpf, y: mp.mpf) -> mp.mpf:
    sigma_e = _sigma2_e(s, xi_f, t, y)
    sigma_h = _sigma2_h(s, xi_f, t, y)
    sigma_total = sigma_e + sigma_h
    if abs(sigma_total) < SMALL:
        return mp.nan
    return (_alpha2_e(s, xi_f) * sigma_e + _alpha2_h(s, xi_f, t) * sigma_h) / sigma_total


def _l2(s: mp.mpf, xi_f: mp.mpf, t: mp.mpf, y: mp.mpf) -> mp.mpf:
    sigma_e = _sigma2_e(s, xi_f, t, y)
    sigma_h = _sigma2_h(s, xi_f, t, y)
    sigma_total = sigma_e + sigma_h
    if abs(sigma_total) < SMALL:
        return mp.nan

    electronic_term = (_l2_e(s, xi_f) * sigma_e + _l2_h(s, xi_f, t) * sigma_h) / sigma_total
    bipolar_term = (
        sigma_e
        * sigma_h
        * (_alpha2_e(s, xi_f) - _alpha2_h(s, xi_f, t)) ** 2
        / sigma_total**2
    )
    return electronic_term + bipolar_term


def _kappa2_e(s: mp.mpf, xi_f: mp.mpf, t: mp.mpf, y: mp.mpf) -> mp.mpf:
    return _l2(s, xi_f, t, y) * t * _sigma2(s, xi_f, t, y)


def _kappa2_l(s: mp.mpf, xi_f: mp.mpf, t: mp.mpf, y: mp.mpf) -> mp.mpf:
    del s, xi_f
    i1, i2, i3 = _phonon_integrals_cached(float(y), float(t))
    if abs(i3) < SMALL:
        return mp.nan
    prefactor = (KB / (2 * mp.pi**2 * _v_s(y))) * ((KB * t) / HBAR) ** 3
    return prefactor * (i1 + (i2**2) / i3)


def _kappa2(s: mp.mpf, xi_f: mp.mpf, t: mp.mpf, y: mp.mpf) -> mp.mpf:
    return _kappa2_e(s, xi_f, t, y) + _kappa2_l(s, xi_f, t, y)


def _zt_value(y: mp.mpf, xi_f: mp.mpf, t: mp.mpf) -> float:
    try:
        alpha_value = _alpha2(S_PARAMETER, xi_f, t, y)
        sigma_value = _sigma2(S_PARAMETER, xi_f, t, y)
        kappa_value = _kappa2(S_PARAMETER, xi_f, t, y)
        if not mp.isfinite(kappa_value) or abs(kappa_value) < SMALL:
            return float("nan")
        zt = alpha_value**2 * sigma_value * t / kappa_value
        if not mp.isfinite(zt):
            return float("nan")
        return float(zt)
    except Exception:
        return float("nan")


def simulate_zt(
    composition: float,
    nd_value: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    composition = float(composition)
    nd_value = float(nd_value)
    nd_candidates = load_nd_candidates(composition)
    y = mp.mpf(composition)

    t_range, xi_values = _resolve_temperature_mode_inputs(
        composition, nd_value, nd_candidates
    )
    if t_range is not None and xi_values is not None:
        x_axis = np.asarray(t_range, dtype=float)
        zt_values = np.asarray(
            [
                _zt_value(y, mp.mpf(float(xi_f)), mp.mpf(float(temp_k)))
                for xi_f, temp_k in zip(xi_values, x_axis, strict=True)
            ],
            dtype=float,
        )
        labels = {
            "mode": "temperature",
            "x_label": "Temperature [K]",
            "y_label": "ZT",
            "composition": composition,
            "nd_value": nd_value,
        }
        return x_axis, zt_values, labels

    xi_f_axis = np.linspace(
        XI_F_FALLBACK_MIN, XI_F_FALLBACK_MAX, XI_F_FALLBACK_POINTS, dtype=float
    )
    fixed_temp = mp.mpf(FALLBACK_TEMPERATURE_K)
    zt_values = np.asarray(
        [_zt_value(y, mp.mpf(float(xi_f)), fixed_temp) for xi_f in xi_f_axis],
        dtype=float,
    )
    labels = {
        "mode": "xi_f",
        "x_label": "xi_F",
        "y_label": "ZT",
        "composition": composition,
        "nd_value": nd_value,
        "temperature_k": float(FALLBACK_TEMPERATURE_K),
    }
    return xi_f_axis, zt_values, labels
