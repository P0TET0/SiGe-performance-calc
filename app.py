from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import simulator

st.set_page_config(page_title="SiGe 熱電性能シミュレーター", layout="centered")

PLOT_SPECS: list[dict[str, Any]] = [
    {
        "key": "zt",
        "title": "ZT",
        "y_label": "無次元性能指数 ZT",
        "scale": 1.0,
        "color": "#005f73",
        "clip_bottom_zero": True,
    },
    {
        "key": "sigma",
        "title": "電気伝導率",
        "y_label": "電気伝導率 σ [S/m]",
        "scale": 1.0,
        "color": "#0a9396",
        "clip_bottom_zero": True,
    },
    {
        "key": "alpha",
        "title": "ゼーベック係数",
        "y_label": "ゼーベック係数 α [mV/K]",
        "scale": 1000.0,  # V/K -> mV/K
        "color": "#ae2012",
        "clip_bottom_zero": False,
    },
    {
        "key": "lorenz",
        "title": "ローレンツ数",
        "y_label": "ローレンツ数 L [WΩ/K^2]",
        "scale": 1.0,
        "color": "#bb3e03",
        "clip_bottom_zero": True,
    },
    {
        "key": "kappa",
        "title": "熱伝導率",
        "y_label": "熱伝導率 κ [W/mK]",
        "scale": 1.0,
        "color": "#ca6702",
        "clip_bottom_zero": True,
    },
]


@st.cache_data(show_spinner=False)
def run_simulation_cached(
    composition: float,
    nd_value: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    return simulator.simulate_properties(composition, nd_value)


def build_plot(
    x_axis: np.ndarray,
    y_values: np.ndarray,
    x_label: str,
    y_label: str,
    plot_title: str,
    composition: float,
    nd_value: float,
    color: str,
    clip_bottom_zero: bool,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9, 5.2))
    axis.plot(x_axis, y_values, color=color, linewidth=2.4)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(
        f"{plot_title} | 組成比(Si$_{{1-y}}$Ge$_{{y}}$) y={composition:.2f}, N_D={nd_value:.2e}"
    )
    axis.grid(True, linestyle=":", alpha=0.6)

    if clip_bottom_zero:
        axis.set_ylim(bottom=0)

    finite_mask = np.isfinite(y_values)
    if finite_mask.any():
        max_index = int(np.nanargmax(y_values))
        axis.scatter(
            [x_axis[max_index]],
            [y_values[max_index]],
            color="#001219",
            s=36,
            zorder=3,
        )

    figure.tight_layout()
    return figure


def figure_to_png_bytes(figure: plt.Figure) -> bytes:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    buffer.seek(0)
    return buffer.read()


def build_download_filename(composition: float, nd_value: float, key: str) -> str:
    comp = f"{composition:.2f}".replace(".", "p")
    nd = f"{nd_value:.2e}".replace("+", "")
    return f"{key}_y{comp}_nd{nd}.png"


st.title("SiGe 熱電性能シミュレーター")

try:
    composition_options = simulator.get_composition_candidates()
except Exception as error:
    st.error(f"組成比候補の読み込みに失敗しました: {error}")
    st.stop()

if not composition_options:
    st.error("組成比の候補がありません。`settings.py` を確認してください。")
    st.stop()

composition = st.selectbox(
    "組成比",
    options=composition_options,
    format_func=lambda value: f"y = {value:.2f}",
)

try:
    nd_options = simulator.load_nd_candidates(float(composition))
except Exception as error:
    st.error(f"N_D 候補の読み込みに失敗しました: {error}")
    st.stop()

if not nd_options:
    st.error("選択した組成比に対する N_D 候補が見つかりませんでした。")
    st.stop()

nd_value = st.selectbox(
    "N_D",
    options=nd_options,
    format_func=lambda value: f"{value:.3e}",
)

calculate_button = st.button("計算して表示", type="primary")

if calculate_button:
    try:
        with st.spinner("物性値を計算中です..."):
            x_axis, curves, labels = run_simulation_cached(
                float(composition), float(nd_value)
            )

        mode = str(labels.get("mode", "temperature"))
        x_label = str(labels.get("x_label", "X"))
        if mode == "xi_f":
            temp_k = float(labels.get("temperature_k", 300.0))
            st.warning(
                f"T_range/xi_F データを使えなかったため、"
                f"横軸を xi_F (T={temp_k:.1f} K 固定) として表示しています。"
            )

        for spec in PLOT_SPECS:
            key = str(spec["key"])
            if key not in curves:
                continue

            raw_values = np.asarray(curves[key], dtype=float)
            y_values = raw_values * float(spec["scale"])

            st.subheader(str(spec["title"]))
            figure = build_plot(
                x_axis=x_axis,
                y_values=y_values,
                x_label=x_label,
                y_label=str(spec["y_label"]),
                plot_title=str(spec["title"]),
                composition=float(composition),
                nd_value=float(nd_value),
                color=str(spec["color"]),
                clip_bottom_zero=bool(spec["clip_bottom_zero"]),
            )
            st.pyplot(figure)

            finite_values = y_values[np.isfinite(y_values)]
            if finite_values.size > 0:
                st.caption(
                    f"min: {float(np.nanmin(finite_values)):.4e}, "
                    f"max: {float(np.nanmax(finite_values)):.4e}"
                )
            else:
                st.caption("有効な計算値を取得できませんでした。")

            png_bytes = figure_to_png_bytes(figure)
            filename = build_download_filename(float(composition), float(nd_value), key)
            st.download_button(
                label=f"{spec['title']}のPNGをダウンロード",
                data=png_bytes,
                file_name=filename,
                mime="image/png",
                key=f"download_{key}",
            )
            plt.close(figure)

    except Exception as error:
        st.error(f"計算中にエラーが発生しました: {error}")
        st.code(str(error))

try:
    nd_path = simulator.get_nd_pickle_path()
    st.caption(f"N_D候補読み込み元: {Path(nd_path)}")
except Exception:
    pass
