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
        "clip_bottom_zero": True,
    },
    {
        "key": "sigma",
        "title": "電気伝導率",
        "y_label": "電気伝導率 σ [S/m]",
        "scale": 1.0,
        "clip_bottom_zero": True,
    },
    {
        "key": "alpha",
        "title": "ゼーベック係数",
        "y_label": "ゼーベック係数 α [mV/K]",
        "scale": 1000.0,  # V/K -> mV/K
        "clip_bottom_zero": False,
    },
    {
        "key": "lorenz",
        "title": "ローレンツ数",
        "y_label": "ローレンツ数 L [WΩ/K^2]",
        "scale": 1.0,
        "clip_bottom_zero": True,
    },
    {
        "key": "kappa",
        "title": "熱伝導率",
        "y_label": "熱伝導率 κ [W/mK]",
        "scale": 1.0,
        "clip_bottom_zero": True,
    },
]

LINE_COLORS = [
    "#005f73",
    "#0a9396",
    "#94d2bd",
    "#ee9b00",
    "#ca6702",
    "#bb3e03",
    "#ae2012",
    "#9b2226",
]
LINE_STYLES = ["-", "--", "-.", ":"]


@st.cache_data(show_spinner=False)
def run_simulation_cached(
    composition: float,
    nd_value: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    return simulator.simulate_properties(composition, nd_value)


def build_plot(
    x_axis: np.ndarray,
    series_by_nd: dict[float, np.ndarray],
    nd_values: list[float],
    x_label: str,
    y_label: str,
    plot_title: str,
    composition: float,
    clip_bottom_zero: bool,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9, 5.2))

    for index, nd_value in enumerate(nd_values):
        y_values = series_by_nd[nd_value]
        color = LINE_COLORS[index % len(LINE_COLORS)]
        linestyle = LINE_STYLES[index % len(LINE_STYLES)]
        axis.plot(
            x_axis,
            y_values,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=f"N_D = {nd_value:.2e}",
        )

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(
        f"{plot_title} | 組成比(Si$_{{1-y}}$Ge$_{{y}}$) y={composition:.2f}"
    )
    axis.grid(True, linestyle=":", alpha=0.6)
    axis.legend(fontsize=9)
    if clip_bottom_zero:
        axis.set_ylim(bottom=0)

    figure.tight_layout()
    return figure


def figure_to_png_bytes(figure: plt.Figure) -> bytes:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    buffer.seek(0)
    return buffer.read()


def build_download_filename(composition: float, nd_values: list[float], key: str) -> str:
    comp = f"{composition:.2f}".replace(".", "p")
    if len(nd_values) == 1:
        nd_text = f"{nd_values[0]:.2e}".replace("+", "")
        nd_token = f"nd{nd_text}"
    else:
        nd_token = f"{len(nd_values)}nds"
    return f"{key}_y{comp}_{nd_token}.png"


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

default_count = min(3, len(nd_options))
selected_nd_values = st.multiselect(
    "N_D（複数選択可）",
    options=nd_options,
    default=nd_options[:default_count],
    format_func=lambda value: f"{value:.3e}",
)

calculate_button = st.button("計算して表示", type="primary")

if calculate_button:
    if not selected_nd_values:
        st.error("N_D を1つ以上選択してください。")
        st.stop()

    try:
        with st.spinner("物性値を計算中です..."):
            result_by_nd: dict[float, dict[str, Any]] = {}
            base_x_axis: np.ndarray | None = None
            base_labels: dict[str, Any] | None = None
            mode_set: set[str] = set()

            for nd_value in selected_nd_values:
                x_axis, curves, labels = run_simulation_cached(
                    float(composition), float(nd_value)
                )

                if base_x_axis is None:
                    base_x_axis = x_axis
                    base_labels = labels
                else:
                    if len(base_x_axis) != len(x_axis) or not np.allclose(
                        base_x_axis, x_axis
                    ):
                        raise ValueError("N_D ごとに横軸が一致しないため重ね描きできません。")

                mode_set.add(str(labels.get("mode", "temperature")))
                result_by_nd[float(nd_value)] = {
                    "curves": curves,
                    "labels": labels,
                }

        if base_x_axis is None or base_labels is None:
            raise RuntimeError("計算結果を取得できませんでした。")

        x_label = str(base_labels.get("x_label", "X"))
        if "xi_f" in mode_set:
            temp_k = float(base_labels.get("temperature_k", 300.0))
            st.warning(
                "T_range/xi_F データを使えなかったため、"
                f"横軸を xi_F (T={temp_k:.1f} K 固定) として表示しています。"
            )

        nd_values = [float(value) for value in selected_nd_values]
        for spec in PLOT_SPECS:
            key = str(spec["key"])
            series_by_nd: dict[float, np.ndarray] = {}

            for nd_value in nd_values:
                curves = result_by_nd[nd_value]["curves"]
                if key not in curves:
                    continue
                raw_values = np.asarray(curves[key], dtype=float)
                series_by_nd[nd_value] = raw_values * float(spec["scale"])

            if not series_by_nd:
                continue
            nd_values_used = list(series_by_nd.keys())

            st.subheader(str(spec["title"]))
            figure = build_plot(
                x_axis=base_x_axis,
                series_by_nd=series_by_nd,
                nd_values=nd_values_used,
                x_label=x_label,
                y_label=str(spec["y_label"]),
                plot_title=str(spec["title"]),
                composition=float(composition),
                clip_bottom_zero=bool(spec["clip_bottom_zero"]),
            )
            st.pyplot(figure)

            summary_parts: list[str] = []
            for nd_value in nd_values_used:
                y_values = series_by_nd[nd_value]
                finite_values = y_values[np.isfinite(y_values)]
                if finite_values.size > 0:
                    summary_parts.append(
                        f"N_D={nd_value:.2e}: max={float(np.nanmax(finite_values)):.4e}"
                    )
                else:
                    summary_parts.append(f"N_D={nd_value:.2e}: 有効値なし")
            st.caption(" / ".join(summary_parts))

            png_bytes = figure_to_png_bytes(figure)
            filename = build_download_filename(float(composition), nd_values_used, key)
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
