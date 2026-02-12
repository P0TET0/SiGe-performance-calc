from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import japanize_matplotlib  # noqa: F401

import simulator

st.set_page_config(page_title="SiGe 熱電性能シミュレーター", layout="centered")


@st.cache_data(show_spinner=False)
def run_simulation_cached(
    composition: float,
    nd_value: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    return simulator.simulate_zt(composition, nd_value)


def build_plot(
    x_axis: np.ndarray,
    zt_values: np.ndarray,
    labels: dict[str, object],
    composition: float,
    nd_value: float,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9, 5.2))
    axis.plot(x_axis, zt_values, color="#005f73", linewidth=2.4)
    axis.set_xlabel(str(labels.get("x_label", "X")))
    axis.set_ylabel(str(labels.get("y_label", "ZT")))
    axis.set_title(
        f"組成比(Si$_{{1-y}}$Ge$_{{y}}$) y={composition:.2f}, N_D={nd_value:.2e}"
    )
    axis.grid(True, linestyle=":", alpha=0.6)
    axis.set_ylim(bottom=0)

    finite_mask = np.isfinite(zt_values)
    if finite_mask.any():
        max_index = int(np.nanargmax(zt_values))
        axis.scatter(
            [x_axis[max_index]],
            [zt_values[max_index]],
            color="#ca6702",
            s=45,
            zorder=3,
        )

    figure.tight_layout()
    return figure


def figure_to_png_bytes(figure: plt.Figure) -> bytes:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    buffer.seek(0)
    return buffer.read()


def build_download_filename(composition: float, nd_value: float) -> str:
    comp = f"{composition:.2f}".replace(".", "p")
    nd = f"{nd_value:.2e}".replace("+", "")
    return f"zt_y{comp}_nd{nd}.png"


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
        with st.spinner("ZTを計算中です..."):
            x_axis, zt_values, labels = run_simulation_cached(
                float(composition), float(nd_value)
            )

        mode = str(labels.get("mode", "temperature"))
        if mode == "xi_f":
            temp_k = float(labels.get("temperature_k", 300.0))
            st.warning(
                f"T_range/xi_F データを使えなかったため、ZT vs xi_F "
                f"(T={temp_k:.1f} K) を表示しています。"
            )

        plot = build_plot(x_axis, zt_values, labels, float(composition), float(nd_value))
        st.pyplot(plot)

        finite_values = zt_values[np.isfinite(zt_values)]
        if finite_values.size > 0:
            st.caption(f"最大ZT: {float(np.nanmax(finite_values)):.4f}")
        else:
            st.caption("ZTの有効値を計算できませんでした。")

        png_bytes = figure_to_png_bytes(plot)
        filename = build_download_filename(float(composition), float(nd_value))
        st.download_button(
            label="PNGをダウンロード",
            data=png_bytes,
            file_name=filename,
            mime="image/png",
        )
        plt.close(plot)

    except Exception as error:
        st.error(f"計算中にエラーが発生しました: {error}")
        st.code(str(error))

try:
    nd_path = simulator.get_nd_pickle_path()
    st.caption(f"N_D候補読み込み元: {Path(nd_path)}")
except Exception:
    pass
