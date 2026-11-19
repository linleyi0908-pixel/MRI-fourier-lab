"""Journey Inside an MRI: Fourier Reconstruction Lab.

Run: streamlit run app.py
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

from math_core import (
    grayscale_rgb,
    kspace_rgb,
    reconstruct,
    reconstruction_metrics,
    to_grayscale_array,
)

BASE_DIR = Path(__file__).resolve().parent
MRI_URL = "https://upload.wikimedia.org/wikipedia/commons/2/26/MRI_Brain_T1_Axial_%2811%29.jpg"
SOURCE_PAGE = "https://commons.wikimedia.org/wiki/File:MRI_Brain_T1_Axial_(11).jpg"

st.set_page_config(page_title="MRI Fourier Lab", page_icon="🧠", layout="wide")

st.markdown("""
<style>
  :root { color-scheme: dark; }
  .stApp { background: #071322; color: #e8f2ff; }
  [data-testid="stHeader"] { background: #071322; }
  h1, h2, h3 { color: #e8f2ff !important; }
  [data-testid="stMetric"] {
    background: #101f31; border: 1px solid #213d50;
    border-radius: 12px; padding: 12px 16px;
  }
  [data-testid="stImage"] img { border-radius: 12px; }
  .eyebrow { font-size: 0.74rem; letter-spacing: 0.24em; color: #77dff4; }
  .lede { font-size: 1.04rem; color: #a6bfd0; max-width: 750px; }
  .note { color: #a9bed0; font-size: .92rem; }
  .sectionline { border-top: 1px solid #284054; margin: 8px 0 18px 0; }
  a { color: #7fdcf4 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_default() -> tuple[bytes, bool]:
    """Use a licensed real brain MRI; offline fallback is clearly labeled."""
    local = BASE_DIR / "brain_mri.jpg"
    if local.is_file():
        return local.read_bytes(), True
    try:
        response = requests.get(MRI_URL, headers={"User-Agent": "MRI-Fourier-Education/1.0"}, timeout=9)
        response.raise_for_status()
        with Image.open(BytesIO(response.content)) as image:
            image.verify()
        return response.content, True
    except (requests.RequestException, OSError, ValueError):
        return (BASE_DIR / "synthetic_fallback.png").read_bytes(), False


def set_preset(high: int, low: int) -> None:
    st.session_state["remove_high"] = high
    st.session_state["remove_low"] = low


st.markdown('<div class="eyebrow">JOURNEY INSIDE AN MRI · EXPERIMENT 02</div>', unsafe_allow_html=True)
st.title("Can you reconstruct a brain?")
st.markdown('<p class="lede">An MRI image becomes a matrix. The Fourier transform decomposes it into spatial frequencies. Remove part of that information and discover what can—and cannot—be reconstructed.</p>', unsafe_allow_html=True)
st.markdown('<div class="sectionline"></div>', unsafe_allow_html=True)

image_bytes, is_real_mri = load_default()
try:
    image_file = Image.open(BytesIO(image_bytes))
    original = to_grayscale_array(image_file)
except (OSError, ValueError):
    st.error("The sample image could not be opened. Confirm the image file is valid.")
    st.stop()

if not is_real_mri:
    st.info("Offline sample mode: the licensed MRI image could not be fetched. A synthetic brain-shaped teaching phantom is shown instead. Add brain_mri.jpg to your repository to guarantee the real MRI is displayed.")

st.subheader("Change the frequency information")
button_cols = st.columns(4)
for column, label, h, l in zip(
    button_cols,
    ["All data", "Blur the image", "Keep edges", "Mix both"],
    [0, 65, 0, 30],
    [0, 0, 22, 11],
):
    with column:
        st.button(label, use_container_width=True, on_click=set_preset, args=(h, l))

control_a, control_b = st.columns(2, gap="large")
with control_a:
    high = st.slider(
        "Remove outer high-frequency band (%)", 0, 90,
        key="remove_high", help="Increasing this blurs sharp details and boundaries. Percentage refers to radius, not the fraction of coefficients."
    )
with control_b:
    low = st.slider(
        "Remove central low-frequency disk (%)", 0, 50,
        key="remove_low", help="Increasing this suppresses broad, smooth structure and the average brightness. Percentage refers to radius."
    )

original_k, mask, filtered_k, recovered, inverse_complex = reconstruct(original, high, low)
metrics = reconstruction_metrics(original, recovered, mask)

if not mask.any():
    st.warning("These settings remove every measured frequency. No image information remains.")

image_cols = st.columns(3, gap="medium")
with image_cols[0]:
    st.markdown("**01 / Original image**")
    st.image(grayscale_rgb(original), use_container_width=True)
    st.caption("An intensity matrix: one number per image pixel")
with image_cols[1]:
    st.markdown("**02 / Observed k-space**")
    st.image(kspace_rgb(filtered_k, original_k), use_container_width=True)
    st.caption("Logarithmic magnitude; center = low spatial frequencies")
with image_cols[2]:
    st.markdown("**03 / Reconstruction**")
    st.image(grayscale_rgb(recovered), use_container_width=True)
    st.caption("Magnitude of the inverse Fourier transform; fixed intensity scale")

metric_cols = st.columns(3)
metric_cols[0].metric("Measurements retained", f"{metrics['kept_pct']:.1f}%")
metric_cols[1].metric("Reconstruction RMSE", f"{metrics['rmse']:.6f}")
metric_cols[2].metric("Maximum pixel error", f"{metrics['max_error']:.6f}")

if metrics["all_data"]:
    st.success("Perfect reconstruction to floating-point precision: the complete Fourier transform is invertible.")
elif high > 0 and low == 0:
    st.info("Removing high frequencies suppresses fine details and sharp edges. The inverse transform produces a smoother image.")
elif low > 0 and high == 0:
    st.info("Removing low frequencies suppresses broad intensity patterns. Edges and rapid brightness changes become more prominent.")
else:
    st.info("You've removed multiple frequency bands. The missing coefficients leave information that this inverse transform cannot recover.")

with st.expander("The linear algebra behind the experiment", expanded=False):
    st.markdown(r"""
    Treat the image as a matrix of pixel intensities, $X\in\mathbb{R}^{N\times N}$. The
    two-dimensional discrete Fourier transform produces $K\in\mathbb{C}^{N\times N}$:

    $$K = F X F^{T}$$

    where $F$ is the $N\times N$ discrete Fourier matrix, with
    $F_{jk}=e^{-2\pi i jk/N}$. Because $F^*F=NI$,

    $$X = \frac{1}{N^2} F^* K\, \overline{F}.$$

    In NumPy, `fft2` computes the forward transform; `ifft2` computes the inverse.
    The `fftshift` operation only moves the zero-frequency coefficient to the center
    for display. It is undone with `ifftshift` before reconstruction.

    When you remove measurements, the display implements
    $K_{\mathrm{observed}}=M\odot K$ (where $M$ is a binary mask).
    Setting coefficients to zero does **not** recover their unknown original values.
    The measurement map is non-injective if a frequency is discarded: different
    original images can give the same observed measurements.

    For MRI, this is an **idealized retrospective experiment** on a finished image.
    A scanner actually measures complex-valued k-space signals produced by
    magnetic-resonance physics. Scanner sampling and reconstruction can be more complex.
    """)

st.markdown("---")
if is_real_mri:
    st.caption(
        "MRI sample: ‘MRI Brain T1 Axial (11).jpg’ by 511KeV, "
        "CC BY-SA 4.0, via Wikimedia Commons. Resized to 256×256, converted to grayscale "
        "and Fourier-filtered. Image and processed image outputs are shared under CC BY-SA 4.0."
    )
    st.markdown(
        f"[View original and attribution]({SOURCE_PAGE}) · "
        "[Creative Commons license](https://creativecommons.org/licenses/by-sa/4.0/)"
    )
else:
    st.caption("Offline fallback: synthetic mathematical image created for this project, not a clinical scan.")
st.caption("Educational demonstration only; no diagnostic use. Mathematical source: NumPy FFT documentation.")
