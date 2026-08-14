import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

from image_colorization import (
    process_images,
    plot_histogram,
    plot_difference_map,
    plot_color_channels,
    plot_lab_channels
)

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="SAR Color AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# GLOBAL CSS
# ---------------------------------------------------------

st.markdown("""
<style>

/* -------------------------------------------------------
   GLOBAL
------------------------------------------------------- */

.stApp {
    background:
        radial-gradient(circle at 12% 15%, rgba(123, 67, 255, 0.16), transparent 24%),
        radial-gradient(circle at 92% 16%, rgba(0, 210, 255, 0.14), transparent 28%),
        radial-gradient(circle at 75% 85%, rgba(255, 82, 150, 0.08), transparent 28%),
        linear-gradient(135deg, #060915 0%, #091527 46%, #071c2d 100%);
}

.block-container {
    max-width: 1480px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4, h5, h6 {
    color: #F8FBFF;
}

p {
    color: #B6C6D7;
}


/* -------------------------------------------------------
   SIDEBAR
------------------------------------------------------- */

section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at top left, rgba(255, 73, 205, 0.26), transparent 35%),
        radial-gradient(circle at bottom right, rgba(0, 210, 255, 0.20), transparent 38%),
        linear-gradient(165deg, #24104f 0%, #161a4a 48%, #07354c 100%);
    border-right: 1px solid rgba(255,255,255,0.12);
}

section[data-testid="stSidebar"]::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 5px;
    background: linear-gradient(
        90deg,
        #ff4ecd,
        #865dff,
        #16d4ff,
        #2de7a4
    );
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #ECF2FF !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12);
}

section[data-testid="stSidebar"] div[data-testid="stAlert"] {
    border-radius: 15px;
    background: linear-gradient(
        135deg,
        rgba(42, 230, 148, 0.18),
        rgba(0, 190, 255, 0.11)
    );
    border: 1px solid rgba(95,255,188,0.28);
    box-shadow: 0 8px 22px rgba(0,0,0,0.20);
}


/* -------------------------------------------------------
   HERO
------------------------------------------------------- */

.hero {
    position: relative;
    overflow: hidden;
    padding: 36px 38px;
    border-radius: 26px;

    background:
        radial-gradient(circle at 90% 10%, rgba(0, 220, 255, 0.18), transparent 30%),
        radial-gradient(circle at 15% 90%, rgba(154, 84, 255, 0.15), transparent 36%),
        linear-gradient(
            135deg,
            rgba(15, 29, 55, 0.96),
            rgba(7, 18, 34, 0.97)
        );

    border: 1px solid rgba(112, 210, 255, 0.17);
    box-shadow: 0 18px 50px rgba(0,0,0,0.30);

    margin-bottom: 28px;
}

.hero::after {
    content: "";
    position: absolute;
    width: 380px;
    height: 380px;
    border-radius: 50%;
    background: rgba(0, 195, 255, 0.05);
    right: -180px;
    top: -170px;
}

.hero-label {
    color: #6FE6FF;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 2.7px;
}

.hero-title {
    font-size: 3.25rem;
    font-weight: 850;
    margin-top: 8px;

    background: linear-gradient(
        90deg,
        #FFFFFF 10%,
        #91EDFF 48%,
        #CF9CFF 80%
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    max-width: 900px;
    color: #AFC1D3;
    margin-top: 13px;
    font-size: 1.04rem;
    line-height: 1.7;
}

.status-row {
    margin-top: 23px;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.status-chip {
    padding: 8px 13px;
    border-radius: 999px;
    color: #86F2C1;
    font-size: 0.80rem;
    font-weight: 700;

    background: rgba(43, 214, 145, 0.12);
    border: 1px solid rgba(67, 230, 167, 0.26);
}

.info-chip {
    padding: 8px 13px;
    border-radius: 999px;
    color: #B6DDF7;
    font-size: 0.80rem;
    font-weight: 700;

    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.10);
}


/* -------------------------------------------------------
   SECTION TITLES
------------------------------------------------------- */

.section-title {
    font-size: 1.65rem;
    font-weight: 800;
    color: #F7FAFF;
    margin-top: 28px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #859CB1;
    font-size: 0.94rem;
    margin-bottom: 18px;
}


/* -------------------------------------------------------
   UPLOAD CARDS
------------------------------------------------------- */

.sar-card {
    padding: 18px 20px;
    border-radius: 18px;
    background:
        linear-gradient(
            135deg,
            rgba(0, 198, 255, 0.17),
            rgba(0, 90, 180, 0.07)
        );
    border: 1px solid rgba(64, 210, 255, 0.27);
    margin-bottom: 12px;
    box-shadow: 0 10px 30px rgba(0, 130, 220, 0.08);
}

.optical-card {
    padding: 18px 20px;
    border-radius: 18px;
    background:
        linear-gradient(
            135deg,
            rgba(255, 137, 71, 0.18),
            rgba(255, 70, 149, 0.08)
        );
    border: 1px solid rgba(255, 155, 91, 0.28);
    margin-bottom: 12px;
    box-shadow: 0 10px 30px rgba(255, 100, 90, 0.07);
}

.card-label-cyan {
    color: #6BE7FF;
    font-size: 0.72rem;
    font-weight: 850;
    letter-spacing: 1.8px;
}

.card-label-orange {
    color: #FFBD82;
    font-size: 0.72rem;
    font-weight: 850;
    letter-spacing: 1.8px;
}

.card-title {
    color: #FFFFFF;
    font-size: 1.30rem;
    font-weight: 800;
    margin-top: 5px;
}

.card-text {
    color: #AABCD0;
    font-size: 0.87rem;
    margin-top: 4px;
}


/* -------------------------------------------------------
   FILE UPLOADER
------------------------------------------------------- */

div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.035);
    border: 1px dashed rgba(255,255,255,0.16);
    border-radius: 15px;
    padding: 10px;
}


/* -------------------------------------------------------
   MAIN BUTTON
------------------------------------------------------- */

.stButton > button {
    width: 100%;
    min-height: 52px;
    border-radius: 15px;

    background:
        linear-gradient(
            90deg,
            #6E4BFF,
            #147BFF,
            #00BFD8
        );

    border: 1px solid rgba(255,255,255,0.16);

    color: white;
    font-size: 1rem;
    font-weight: 800;

    box-shadow:
        0 12px 30px rgba(52, 100, 255, 0.25);
}

.stButton > button:hover {
    transform: translateY(-1px);
    border-color: rgba(130,230,255,0.60);
    color: #FFFFFF;
}


/* -------------------------------------------------------
   METRICS
------------------------------------------------------- */

.metric-card {
    min-height: 155px;
    padding: 24px;
    border-radius: 20px;
    position: relative;
    overflow: hidden;
}

.metric-purple {
    background:
        linear-gradient(
            135deg,
            rgba(148, 83, 255, 0.25),
            rgba(78, 47, 170, 0.13)
        );
    border: 1px solid rgba(189, 143, 255, 0.28);
}

.metric-blue {
    background:
        linear-gradient(
            135deg,
            rgba(0, 188, 255, 0.22),
            rgba(0, 85, 190, 0.12)
        );
    border: 1px solid rgba(75, 211, 255, 0.26);
}

.metric-green {
    background:
        linear-gradient(
            135deg,
            rgba(46, 224, 145, 0.21),
            rgba(0, 134, 121, 0.12)
        );
    border: 1px solid rgba(79, 238, 171, 0.27);
}

.metric-label {
    font-size: 0.73rem;
    font-weight: 850;
    letter-spacing: 1.4px;
}

.metric-value {
    color: #FFFFFF;
    font-size: 2rem;
    font-weight: 850;
    margin-top: 7px;
}

.metric-caption {
    color: #99ABBE;
    font-size: 0.82rem;
    margin-top: 5px;
}


/* -------------------------------------------------------
   TABS
------------------------------------------------------- */

div[data-testid="stTabs"] button {
    font-weight: 700;
    border-radius: 10px;
}

div[data-testid="stTabs"] {
    background: rgba(255,255,255,0.02);
    border-radius: 18px;
}


/* -------------------------------------------------------
   DOWNLOAD BUTTON
------------------------------------------------------- */

.stDownloadButton > button {
    width: 100%;
    border-radius: 13px;
    min-height: 45px;
}


/* -------------------------------------------------------
   FOOTER
------------------------------------------------------- */

.footer {
    margin-top: 50px;
    padding: 22px;
    text-align: center;

    border-top: 1px solid rgba(255,255,255,0.08);

    color: #6E8294;
    font-size: 0.84rem;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

defaults = {
    "processed": False,
    "src": None,
    "gen": None,
    "tar": None,
    "ssim_value": None,
    "psnr_value": None,
    "result_fig": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.markdown("""
<div style="
    padding:18px;
    border-radius:20px;
    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.13),
            rgba(255,255,255,0.045)
        );
    border:1px solid rgba(255,255,255,0.15);
    box-shadow:0 12px 30px rgba(0,0,0,0.20);
    margin-bottom:18px;
">

<div style="
    font-size:25px;
    font-weight:850;
    background:linear-gradient(90deg,#ffffff,#8feeff,#d49dff);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
">
🛰️ SAR Color AI
</div>

<div style="
    margin-top:8px;
    font-size:13px;
    line-height:1.55;
    color:#D7DFF4;
">
Remote sensing intelligence powered by Pix2Pix GAN.
</div>

</div>
""", unsafe_allow_html=True)

    st.markdown("### 🧠 Model Engine")

    st.success("● Generator Online")

    st.markdown("""
**Architecture:** Pix2Pix GAN  
**Input:** SAR / Grayscale  
**Output:** Optical-style RGB  
**Resolution:** 256 × 256
""")

    st.markdown("---")

    st.markdown("### 🚀 Workflow")

    st.markdown("""
**01** Upload SAR source image  
**02** Upload optical reference  
**03** Generate AI reconstruction  
**04** Inspect SSIM and PSNR  
**05** Explore visual analytics
""")

    st.markdown("---")

    st.markdown("### 🛰 Data Compatibility")

    st.caption(
        "Best results are obtained when source and target images represent the same geographic region."
    )


# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------

st.markdown("""
<div class="hero">

<div class="hero-label">
REMOTE SENSING INTELLIGENCE
</div>

<div class="hero-title">
🛰️ SAR Color AI
</div>

<div class="hero-subtitle">
AI-powered satellite image colorization for transforming Synthetic Aperture Radar imagery into optical-style RGB reconstructions. Evaluate generated outputs against ground-truth imagery using structural, perceptual and channel-level analysis.
</div>

<div class="status-row">
<span class="status-chip">● Model Online</span>
<span class="info-chip">Pix2Pix GAN</span>
<span class="info-chip">Sentinel-1 → Optical</span>
<span class="info-chip">256 × 256</span>
<span class="info-chip">SSIM + PSNR</span>
</div>

</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# INPUT AREA
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Satellite Image Input</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Upload a paired SAR and optical image from the same geographic location.</div>',
    unsafe_allow_html=True
)

left, right = st.columns(2, gap="large")


with left:

    st.markdown("""
<div class="sar-card">
<div class="card-label-cyan">SENTINEL-1 / SAR INPUT</div>
<div class="card-title">📡 Source SAR Image</div>
<div class="card-text">Upload the grayscale radar image that will be passed to the generator.</div>
</div>
""", unsafe_allow_html=True)

    src_image = st.file_uploader(
        "Source SAR image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        key="src_uploader"
    )


with right:

    st.markdown("""
<div class="optical-card">
<div class="card-label-orange">SENTINEL-2 / GROUND TRUTH</div>
<div class="card-title">🌍 Target Optical Image</div>
<div class="card-text">Upload the matching RGB optical image used for visual comparison and metrics.</div>
</div>
""", unsafe_allow_html=True)

    tar_image = st.file_uploader(
        "Target Optical image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        key="target_uploader"
    )


# ---------------------------------------------------------
# PROCESS BUTTON
# ---------------------------------------------------------

if src_image and tar_image:

    st.markdown("")

    run_process = st.button(
        "✨ Generate AI Color Reconstruction",
        use_container_width=True
    )

    if run_process or not st.session_state.processed:

        with st.spinner(
            "Running Pix2Pix generator and satellite image analysis..."
        ):

            fig, ssim_value, psnr_value, src, gen, tar = process_images(
                src_image,
                tar_image
            )

            st.session_state.processed = True
            st.session_state.src = src
            st.session_state.gen = gen
            st.session_state.tar = tar
            st.session_state.ssim_value = ssim_value
            st.session_state.psnr_value = psnr_value
            st.session_state.result_fig = fig


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

if st.session_state.processed:

    src = st.session_state.src
    gen = st.session_state.gen
    tar = st.session_state.tar

    ssim_value = st.session_state.ssim_value
    psnr_value = st.session_state.psnr_value

    st.markdown(
        '<div class="section-title">AI Reconstruction Result</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Source SAR image, AI-generated optical reconstruction and ground-truth target.</div>',
        unsafe_allow_html=True
    )

    if st.session_state.result_fig is not None:
        st.pyplot(
            st.session_state.result_fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # PERFORMANCE METRICS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">Model Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Key image reconstruction quality indicators.</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3, gap="large")


    with m1:

        st.markdown(f"""
<div class="metric-card metric-purple">

<div class="metric-label" style="color:#C8A7FF;">
STRUCTURAL SIMILARITY
</div>

<div class="metric-value">
{ssim_value:.4f}
</div>

<div class="metric-caption">
SSIM Score · Higher is better
</div>

</div>
""", unsafe_allow_html=True)


    with m2:

        st.markdown(f"""
<div class="metric-card metric-blue">

<div class="metric-label" style="color:#7DE4FF;">
RECONSTRUCTION QUALITY
</div>

<div class="metric-value">
{psnr_value:.2f} dB
</div>

<div class="metric-caption">
Peak Signal-to-Noise Ratio
</div>

</div>
""", unsafe_allow_html=True)


    with m3:

        st.markdown("""
<div class="metric-card metric-green">

<div class="metric-label" style="color:#7CE9B8;">
ACTIVE GENERATOR
</div>

<div class="metric-value" style="font-size:1.65rem;">
Pix2Pix GAN
</div>

<div class="metric-caption">
256 × 256 image reconstruction
</div>

</div>
""", unsafe_allow_html=True)


    with st.expander("📘 Understanding the metrics"):

        st.markdown("""
**SSIM** evaluates structural similarity between the generated image and the target image. Values closer to **1** indicate stronger structural similarity.

**PSNR** measures reconstruction quality based on pixel-level error. Higher PSNR values generally indicate lower distortion.
""")


    # -----------------------------------------------------
    # ANALYTICS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">Satellite Image Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Explore reconstruction behavior across color, difference and perceptual spaces.</div>',
        unsafe_allow_html=True
    )

    tab_hist, tab_diff, tab_rgb, tab_lab, tab_compare, tab_download = st.tabs(
        [
            "📊 Histograms",
            "🔥 Difference Map",
            "🌈 RGB Channels",
            "🧪 LAB Space",
            "🌓 Compare",
            "⬇ Export"
        ]
    )


    # -----------------------------------------------------
    # HISTOGRAM TAB
    # -----------------------------------------------------

    with tab_hist:

        st.markdown("### RGB Intensity Distribution")

        st.caption(
            "Compare pixel intensity distributions across source, generated and target imagery."
        )

        hist_fig, (ax1, ax2, ax3) = plt.subplots(
            1,
            3,
            figsize=(15, 5)
        )

        plot_histogram(
            (src + 1) / 2,
            ax1,
            "Source"
        )

        plot_histogram(
            (gen + 1) / 2,
            ax2,
            "Generated"
        )

        plot_histogram(
            (tar + 1) / 2,
            ax3,
            "Target"
        )

        st.pyplot(
            hist_fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # DIFFERENCE MAP TAB
    # -----------------------------------------------------

    with tab_diff:

        st.markdown("### Reconstruction Difference Map")

        diff_fig, mean_diff = plot_difference_map(
            (tar + 1) / 2,
            (gen + 1) / 2
        )

        d1, d2 = st.columns(
            [2, 1],
            gap="large"
        )

        with d1:

            st.pyplot(
                diff_fig,
                use_container_width=True
            )

        with d2:

            st.markdown(f"""
<div style="
    padding:22px;
    border-radius:18px;
    background:linear-gradient(
        135deg,
        rgba(255,95,75,0.20),
        rgba(255,164,55,0.10)
    );
    border:1px solid rgba(255,150,90,0.24);
">

<div style="
    color:#FFB280;
    font-size:12px;
    font-weight:800;
    letter-spacing:1.3px;
">
MEAN DIFFERENCE
</div>

<div style="
    color:white;
    font-size:31px;
    font-weight:850;
    margin-top:7px;
">
{mean_diff:.4f}
</div>

<div style="
    color:#AEBECC;
    font-size:13px;
    margin-top:6px;
">
Lower values indicate closer reconstruction.
</div>

</div>
""", unsafe_allow_html=True)

            st.info(
                "Bright regions in the map represent larger differences between the generated reconstruction and the ground-truth optical image."
            )


    # -----------------------------------------------------
    # RGB TAB
    # -----------------------------------------------------

    with tab_rgb:

        st.markdown("### RGB Channel Intelligence")

        st.caption(
            "Inspect individual red, green and blue channel responses."
        )

        c1, c2 = st.columns(
            2,
            gap="large"
        )

        with c1:

            st.markdown("#### 🎨 AI Generated")

            st.pyplot(
                plot_color_channels(
                    (gen + 1) / 2,
                    "Generated Image Color Channels"
                ),
                use_container_width=True
            )

        with c2:

            st.markdown("#### 🌍 Ground Truth")

            st.pyplot(
                plot_color_channels(
                    (tar + 1) / 2,
                    "Target Image Color Channels"
                ),
                use_container_width=True
            )


    # -----------------------------------------------------
    # LAB TAB
    # -----------------------------------------------------

    with tab_lab:

        st.markdown("### LAB Perceptual Color Space")

        st.markdown("""
**L** → Lightness  
**a** → Green ↔ Red  
**b** → Blue ↔ Yellow
""")

        l1, l2 = st.columns(
            2,
            gap="large"
        )

        with l1:

            st.markdown("#### AI Reconstruction")

            st.pyplot(
                plot_lab_channels(
                    (gen + 1) / 2,
                    "Generated Image in LAB Color Space"
                ),
                use_container_width=True
            )

        with l2:

            st.markdown("#### Ground Truth")

            st.pyplot(
                plot_lab_channels(
                    (tar + 1) / 2,
                    "Target Image in LAB Color Space"
                ),
                use_container_width=True
            )


    # -----------------------------------------------------
    # COMPARE TAB
    # -----------------------------------------------------

    with tab_compare:

        st.markdown("### Interactive SAR ↔ AI Comparison")

        st.caption(
            "Blend between the original SAR input and the generated optical-style reconstruction."
        )

        comparison_value = st.slider(
            "AI reconstruction visibility",
            0.0,
            1.0,
            0.5,
            0.01
        )

        comparison_fig, ax = plt.subplots(
            figsize=(11, 6)
        )

        blended = (
            ((src + 1) / 2) * (1 - comparison_value)
            +
            ((gen + 1) / 2) * comparison_value
        )

        ax.imshow(blended)
        ax.axis("off")

        st.pyplot(
            comparison_fig,
            use_container_width=True
        )

        st.caption(
            f"AI-generated image visibility: {comparison_value:.0%}"
        )


    # -----------------------------------------------------
    # EXPORT TAB
    # -----------------------------------------------------

    with tab_download:

        st.markdown("### Export Satellite Results")

        st.caption(
            "Download the source, generated reconstruction and target optical image."
        )

        dl1, dl2, dl3 = st.columns(
            3,
            gap="large"
        )

        with dl1:

            src_image.seek(0)

            st.download_button(
                "📡 Download SAR Source",
                data=src_image.getvalue(),
                file_name="source_sar_image.png",
                mime="image/png",
                use_container_width=True
            )

        with dl2:

            gen_img = np.clip(
                ((gen + 1) / 2) * 255,
                0,
                255
            ).astype(np.uint8)

            generated_pil = Image.fromarray(gen_img)

            generated_buffer = BytesIO()

            generated_pil.save(
                generated_buffer,
                format="PNG"
            )

            generated_buffer.seek(0)

            st.download_button(
                "✨ Download AI Reconstruction",
                data=generated_buffer.getvalue(),
                file_name="generated_optical_image.png",
                mime="image/png",
                use_container_width=True
            )

        with dl3:

            tar_image.seek(0)

            st.download_button(
                "🌍 Download Optical Target",
                data=tar_image.getvalue(),
                file_name="target_optical_image.png",
                mime="image/png",
                use_container_width=True
            )


# ---------------------------------------------------------
# EMPTY STATE
# ---------------------------------------------------------

else:

    st.markdown("""
<div style="
    padding:22px;
    margin-top:20px;
    border-radius:18px;
    background:
        linear-gradient(
            135deg,
            rgba(70,100,255,0.08),
            rgba(0,200,255,0.05)
        );
    border:1px solid rgba(105,190,255,0.12);
    text-align:center;
">

<div style="
    font-size:30px;
">
🛰️
</div>

<div style="
    color:white;
    font-size:18px;
    font-weight:750;
    margin-top:5px;
">
Ready for Satellite Analysis
</div>

<div style="
    color:#8EA5B9;
    font-size:14px;
    margin-top:6px;
">
Upload both a SAR source image and the matching optical target to start reconstruction.
</div>

</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("""
<div class="footer">
🛰️ SAR Color AI
&nbsp; • &nbsp;
Pix2Pix GAN Remote Sensing Intelligence
&nbsp; • &nbsp;
Team ERROR 404
</div>
""", unsafe_allow_html=True)