"""
DeepFake Images Detector – Interactive GUI
==========================================
Jupyter / Kaggle notebook widget app.
Requires: tensorflow, ipywidgets, pillow, opencv-python, pandas, matplotlib

Usage
-----
Run this notebook cell AFTER the model has been trained and saved via src/model.py.
The app will look for the saved model in /kaggle/working/.
"""

import os
import io
import base64
import warnings
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import ipywidgets as widgets
from IPython.display import display, HTML
from PIL import Image as PILImage

tf.get_logger().setLevel("ERROR")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

IMG_SIZE = (128, 128)

MODEL_SEARCH_PATHS = [
    "/kaggle/working/final_deepfake_model.h5",
    "final_deepfake_model.h5",
    "/kaggle/working/best_model.h5",
    "best_model.h5",
]

# Dark theme palette
THEME = {
    "bg":     "#0b1220",
    "card":   "#111827",
    "border": "#243041",
    "text":   "#e5e7eb",
    "muted":  "#94a3b8",
    "accent": "#8b5cf6",
    "green":  "#10b981",
    "red":    "#ef4444",
}

# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────

def load_model() -> tf.keras.Model:
    """Search known paths and return the first loadable model."""
    for path in MODEL_SEARCH_PATHS:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path)
                print(f"✅ Model loaded from: {path}")
                return model
            except Exception as exc:
                print(f"⚠️  Could not load {path}: {exc}")

    raise FileNotFoundError(
        "❌ No model file found. "
        "Run src/model.py first to train and save the model."
    )


# ──────────────────────────────────────────────
# Image Utilities
# ──────────────────────────────────────────────

def parse_upload(upload_value) -> list[tuple[str, bytes]]:
    """
    Normalise ipywidgets FileUpload value across Kaggle / Jupyter formats.

    Returns
    -------
    list of (filename, bytes) tuples
    """
    files = []
    if isinstance(upload_value, dict):
        for name, info in upload_value.items():
            files.append((name, bytes(info["content"])))
    elif isinstance(upload_value, tuple):
        for item in upload_value:
            if isinstance(item, dict):
                files.append((item.get("name", "image.jpg"), bytes(item["content"])))
    return files


def load_pil_image(raw_bytes: bytes) -> PILImage.Image:
    """
    Decode bytes to a PIL RGB image.

    Raises ValueError with a user-friendly message on failure.
    """
    if isinstance(raw_bytes, memoryview):
        raw_bytes = raw_bytes.tobytes()

    try:
        img = PILImage.open(io.BytesIO(raw_bytes))
        return img.convert("RGB")
    except Exception as exc:
        raise ValueError(
            "Cannot read this file as an image.\n"
            "Supported formats: JPG, JPEG, PNG.\n"
            "HEIC / WEBP files must be converted first."
        ) from exc


def preprocess(img_array: np.ndarray) -> np.ndarray:
    """
    Resize, strip alpha channel if present, normalise to [0,1],
    and add batch dimension. Returns shape (1, H, W, 3).
    """
    # Drop alpha channel
    if img_array.ndim == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    # Grayscale → RGB
    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    img = cv2.resize(img_array, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────

def predict(model: tf.keras.Model, img_array: np.ndarray) -> dict:
    """
    Run inference on a single image array.

    Model convention
    ----------------
    raw score < 0.5  →  REAL
    raw score >= 0.5 →  FAKE

    Returns
    -------
    dict with keys: raw_pred, label, is_real, emoji,
                    confidence, real_prob, fake_prob
    """
    x       = preprocess(img_array)
    raw     = float(model.predict(x, verbose=0)[0][0])
    is_real = raw < 0.5

    return {
        "raw_pred":   raw,
        "label":      "REAL" if is_real else "FAKE",
        "is_real":    is_real,
        "emoji":      "👤" if is_real else "🤖",
        "confidence": max(raw, 1 - raw) * 100,
        "real_prob":  (1 - raw) * 100,
        "fake_prob":  raw * 100,
    }


# ──────────────────────────────────────────────
# History
# ──────────────────────────────────────────────

prediction_history: list[dict] = []


def record(filename: str, result: dict, analysis_type: str = "Basic") -> None:
    prediction_history.append({
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename":         filename,
        "prediction":       result["label"],
        "confidence":       f"{result['confidence']:.1f}%",
        "confidence_value": result["confidence"],
        "raw_score":        result["raw_pred"],
        "analysis_type":    analysis_type,
        "is_real":          result["is_real"],
    })


# ──────────────────────────────────────────────
# Charts
# ──────────────────────────────────────────────

def _dark_fig(figsize=(6, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(THEME["bg"])
    ax.set_facecolor(THEME["bg"])
    return fig, ax


def plot_confidence_donut(result: dict) -> None:
    conf   = result["confidence"]
    color  = THEME["green"] if result["is_real"] else THEME["red"]
    fig, ax = _dark_fig()

    ax.pie(
        [conf, 100 - conf],
        colors=[color, "#1f2937"],
        startangle=90,
        wedgeprops={"width": 0.35, "edgecolor": THEME["bg"]},
        autopct="%1.1f%%",
        pctdistance=0.85,
        textprops={"color": "white", "fontsize": 10},
    )
    ax.text(0, 0, f"{conf:.1f}%\nConfidence",
            ha="center", va="center",
            fontsize=14, fontweight="bold", color="white")
    ax.set_title("Prediction Confidence", fontsize=13,
                 fontweight="bold", color="white")
    ax.axis("equal")
    plt.show()


def plot_probability_bars(result: dict) -> None:
    fig, ax = _dark_fig()
    bars = ax.bar(
        ["REAL", "FAKE"],
        [result["real_prob"], result["fake_prob"]],
        color=[THEME["green"], THEME["red"]],
        edgecolor="white", width=0.6,
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)", fontweight="bold", color="white")
    ax.set_title("Probability Distribution", fontweight="bold", color="white")
    ax.grid(True, alpha=0.25, axis="y")
    ax.tick_params(colors="white")
    for b, p in zip(bars, [result["real_prob"], result["fake_prob"]]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{p:.1f}%", ha="center", va="bottom",
                fontweight="bold", color="white")
    plt.show()


def plot_distribution_pie(real: int, fake: int, title: str = "Real vs Fake") -> None:
    fig, ax = _dark_fig()
    ax.pie([real, fake], labels=["REAL", "FAKE"], autopct="%1.1f%%",
           startangle=90,
           wedgeprops={"width": 0.4, "edgecolor": THEME["bg"]},
           textprops={"color": "white"})
    ax.set_title(title, fontweight="bold", color="white")
    ax.axis("equal")
    plt.show()


# ──────────────────────────────────────────────
# Widget Helpers
# ──────────────────────────────────────────────

def _html_card(title: str, subtitle: str = "") -> widgets.HTML:
    return widgets.HTML(f"""
    <div style="background:{THEME['card']}; color:{THEME['text']};
                padding:18px; border-radius:14px; margin-bottom:15px;
                border:1px solid {THEME['border']};">
      <h2 style="margin:0;">{title}</h2>
      <p style="margin:6px 0 0; color:{THEME['muted']};">{subtitle}</p>
    </div>""")


def _output_box(height: str = "auto") -> widgets.Output:
    return widgets.Output(layout=widgets.Layout(
        border=f"1px solid {THEME['border']}",
        border_radius="14px",
        padding="15px",
        margin="15px 0",
        background_color=THEME["card"],
        max_height=height,
        overflow_y="auto" if height != "auto" else "visible",
    ))


# ──────────────────────────────────────────────
# Tab 1 – Detect
# ──────────────────────────────────────────────

def build_detect_tab(model):
    # Single image widgets
    single_upload  = widgets.FileUpload(accept="image/*", multiple=False,
                                        description="📁 Upload Image",
                                        button_style="primary",
                                        layout=widgets.Layout(width="260px"))
    single_preview = _output_box()
    single_out     = _output_box()
    btn_basic      = widgets.Button(description="🔍 Basic Detect",      button_style="success",
                                    layout=widgets.Layout(width="200px"), disabled=True)
    btn_diag       = widgets.Button(description="🔬 Diagnostic Detect", button_style="info",
                                    layout=widgets.Layout(width="200px"), disabled=True)

    # Batch image widgets
    batch_upload  = widgets.FileUpload(accept="image/*", multiple=True,
                                       description="📁📁 Upload Batch",
                                       button_style="warning",
                                       layout=widgets.Layout(width="260px"))
    batch_info    = widgets.HTML(f"<p style='text-align:center;color:{THEME['muted']};'>No batch selected</p>")
    batch_out     = _output_box(height="450px")
    btn_b_basic   = widgets.Button(description="🔍 Batch Basic",      button_style="success",
                                   layout=widgets.Layout(width="200px"), disabled=True)
    btn_b_diag    = widgets.Button(description="🔬 Batch Diagnostic", button_style="info",
                                   layout=widgets.Layout(width="200px"), disabled=True)
    progress      = widgets.IntProgress(value=0, min=0, max=100,
                                        description="Progress:",
                                        bar_style="info",
                                        layout=widgets.Layout(width="85%", display="none"))

    # ── Events ──────────────────────────────

    def on_single_upload(change):
        if not single_upload.value:
            return
        files = parse_upload(single_upload.value)
        if not files:
            return
        filename, data = files[0]
        single_preview.clear_output()
        with single_preview:
            try:
                img = load_pil_image(data)
                img.thumbnail((300, 300))
                display(img)
                print(f"📁 {filename}")
            except ValueError as exc:
                print(exc)
                return
        btn_basic.disabled = False
        btn_diag.disabled  = False

    single_upload.observe(on_single_upload, names="value")

    def _run_single(analysis_type: str):
        files = parse_upload(single_upload.value)
        filename, data = files[0]
        img_array = np.array(load_pil_image(data))
        res = predict(model, img_array)
        record(filename, res, analysis_type)

        single_out.clear_output()
        with single_out:
            tag = "🔍 BASIC" if analysis_type == "Basic" else "🔬 DIAGNOSTIC"
            print(f"{tag} DETECTION RESULT")
            print("─" * 45)
            print(f"File       : {filename}")
            print(f"Prediction : {res['emoji']} {res['label']}")
            print(f"Confidence : {res['confidence']:.1f}%")
            print(f"Raw score  : {res['raw_pred']:.6f}")
            if analysis_type == "Diagnostic":
                print(f"\nREAL prob  : {res['real_prob']:.1f}%")
                print(f"FAKE prob  : {res['fake_prob']:.1f}%")
                plot_probability_bars(res)
            plot_confidence_donut(res)

    def on_basic(btn):
        btn_basic.disabled, btn_basic.description = True, "⏳ Detecting..."
        try:
            _run_single("Basic")
        except Exception as exc:
            with single_out:
                print(f"❌ {exc}")
        btn_basic.disabled, btn_basic.description = False, "🔍 Basic Detect"

    def on_diag(btn):
        btn_diag.disabled, btn_diag.description = True, "⏳ Detecting..."
        try:
            _run_single("Diagnostic")
        except Exception as exc:
            with single_out:
                print(f"❌ {exc}")
        btn_diag.disabled, btn_diag.description = False, "🔬 Diagnostic Detect"

    btn_basic.on_click(on_basic)
    btn_diag.on_click(on_diag)

    def on_batch_upload(change):
        n = len(parse_upload(batch_upload.value)) if batch_upload.value else 0
        if n:
            batch_info.value = (f"<p style='text-align:center;color:{THEME['green']};"
                                f"font-weight:700;'>✅ {n} images selected</p>")
            btn_b_basic.disabled = False
            btn_b_diag.disabled  = False
        else:
            batch_info.value = (f"<p style='text-align:center;color:{THEME['muted']};'>"
                                f"No batch selected</p>")

    batch_upload.observe(on_batch_upload, names="value")

    def _run_batch(analysis_type: str):
        files = parse_upload(batch_upload.value)
        progress.max = len(files)
        progress.layout.display = "flex"
        results = []

        batch_out.clear_output()
        with batch_out:
            tag = "🔍 BATCH BASIC" if analysis_type == "Batch Basic" else "🔬 BATCH DIAGNOSTIC"
            print(f"{tag} RESULTS")
            print("─" * 60)
            for i, (filename, data) in enumerate(files, 1):
                progress.value = i
                res = predict(model, np.array(load_pil_image(data)))
                record(filename, res, analysis_type)
                results.append(res)
                print(f"{i:02d}. {filename[:35]:35s} → {res['emoji']} {res['label']} "
                      f"| {res['confidence']:.1f}%")

            real_n = sum(r["is_real"] for r in results)
            fake_n = len(results) - real_n
            print(f"\n📌 Summary — Total: {len(results)} | Real: {real_n} | Fake: {fake_n}")
            plot_distribution_pie(real_n, fake_n, "Batch Real vs Fake")

        progress.layout.display = "none"
        progress.value = 0

    def on_batch_basic(btn):
        btn_b_basic.disabled, btn_b_basic.description = True, "⏳ Processing..."
        try:
            _run_batch("Batch Basic")
        except Exception as exc:
            with batch_out:
                print(f"❌ {exc}")
        btn_b_basic.disabled, btn_b_basic.description = False, "🔍 Batch Basic"

    def on_batch_diag(btn):
        btn_b_diag.disabled, btn_b_diag.description = True, "⏳ Processing..."
        try:
            _run_batch("Batch Diagnostic")
        except Exception as exc:
            with batch_out:
                print(f"❌ {exc}")
        btn_b_diag.disabled, btn_b_diag.description = False, "🔬 Batch Diagnostic"

    btn_b_basic.on_click(on_batch_basic)
    btn_b_diag.on_click(on_batch_diag)

    # ── Layout ──────────────────────────────

    return widgets.VBox([
        _html_card("🔍 Detect Images",
                   "Upload a single image or a batch to classify as REAL or FAKE."),
        widgets.HTML(f"<h3 style='color:{THEME['text']};'>🖼️ Single Image</h3>"),
        single_upload, single_preview,
        widgets.HBox([btn_basic, btn_diag],
                     layout=widgets.Layout(justify_content="center", gap="10px")),
        single_out,
        widgets.HTML("<hr style='margin:20px 0;border-color:#243041;'>"),
        widgets.HTML(f"<h3 style='color:{THEME['text']};'>🖼️🖼️ Batch Images</h3>"),
        batch_upload, batch_info,
        widgets.HBox([btn_b_basic, btn_b_diag],
                     layout=widgets.Layout(justify_content="center", gap="10px")),
        progress, batch_out,
    ])


# ──────────────────────────────────────────────
# Tab 2 – Dashboard
# ──────────────────────────────────────────────

def build_dashboard_tab():
    dash_out   = _output_box()
    btn_refresh = widgets.Button(description="🔄 Refresh",
                                 button_style="primary",
                                 layout=widgets.Layout(width="200px"))

    def render(_=None):
        dash_out.clear_output()
        with dash_out:
            if not prediction_history:
                print("No data yet. Run detections first.")
                return

            df    = pd.DataFrame(prediction_history)
            total = len(df)
            real  = int(df["is_real"].sum())
            fake  = total - real
            avg   = df["confidence_value"].mean()

            print(f"📌 Total: {total}  |  Real: {real}  |  Fake: {fake}  "
                  f"|  Avg Confidence: {avg:.1f}%")

            plot_distribution_pie(real, fake, "Overall Real vs Fake")

            fig, ax = _dark_fig()
            ax.hist(df["confidence_value"], bins=10, edgecolor="white", alpha=0.7)
            ax.set_title("Confidence Distribution", fontweight="bold", color="white")
            ax.set_xlabel("Confidence (%)", color="white")
            ax.set_ylabel("Count",          color="white")
            ax.grid(True, alpha=0.25)
            ax.tick_params(colors="white")
            plt.show()

    btn_refresh.on_click(render)

    return widgets.VBox([
        _html_card("📊 Analytics Dashboard",
                   "Live statistics based on your detection history."),
        btn_refresh, dash_out,
    ]), render


# ──────────────────────────────────────────────
# Tab 3 – History
# ──────────────────────────────────────────────

def build_history_tab():
    hist_out   = _output_box(height="450px")
    btn_clear  = widgets.Button(description="🗑️ Clear",      button_style="danger",
                                layout=widgets.Layout(width="180px"))
    btn_export = widgets.Button(description="📊 Export CSV", button_style="success",
                                layout=widgets.Layout(width="180px"))

    def render():
        hist_out.clear_output()
        with hist_out:
            if not prediction_history:
                print("No history yet.")
                return
            display(pd.DataFrame(prediction_history).tail(50))

    def clear(_):
        global prediction_history
        prediction_history = []
        render()

    def export(_):
        if not prediction_history:
            with hist_out:
                print("❌ Nothing to export.")
            return
        df   = pd.DataFrame(prediction_history)
        name = f"deepfake_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(name, index=False)
        b64  = base64.b64encode(df.to_csv(index=False).encode()).decode()
        link = (f'<a style="color:#60a5fa;font-weight:700;" '
                f'download="{name}" href="data:text/csv;base64,{b64}">📥 Download {name}</a>')
        with hist_out:
            print(f"✅ Saved: {name}")
            display(HTML(link))

    btn_clear.on_click(clear)
    btn_export.on_click(export)

    tab = widgets.VBox([
        _html_card("📋 Prediction History",
                   "Record of all predictions with confidence scores."),
        hist_out,
        widgets.HBox([btn_clear, btn_export],
                     layout=widgets.Layout(justify_content="center", gap="10px")),
    ])
    return tab, render


# ──────────────────────────────────────────────
# App Entry Point
# ──────────────────────────────────────────────

def launch():
    """Load model and display the full app."""
    print("🚀 DeepFake Detection App (GUI)")
    print("=" * 50)

    model = load_model()

    detect_tab              = build_detect_tab(model)
    dashboard_tab, refresh  = build_dashboard_tab()
    history_tab,   render   = build_history_tab()

    tabs = widgets.Tab(children=[detect_tab, dashboard_tab, history_tab])
    tabs.set_title(0, "🔍 Detect")
    tabs.set_title(1, "📊 Dashboard")
    tabs.set_title(2, "📋 History")

    header = widgets.HTML(f"""
    <div style="text-align:center; padding:25px; margin-bottom:15px;
                background:{THEME['bg']}; border-radius:14px;
                border:1px solid {THEME['border']};">
      <h1 style="margin:0; font-size:36px; font-weight:800; color:{THEME['text']};">
        🔍 DeepFake Images Detector
      </h1>
      <p style="margin:8px 0 0; color:{THEME['muted']}; font-size:16px;">
        ML Semester Project | Real vs Fake Image Classification
      </p>
      <div style="width:220px; height:4px;
                  background:linear-gradient(90deg,#4f46e5,{THEME['accent']});
                  margin:18px auto; border-radius:2px;"></div>
    </div>""")

    refresh()
    render()
    display(widgets.VBox([header, tabs]))


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────

launch()
