# app.py
import io
import math
import requests
from copy import deepcopy

import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import DetrFeatureExtractor, DetrForSegmentation

# Detectron2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color

# ---------- Configuración inicial y estilos -----------------------------
st.set_page_config(
    page_title="Inferencia Panóptica DETR-ResNet-101",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #1E3A8A;
    margin-bottom: 2rem;
    text-align: center;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #1E3A8A;
}
.section-subtitle {
    font-size: 1.3rem;
    font-weight: 600;
    color: #1E40AF;
    margin: 1.2rem 0 0.6rem 0;
}
.output-card {
    background-color: #D6E8FF;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin: 1rem 0 2rem 0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    border-left: 6px solid #1E40AF;
    color: #0D47A1;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Inferencia Panóptica con DETR-ResNet-101</div>', unsafe_allow_html=True)

# -- Bloque introductorio ----------------------------------------------------
st.markdown("""
<div class="output-card">
Esta aplicación realiza segmentación panóptica usando el modelo DETR‑ResNet‑101, que combina detección y segmentación para identificar objetos y regiones en la imagen.<br><br>
El proceso general del modelo es:
<ul>
  <li><em>Detección de objetos y regiones:</em> El modelo identifica qué partes de la imagen corresponden a distintos objetos o áreas.</li>
  <li><em>Segmentación panóptica:</em> Asigna una máscara a cada objeto o región, tanto "things" (objetos con forma definida) como "stuff" (fondos o áreas sin forma específica).</li>
  <li><em>Post‑procesamiento:</em> Se refinan las máscaras y se produce un mapa segmentado que representa toda la escena.</li>
</ul>
<p>Puedes subir una imagen propia o usar la de ejemplo para ver estos pasos reflejados en los resultados.</p>
</div>
""", unsafe_allow_html=True)

# ---------- utilidades -------------------------------------------------
def load_image(src):
    if isinstance(src, str) and src.startswith("http"):
        return Image.open(requests.get(src, stream=True).raw).convert("RGB")
    elif hasattr(src, "read"):
        return Image.open(src).convert("RGB")
    else:
        raise ValueError("Input no válido (ni URL ni fichero)")

@st.cache_resource(show_spinner=False)
def load_model():
    extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-101-panoptic")
    model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-101-panoptic")
    model.eval()
    return extractor, model

def predict_panoptic(img: Image.Image, extractor, model, thr=0.85):
    inputs = extractor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    panoptic = extractor.post_process_panoptic_segmentation(
        outputs,
        target_sizes=[img.size[::-1]],
        threshold=thr,
    )[0]
    return panoptic, outputs

# ---------- visualización detectron2 -----------------------------------
def visualize_with_detectron2(img_pil: Image.Image, result_dict: dict) -> np.ndarray:
    segments_info = deepcopy(result_dict["segments_info"])
    panoptic_seg = result_dict["segmentation"]
    if isinstance(panoptic_seg, torch.Tensor):
        panoptic_seg = panoptic_seg.cpu().numpy()
    final_h, final_w = panoptic_seg.shape

    meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")

    for seg in segments_info:
        cid = seg.get("category_id", seg.get("label_id"))
        if cid is None:
            raise KeyError("No se encontró 'category_id' ni 'label_id' en segments_info")
        seg["isthing"] = cid in meta.thing_dataset_id_to_contiguous_id
        seg["category_id"] = meta.thing_dataset_id_to_contiguous_id.get(cid, cid) if seg["isthing"] else meta.stuff_dataset_id_to_contiguous_id.get(cid, cid)

    vis = Visualizer(np.array(img_pil.resize((final_w, final_h)))[:, :, ::-1], meta, scale=1.0)
    vis._default_font_size = 18
    panoptic_seg_tensor = torch.from_numpy(panoptic_seg.astype(np.int32))
    vis = vis.draw_panoptic_seg_predictions(panoptic_seg_tensor, segments_info, area_threshold=0)
    return vis.get_image()[:, :, ::-1]

# ---------- visualización matplotlib -----------------------------------
def fig_to_buf(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_panoptic(pano_img: np.ndarray) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(pano_img)
    ax.axis("off")
    return fig_to_buf(fig)

def plot_masks_grid(outputs, keep_bool, ncols=5) -> io.BytesIO:
    masks = outputs.pred_masks[0][keep_bool].sigmoid().cpu()
    n = masks.size(0)
    nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.6 * ncols, 3.6 * nrows), squeeze=False)
    for row in axs:
        for ax in row:
            ax.axis("off")
    for i, m in enumerate(masks):
        r, c = divmod(i, ncols)
        axs[r][c].imshow(m, cmap="cividis")
    fig.tight_layout()
    return fig_to_buf(fig)

# ---------- Streamlit main ---------------------------------------------
def main():
    with st.sidebar:
        st.header("Configuración")
        up = st.file_uploader("Sube una imagen (jpg/png) o deja vacío para usar la de ejemplo", type=["jpg", "jpeg", "png"])
        run_infer = st.button("Ejecutar inferencia")

    default_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = load_image(default_url if up is None else up)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(img, caption="Imagen original", use_container_width=True)

    if run_infer:
        extractor, model = load_model()
        with st.spinner("Realizando inferencia…"):
            panoptic, raw_out = predict_panoptic(img, extractor, model, thr=0.85)

        st.success("Inferencia completada.")
        tabs = st.tabs(["Máscaras individuales", "Segmentación básica", "Con etiquetas (Detectron2)"])

        with tabs[0]:
            st.markdown('<div class="section-subtitle">Máscaras individuales con alta confianza</div>', unsafe_allow_html=True)
            st.markdown('<div class="output-card">Esta sección muestra las máscaras segmentadas detectadas con un nivel de confianza superior al umbral seleccionado. Cada máscara representa una región específica detectada por el modelo en la imagen. Útil para analizar qué objetos o áreas fueron identificados con mayor seguridad.</div>', unsafe_allow_html=True)
            scores = raw_out.logits.softmax(-1)[0, :, :-1].max(-1)[0]
            keep = scores > 0.85
            if keep.sum() == 0:
                st.warning("Ninguna máscara supera el umbral de confianza")
            else:
                colA, colB, colC = st.columns([1, 2, 1])
                with colB:
                    st.image(
                        plot_masks_grid(raw_out, keep),
                        caption=f"Máscaras con confianza > 0.85 ({keep.sum().item()} total)",
                        use_container_width=True,
                    )

        with tabs[1]:
            st.markdown('<div class="section-subtitle">Segmentación panóptica básica</div>', unsafe_allow_html=True)
            st.markdown('<div class="output-card">Aquí se visualiza el mapa completo de segmentación, donde cada color representa una categoría o región segmentada. No se muestran etiquetas, solo el mapa de colores para facilitar la identificación visual rápida.</div>', unsafe_allow_html=True)
            colA, colB, colC = st.columns([1, 2, 1])
            with colB:
                st.image(
                    plot_panoptic(panoptic["segmentation"]),
                    caption="Segmentación panóptica (colores sin etiquetas)",
                    use_container_width=True,
                )

        with tabs[2]:
            st.markdown('<div class="section-subtitle">Segmentación panóptica con etiquetas Detectron2</div>', unsafe_allow_html=True)
            st.markdown('<div class="output-card">Detectron2 es una librería avanzada desarrollada por Facebook AI Research para tareas de visión por computador, especializada en segmentación, detección y reconocimiento de objetos. Aquí usamos Detectron2 para representar visualmente las regiones segmentadas con etiquetas claras y colores distintivos, facilitando la interpretación y análisis de los resultados del modelo DETR. Esta combinación potencia la capacidad de visualización y comprensión de las segmentaciones panópticas generadas.</div>', unsafe_allow_html=True)
            with st.spinner("Visualizando con Detectron2…"):
                colA, colB, colC = st.columns([1, 2, 1])
                with colB:
                    vis_img = visualize_with_detectron2(img, panoptic)
                    st.image(vis_img, caption="Segmentación panóptica con etiquetas (Detectron2)", use_container_width=True)

if __name__ == "__main__":
    main()
