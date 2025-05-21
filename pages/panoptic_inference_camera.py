import cv2 
import numpy as np
from PIL import Image
import streamlit as st
import torch
from transformers import DetrFeatureExtractor, DetrForSegmentation
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from copy import deepcopy


st.set_page_config(page_title="Inferencia Segmentación Panóptica", layout="centered")

# Estilo CSS personalizado para los botones
st.markdown("""
    <style>
    /* Botón de Iniciar/Detener Cámara - Color turquesa (#0FC2C0) */
    button[kind="secondary"]:nth-of-type(1) {
        background-color: #0FC2C0 !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
    }
    button[kind="secondary"]:nth-of-type(1):hover {
        background-color: #0CABA8 !important;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    """
    Carga y devuelve el modelo DETR preentrenado para segmentación panóptica y su extractor de características.
    """
    extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-101-panoptic")
    model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-101-panoptic")
    model.eval()
    return extractor, model


def predict_panoptic(img, extractor, model, thr=0.85):
    """
    Realiza la predicción panóptica en una imagen usando el modelo DETR.
    """
    inputs = extractor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-procesa la salida del modelo para obtener la segmentación panóptica
    panoptic = extractor.post_process_panoptic_segmentation(
        outputs, target_sizes=[img.size[::-1]], 
        threshold=thr 
    )[0]
    return panoptic


def visualize_with_detectron2(img_pil, result_dict):
    """
    Visualiza los resultados de la segmentación panóptica usando Detectron2.
    """
    segments_info = deepcopy(result_dict["segments_info"])
    
    # Obtiene el mapa de segmentación panóptica
    panoptic_seg = result_dict["segmentation"]

    if isinstance(panoptic_seg, torch.Tensor):
        panoptic_seg = panoptic_seg.cpu().numpy()
    
    # Obtiene los metadatos del dataset COCO panóptico
    meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")

    for seg in segments_info:
        # Obtiene el ID de categoría (usa 'label_id' como fallback)
        cid = seg.get("category_id", seg.get("label_id", 0))
        
        # Determina si es una "cosa" (objeto) o "stuff" (fondo/textura)
        seg["isthing"] = cid in meta.thing_dataset_id_to_contiguous_id
        
        # Mapea el ID de categoría al formato interno de Detectron2
        if seg["isthing"]:
            seg["category_id"] = meta.thing_dataset_id_to_contiguous_id.get(cid, cid)
        else:
            seg["category_id"] = meta.stuff_dataset_id_to_contiguous_id.get(cid, cid)

    # Crea el visualizador con la imagen (convertida a BGR) y los metadatos
    vis = Visualizer(np.array(img_pil)[:, :, ::-1], meta, scale=1.0)
    vis._default_font_size = 10
    
    # Dibuja las predicciones de segmentación panóptica
    vis = vis.draw_panoptic_seg_predictions(
        torch.from_numpy(panoptic_seg.astype(np.int32)),  # Mapa de segmentación
        segments_info,  
        area_threshold=0  # Umbral de área (0 = mostrar todos)
    )
    return vis.get_image()[:, :, ::-1]


def main():
    # Título
    st.markdown(
        """
        <h1 style='text-align: center; margin-bottom: 30px; color: #1E3A8A;'>
        Inferencia Segmentación Panóptica en video con Cámara Web
        </h1>
        """,
        unsafe_allow_html=True
    )
    
    # Subtítulo descriptivo
    st.markdown(
        """
        <div style='text-align: center; margin-bottom: 25px; color: #1E40AF;'>
        <h3>Visualización con etiquetas de clases - Detectron2</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Cargar modelo
    extractor, model = load_model()
    
    # Configuración fija
    CONFIDENCE_THRESHOLD = 0.85
    FRAME_SKIP = 3
    
    # Inicializar estado de la cámara
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False

    # Sección de control de cámara
    st.markdown("---")
    st.markdown("#### Control de Cámara")

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("Iniciar Cámara", key="start_btn")
    
    with col2:
        stop_btn = st.button("Detener", key="stop_btn")

    # Lógica de los botones
    if start_btn and not st.session_state.camera_active:
        st.session_state.camera_active = True
        st.rerun()
    
    if stop_btn and st.session_state.camera_active:
        st.session_state.camera_active = False
        st.rerun()
    
    frame_placeholder = st.empty()
    status_text = st.empty()
    
    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("No se pudo acceder a la cámara web. Verifica los permisos.")
            st.session_state.camera_active = False
            return
        
        frame_count = 0
        
        try:
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    status_text.warning("Error al capturar el fotograma")
                    break
                
                # Redimensionar para mejor rendimiento
                frame = cv2.resize(frame, (640, 480))
                
                # Procesar según FRAME_SKIP
                if frame_count % FRAME_SKIP == 0:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    panoptic = predict_panoptic(img, extractor, model, CONFIDENCE_THRESHOLD)

                    vis_img = visualize_with_detectron2(img, panoptic) 
                    frame_placeholder.image(vis_img, channels="BGR", 
                                          caption="Visualización Detectron2 - Segmentación Panóptica",
                                          use_container_width=True)
                
                # Actualizar contador
                frame_count += 1
                status_text.text(f"Procesando frame {frame_count}")
                
                # Pequeña pausa para permitir la interrupción
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    st.session_state.camera_active = False
                    st.rerun()
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if not st.session_state.camera_active:
                status_text.text("Cámara detenida")

if __name__ == "__main__":
    main()
