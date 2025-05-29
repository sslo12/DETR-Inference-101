import cv2 
import numpy as np
from PIL import Image
import streamlit as st
import torch
from transformers import DetrFeatureExtractor, DetrForSegmentation
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from copy import deepcopy
import tempfile
import time


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
    
    # Post-procesa la salida del modelo para obtener la segmentación panóptica con umbral de confianza
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


def camera_inference():
    """
    Ejecuta inferencia de segmentación panóptica en tiempo real usando cámara web.
    """

    st.markdown(
        """
        <h1 style='text-align: center; margin-bottom: 30px; color: #1E3A8A;'>
        Inferencia Segmentación Panóptica con Cámara Web
        </h1>
        """,
        unsafe_allow_html=True
    )
    
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
    FRAME_SKIP = 3 # Procesar cada N frames
    
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
    
    # Contenedores para actualizar imagen y métricas en tiempo real
    frame_placeholder = st.empty()
    metrics_placeholder = st.empty()

    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("No se pudo acceder a la cámara web. Verifica los permisos.")
            st.session_state.camera_active = False
            return
        
        frame_count = 0
        processed_frames = 0
        total_inference_time = 0
        
        try:
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    metrics_placeholder.warning("Error al capturar el fotograma")
                    break
                
                # Redimensionar para mejor rendimiento
                frame = cv2.resize(frame, (640, 480))
                
                # Contador de frames totales
                frame_count += 1
                
                if frame_count % FRAME_SKIP == 0:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    # Medir tiempo de inferencia
                    inference_start = time.time()
                    panoptic = predict_panoptic(img, extractor, model, CONFIDENCE_THRESHOLD)
                    inference_time = time.time() - inference_start
                    
                    total_inference_time += inference_time
                    processed_frames += 1
                    avg_inference_time = total_inference_time / processed_frames if processed_frames > 0 else 0
                    
                    vis_img = visualize_with_detectron2(img, panoptic)
                    
                    # Mostrar la imagen segmentada
                    frame_placeholder.image(
                        vis_img, 
                        channels="BGR", 
                        caption="Resultados Segmentación Panóptica por Cámara",
                        use_container_width=True
                    )
                    
                    # Mostrar estadísticas de rendimiento
                    metrics_placeholder.markdown(
                        f"""
                        **Métricas de Rendimiento**  
                        • Frame actual: {frame_count}  
                        • Frames procesados: {processed_frames}  
                        • Tiempo de inferencia (frame actual): {inference_time:.3f}s  
                        • Tiempo promedio de inferencia: {avg_inference_time:.3f}s
                        """
                    )
                
                # Pequeña pausa para permitir la interrupción
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    st.session_state.camera_active = False
                    st.rerun()
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if not st.session_state.camera_active:
                metrics_placeholder.text("Cámara detenida")


def video_inference():
    """
    Ejecuta inferencia de segmentación panóptica sobre un archivo de video cargado.
    """

    st.markdown(
        """
        <h1 style='text-align: center; margin-bottom: 30px; color: #1E3A8A;'>
        Inferencia Segmentación Panóptica en Video
        </h1>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div style='text-align: center; margin-bottom: 25px; color: #1E40AF;'>
        <h3>Visualización con etiquetas de clases - Detectron2</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    extractor, model = load_model()
    
    # Configuración fija
    CONFIDENCE_THRESHOLD = 0.85 # Umbral mínimo para detecciones
    FRAME_SKIP = 3
    DISPLAY_WIDTH = 800  

    st.markdown("---")
    st.markdown("#### Sube un video (mp4, avi, mov)")

    video_file = st.file_uploader(
        label="", 
        type=["mp4", "avi", "mov"], 
        label_visibility="collapsed" 
    )
 
    if 'video_active' not in st.session_state:
        st.session_state.video_active = False
    
    if video_file:
        col1, col2 = st.columns(2)
        with col1:
            process_btn = st.button("Procesar Video", key="process_vid_btn")
        
        with col2:
            stop_btn = st.button("Detener Procesamiento", key="stop_vid_btn", disabled=not st.session_state.video_active)
        
        if process_btn and not st.session_state.video_active:
            st.session_state.video_active = True
            st.rerun()
        
        if stop_btn and st.session_state.video_active:
            st.session_state.video_active = False
            st.rerun()

        if st.session_state.video_active:
            # Guardar el archivo temporalmente
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            
            # Abrir el archivo de video
            cap = cv2.VideoCapture(tfile.name)
            if not cap.isOpened():
                st.error("No se pudo abrir el archivo de video")
                st.session_state.video_active = False
                return
            
            # Obtener dimensiones originales
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            aspect_ratio = width / height
            new_height = int(DISPLAY_WIDTH / aspect_ratio)
            
            # Contenedores para imagen y métricas
            frame_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            frame_count = 0
            processed_frames = 0
            total_inference_time = 0
            
            try:
                while cap.isOpened() and st.session_state.video_active:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Procesar solo algunos frames (salto)
                    if frame_count % FRAME_SKIP == 0:
                        frame = cv2.resize(frame, (DISPLAY_WIDTH, new_height)) # Redimensionar el frame manteniendo proporción
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        
                        # Inferencia
                        inference_start = time.time()
                        panoptic = predict_panoptic(img, extractor, model, CONFIDENCE_THRESHOLD)
                        inference_time = time.time() - inference_start
                        
                        total_inference_time += inference_time
                        processed_frames += 1
                        avg_inference_time = total_inference_time / processed_frames if processed_frames > 0 else 0
                        
                        vis_img = visualize_with_detectron2(img, panoptic)
                        
                        # Mostrar resultados
                        frame_placeholder.image(
                            vis_img, 
                            channels="BGR",
                            caption="Resultados Segmentación Panóptica del Video",
                            use_container_width=True
                        )
                        
                        # Mostrar métricas
                        metrics_placeholder.markdown(
                            f"""
                            **Métricas de Rendimiento**  
                            • Frame actual: {frame_count}  
                            • Frames procesados: {processed_frames}  
                            • Tiempo de inferencia (frame actual): {inference_time:.3f}s  
                            • Tiempo promedio de inferencia: {avg_inference_time:.3f}s
                            """
                        )
                    
                    # Control de salida, pausa para permitir la interrupción
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        st.session_state.video_active = False
                        st.rerun()
                    
            finally:
                cap.release()
                cv2.destroyAllWindows()
                tfile.close()
                if not st.session_state.video_active:
                    metrics_placeholder.text("Procesamiento detenido")


def main():

    # Estilo CSS personalizado para las pestañas
    st.markdown("""
        <style>
            .stTabs [data-baseweb="tab-list"] {
                gap: 10px;
            }
            
            .stTabs [data-baseweb="tab"] {
                background-color: #F0F2F6;
                color: #1E3A8A;
                padding: 10px 20px;
                border-radius: 8px 8px 0 0;
                transition: all 0.3s ease;
            }
            
            .stTabs [data-baseweb="tab"]:hover {
                background-color: #D1D5DB;
                color: #1E3A8A;
            }
            
            .stTabs [aria-selected="true"] {
                background-color: #1E3A8A !important;
                color: white !important;
            }
        </style>
        """, unsafe_allow_html=True)


    tab1, tab2 = st.tabs(["Subir Video", "Cámara Web"])
    
    with tab1:
        video_inference()
    
    with tab2:
        camera_inference()


if __name__ == "__main__":
    main()
