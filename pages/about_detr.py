import streamlit as st
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="DETR: Detección y Segmentación",
    layout="wide"  
)

# Estilos CSS personalizados
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
    .section-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1E40AF;
        margin: 1.2rem 0 0.8rem 0;
    }
    .task-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #1E40AF;
    }
    .io-card {
        background-color: #EFF6FF;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .output-card {
        background-color: #EFF6FF;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #1E40AF;
    }
    .image-wrapper {
        display: flex;
        justify-content: center;
        margin: 2rem 0;
    }
    .nav-button {
        text-align: center;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title">Tareas del Modelo DETR</div>', unsafe_allow_html=True)


col1, col2 = st.columns(2)

with col1:
    # Sección: Detección de Objetos
    st.markdown('<div class="section-title">Detección de Objetos</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="task-card">
            <p>DETR (DEtection TRansformer) está diseñado para detectar objetos en imágenes, prediciendo un conjunto de bounding boxes (cajas delimitadoras) y sus respectivas etiquetas de clase.</p>
            <p>DETR aborda la detección como un problema de predicción directa de conjuntos.</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Sección: Segmentación Panóptica
    st.markdown('<div class="section-title">Segmentación Panóptica</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="task-card">
            <p>DETR puede extenderse para realizar segmentación panóptica, que combina:</p>
            <ul>
                <li><strong>Segmentación semántica:</strong> Clasifica cada píxel en categorías de regiones ("stuff") como cielo, carretera o vegetación</li>
                <li><strong>Segmentación de instancias:</strong> Identifica objetos individuales (things) como "personas" o "coches"</li>
            </ul>
            <p>Esto se logra añadiendo una cabeza adicional de máscara binaria a la salida del decodificador.</p>
        </div>
    """, unsafe_allow_html=True)

# Imagen 
st.markdown('<div class="image-container">', unsafe_allow_html=True)
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])  # Columnas para centrado
    with col2:
        image = Image.open("static/semantic_comparison.png")
        st.image(image, width=550)  

# Sección de Entradas
st.markdown('<div class="section-title">Entradas de la Red</div>', unsafe_allow_html=True)
st.markdown("""
    <div class="io-card">
        <p><strong>Formato de entrada:</strong></p>
        <ul>
            <li>Imagen RGB como tensor <strong>[3 canales, altura, ancho]</strong></li>
            <li>Normalizada con media y desviación estándar de ImageNet</li>
            <li>Redimensionada (lado corto a 800px, sin exceder 1333px en el lado largo)</li>
        </ul>
        <p>La imagen pasa por una red backbone CNN (como ResNet), para extraer un mapa de características de menor resolución.</p>
    </div>
""", unsafe_allow_html=True)

# Sección de Salidas
st.markdown('<div class="section-title">Salidas de la Red</div>', unsafe_allow_html=True)

# Caja para Detección de Objetos
st.markdown("""
    <div class="output-card">
        <p><strong>Detección de objetos:</strong></p>
        <ul>
            <li>Coordenadas de las cajas delimitadoras</li>
            <li>Probabilidades de <strong>clases de objetos</strong> (incluyendo clase <code>∅</code> para "no objeto")</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Caja para Segmentación Panóptica
st.markdown("""
    <div class="output-card">
        <p><strong>Segmentación panóptica:</strong></p>
        <ul>
            <li>Máscaras binarias para cada objeto detectado</li>
            <li>Combina las máscaras para crear un Mapa Panóptico con:
                <ul>
                    <li>IDs únicos para instancias</li>
                    <li>Categorías semánticas para fondo</li>
                </ul>
            </li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Navegación
st.markdown("---")
st.markdown('<div class="nav-button">', unsafe_allow_html=True)
st.page_link("home.py", label="Volver a la Página Principal", use_container_width=False)
st.markdown('</div>', unsafe_allow_html=True)
