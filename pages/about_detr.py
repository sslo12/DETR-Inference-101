import streamlit as st
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="DETR: Detección y Segmentación",
    layout="wide"
)

# Estilos CSS personalizados con diferencia entre título y subtítulo
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
        font-size: 1.8rem; /* Más grande para título principal */
        font-weight: 700;
        color: #1E40AF;
        margin: 2rem 0 1rem 0;
    }
    .section-subtitle {
        font-size: 1.3rem; /* Más pequeño para subtítulo */
        font-weight: 600;
        color: #1E40AF;
        margin: 1.2rem 0 0.6rem 0;
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
    .output-card, .architecture-section {
        background-color: #D6E8FF; /* Caja azul clara */
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        border-left: 6px solid #1E40AF; /* Borde azul fuerte */
        color: #0D47A1;
        line-height: 1.5;
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

# Título más pequeño en español antes de las imágenes
st.markdown('<h2 style="color:#1E3A8A; font-weight:700; text-align:center; margin-bottom:1rem;">Arquitectura DETR</h2>', unsafe_allow_html=True)

with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])  # Columna central para centrar
    with col2:
        # Primera imagen con título
        image1 = Image.open("static/detr_architecture.jpg")
        st.image(image1, width=450)
        st.markdown('<p style="text-align:center; font-weight:bold; color:#1E3A8A; margin-top:0.5rem;">DETR Architecture</p>', unsafe_allow_html=True)

        # Segunda imagen con título
        image2 = Image.open("static/detr_detailed_architecture.jpg")
        st.image(image2, width=450)
        st.markdown('<p style="text-align:center; font-weight:bold; color:#1E3A8A; margin-top:0.5rem;">DETR Detailed Architecture</p>', unsafe_allow_html=True)

# Sección nueva: Arquitectura DETR (texto completo que enviaste)
st.markdown('<div class="section-subtitle">1.1. Arquitectura DETR</div>', unsafe_allow_html=True)
st.markdown("""
<div class="architecture-section">
DETR usa una CNN backbone convencional para aprender una representación 2D de la imagen de entrada. Normalmente, la salida tiene canales C=2048 y tamaño H,W=H0/32, W0/32, donde H0 y W0 son altura y ancho de la imagen original.<br><br>
La backbone suele ser ResNet-50 o ResNet-101 preentrenada en ImageNet con batch norm congelado, llamadas DETR y DETR-R101 respectivamente.<br><br>
La resolución de características puede incrementarse aplicando dilatación en la última etapa de la backbone, generando DETR-DC5 y DETR-DC5-R101.<br><br>
El modelo aplana estas características y les añade un codificado posicional antes de pasarlas a un codificador Transformer.<br><br>
Se usa un Transformer con 6 capas de codificador y 6 de decodificador, ancho 256 y 8 cabezas de atención.<br><br>
El decodificador recibe un número fijo pequeño de embeddings aprendidos llamados <em>object queries</em>, y atiende la salida del codificador.<br><br>
Usando autoatención y atención codificador-decodificador, el modelo razona globalmente sobre los objetos y sus relaciones, con contexto de toda la imagen.<br><br>
Cada embedding del decodificador pasa por una red feed-forward (FFN) compartida que predice detección (clase y caja) o “no objeto”.<br><br>
Las cajas tienen coordenadas normalizadas del centro, altura y ancho relativas a la imagen, y la clase se predice con softmax.
</div>
""", unsafe_allow_html=True)


st.markdown('<div class="section-subtitle">1.2. Pérdidas Auxiliares de Decodificador</div>', unsafe_allow_html=True)
st.markdown("""
<div class="architecture-section">
Durante el entrenamiento, se añaden pérdidas auxiliares en el decodificador para ayudar al modelo a predecir correctamente el número de objetos de cada clase.<br><br>
Después de cada capa del decodificador se añaden predicciones FFN y la pérdida Hungarian.<br><br>
Todas las FFNs comparten parámetros, y se usa una normalización Layer Norm compartida para las entradas de estas predicciones.
</div>
""", unsafe_allow_html=True)


# Sección 2. Resultados de Detección de Objetos
st.markdown('<div class="section-title">2. Resultados de Detección de Objetos</div>', unsafe_allow_html=True)

# 2.1 Comparaciones con Faster R-CNN
st.markdown('<div class="section-subtitle">2.1. Comparaciones con Faster R-CNN</div>', unsafe_allow_html=True)

# Imagen centrada con título debajo
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image_results = Image.open("static/results_detr.jpg")
        st.image(image_results, width=600)
        st.markdown('<p style="text-align:center; font-weight:bold; color:#1E3A8A; margin-top:0.5rem;">Comparación con Faster R-CNN con backbone ResNet-50 y ResNet-101 en el conjunto de validación COCO (‘+’: 9× programación)</p>', unsafe_allow_html=True)

# Texto descriptivo después de la imagen
st.markdown("""
<div class="architecture-section">
El entrenamiento del modelo base por 300 épocas en 16 GPUs V100 toma 3 días, con 4 imágenes por GPU (batch size total 64).<br><br>
DETR es competitivo con Faster R-CNN con igual número de parámetros, alcanzando 42 AP en val COCO.<br><br>
Mejora principalmente APL (+7.8), aunque aún está por detrás en APS (-5.5).<br><br>
DETR-DC5 con similar cantidad de parámetros y FLOPs tiene mayor AP, pero también queda detrás en APS.
</div>
""", unsafe_allow_html=True)

# 2.2 Capas Transformer
st.markdown('<div class="section-subtitle">2.2. Capas Transformer</div>', unsafe_allow_html=True)

# Imagen centrada con título debajo
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image_transformer = Image.open("static/transformer_layers.jpg")
        st.image(image_transformer, width=600)
        st.markdown('<p style="text-align:center; font-weight:bold; color:#1E3A8A; margin-top:0.5rem;">Transformer Layers</p>', unsafe_allow_html=True)

# Texto descriptivo después de la imagen
st.markdown("""
<div class="architecture-section">
Estudio con modelo DETR basado en ResNet-50 con 6 capas encoder y decoder, ancho 256.<br><br>
Tiene 41.3M parámetros, alcanza 40.6 y 42.0 AP en schedules cortos y largos, y corre a 28 FPS, similar a Faster R-CNN-FPN.<br><br>
Sin capas encoder, el AP total cae 3.9 puntos, con caída mayor (6.0 AP) en objetos grandes.<br><br>
Se hipotetiza que el encoder con razonamiento global ayuda a separar objetos.
</div>
""", unsafe_allow_html=True)

# 2.3 Atención Encoder
st.markdown('<div class="section-subtitle">2.3. Atención Encoder</div>', unsafe_allow_html=True)

# Imagen centrada con título debajo
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image_transformer = Image.open("static/encoder_attention.jpg")
        st.image(image_transformer, width=600)
        st.markdown('<p style="text-align:center; font-weight:bold; color:#1E3A8A; margin-top:0.5rem;">La autoatención del codificador para un conjunto de puntos de referencia. El codificador es capaz de separar instancias individuales.</p>', unsafe_allow_html=True)
        
st.markdown("""
<div class="architecture-section">
La auto-atención del encoder para un conjunto de puntos de referencia.<br><br>
El encoder es capaz de separar instancias individuales.<br><br>
La imagen muestra los mapas de atención de la última capa encoder de un modelo entrenado, enfocado en puntos específicos.<br><br>
El encoder ya parece separar instancias, facilitando extracción y localización para el decoder.
</div>
""", unsafe_allow_html=True)

# Imagen adicional centrada con título pequeño debajo
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image_additional = Image.open("static/decoder_layer.jpg")  # Cambia el nombre del archivo
        st.image(image_additional, width=400)  # Tamaño más pequeño
        st.markdown('<p style="text-align:center; font-weight:600; color:#1E3A8A; margin-top:0.3rem; font-size:0.9rem;">Rendimiento de AP y AP50 después de cada capa del decodificador</p>', unsafe_allow_html=True)

# 2.4 Atención Decoder
st.markdown('<div class="section-subtitle">2.4. Atención Decoder</div>', unsafe_allow_html=True)

# Imagen centrada con título debajo
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image_transformer = Image.open("static/decoder_attention.jpg")
        st.image(image_transformer, width=600)
        st.markdown('<p style="text-align:center; font-weight:bold; color:#1E3A8A; margin-top:0.5rem;">RVisualización de la atención del decodificador para cada objeto predicho (imágenes del conjunto de validación COCO).</p>', unsafe_allow_html=True)
st.markdown("""
            
<div class="architecture-section">
Atención decoder para cada objeto predicho (imágenes del set COCO val).<br><br>
Es atención bastante local, concentrándose en extremidades de objetos como cabezas o patas.<br><br>
Se hipotetiza que luego que el encoder separa las instancias, el decoder solo atiende las extremidades para extraer clase y límites.
</div>
""", unsafe_allow_html=True)

# 2.5 FFN
st.markdown('<div class="section-subtitle">2.5. FFN</div>', unsafe_allow_html=True)
st.markdown("""
<div class="architecture-section">
Intento de remover FFN dejando solo atención en Transformer.<br><br>
Reduciendo parámetros de 41.3M a 28.7M (solo 10.8M en Transformer), el rendimiento cae 2.3 AP.<br><br>
Conclusión: las FFNs son importantes para buenos resultados.
</div>
""", unsafe_allow_html=True)

# 2.6 Codificado Posicional
st.markdown('<div class="section-subtitle">2.6. Codificado Posicional</div>', unsafe_allow_html=True)

# Imagen centrada con título debajo
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image_transformer = Image.open("static/position_encoding.jpg")
        st.image(image_transformer, width=600)
        st.markdown('<p style="text-align:center; font-weight:bold; color:#1E3A8A; margin-top:0.5rem;">Resultados comparativos de diferentes codificaciones posicionales frente al modelo base (última fila).</p>', unsafe_allow_html=True)
st.markdown("""

<div class="architecture-section">
Resultados para distintos codificados posicionales frente al baseline (última fila).<br><br>
Hay dos tipos: codificado posicional espacial y codificado posicional de salida (object queries).<br><br>
Eliminando codificado espacial, solo codificado de salida, se pierde 7.8 AP.<br><br>
Sorprendentemente, no pasar codificado espacial en encoder solo reduce 1.3 AP.<br><br>
La auto-atención global en encoder, FFN, capas de decoder y codificados posicionales contribuyen significativamente al rendimiento final.
</div>
""", unsafe_allow_html=True)

# 2.7 Pérdida
st.markdown('<div class="section-subtitle">2.7. Pérdida</div>', unsafe_allow_html=True)

# Imagen centrada con título debajo
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image_transformer = Image.open("static/loss.jpg")
        st.image(image_transformer, width=600)
        st.markdown('<p style="text-align:center; font-weight:bold; color:#1E3A8A; margin-top:0.5rem;">Efecto de los componentes de la función de pérdida en el AP (Average Precision).</p>', unsafe_allow_html=True)
st.markdown("""

<div class="architecture-section">
Componentes de la pérdida: clasificación, l1 distancia de caja, y GIoU.<br><br>
Usar l1 sin GIoU muestra resultados pobres.
</div>
""", unsafe_allow_html=True)

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