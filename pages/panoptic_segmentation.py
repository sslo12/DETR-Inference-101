import streamlit as st
from PIL import Image

# Configuración de página
st.set_page_config(
    page_title="DETR - Segmentación Panóptica",
    layout="centered",
    initial_sidebar_state="auto"
)

# CSS personalizado
st.markdown("""
    <style>
    .title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin: 2rem 0;
        line-height: 1.2;
    }
    .authors {
        font-size: 1.2rem;
        text-align: center;
        color: #1E40AF;
        margin-bottom: 3rem;
    }
    .team {
        font-size: 1rem;
        text-align: center;
        color: #1E3A8A;
        margin-top: 5rem;
        padding-top: 1rem;
        border-top: 1px solid #EFF6FF;
    }
    .nav-button {
        text-align: center;
        margin-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)


with st.container():
    st.markdown("## 1. ¿Qué es la Segmentación Panóptica (PS)?")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image1 = Image.open("./static/sp/que_es_sp.jpg")
        st.image(image1, width=450)
        st.markdown('<p style="text-align:center; font-weight:bold; color:#1E3A8A; margin-top:0.5rem;">Comparación de tipos de segmentación</p>', unsafe_allow_html=True)

with st.container():
    st.write("""
    Para una imagen dada (a), mostramos la verdad fundamental para: (b) segmentación semántica (etiquetas de clase por píxel), (c) segmentación de instancia (máscara por objeto y etiqueta de clase) y (d) la tarea de segmentación panóptica propuesta (etiquetas de instancia por píxel class+).

    **Cosas:** objetos contables como personas, animales y herramientas.

    **Cosas:** regiones amorfas de textura o material similar, como hierba, cielo y carretera.

    **La tarea de PS:**
    - abarca tanto las cosas como las clases de cosas;
    - utiliza un formato simple pero general; y
    - introduce una métrica de evaluación uniforme para todas las clases.

    La segmentación panóptica generaliza la segmentación semántica y de instancia y se espera que la tarea unificada presente nuevos desafíos y permita nuevos métodos innovadores.
    """)

# 2. Calidad Panóptica (PQ) Métrica
with st.container():
    st.markdown("## 2. Calidad Panóptica (PQ) Métrica")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image2 = Image.open("./static/sp/calidad_sp.jpg")
        st.image(image2, width=450)
        st.markdown('<p style="text-align:center; font-weight:bold; color:#1E3A8A; margin-top:0.5rem;">Ilustración de juguete de la verdad fundamental y segmentaciones panópticas predichas de una imagen</p>', unsafe_allow_html=True)

with st.container():
    st.write("""
    Implica dos pasos: (1) coincidencia de segmentos y (2) Cálculo de PQ dados los partidos.
    """)

    st.markdown("### 2.1. Coincidencia de Segmento")

    st.write("""
    Un segmento predicho y un segmento de verdad fundamental pueden solo coincide si su intersección sobre la unión (IoU) es estrictamente mayor que 0.5.

    Este requisito, junto con la propiedad de no superposición de una segmentación panóptica, da un coincidencia única: puede haber como máximo un segmento predicho coincidente con cada segmento de verdad fundamental.

    Debido a la propiedad de singularidad, para IoU>0.5, cualquier estrategia de coincidencia razonable (incluida la codiciosa y óptima) producirá una coincidencia idéntica.

    Los umbrales más bajos son innecesarios ya que las coincidencias con IoU≤0.5 son raras en la práctica.

    Un ejemplo de juguete se muestra arriba: Pares de segmentos del mismo color tener IoU mayor que 0.5 y son por lo tanto emparejado. La figura muestra cómo los segmentos para el clase de persona se dividen en verdaderos positivos TP, falsos negativos FN y falsos positivos FP.
    """)

with st.container():
    st.markdown("### 2.2. PQ Computación")

    st.write("""
    Para cada clase, la coincidencia única divide los segmentos de verdad predichos y básicos en tres conjuntos: verdaderos positivos (TP), falsos positivos (FP) y falsos negativos (FN), que representan pares de segmentos coincidentes, segmentos predichos inigualables y segmentos de verdad básica inigualables, respectivamente.

    PQ está definido como:
    """)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image3 = Image.open("./static/sp/pq_definido_sp.jpg")
        st.image(image3, width=300)
        st.markdown('<p style="text-align:center; font-weight:bold; color:#1E3A8A; margin-top:0.5rem;">Panoptic Quality</p>', unsafe_allow_html=True)

    st.write("""
    dónde:
    """)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image4 = Image.open("./static/sp/pq_definido_1_sp.jpg")
        st.image(image4, width=300)
        st.markdown('<p style="text-align:center; font-weight:bold; color:#1E3A8A; margin-top:0.5rem;">Componente de calidad de segmentación</p>', unsafe_allow_html=True)

    st.write("""
    es simplemente el IoU promedio de segmentos coincidentes, mientras.
    """)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image5 = Image.open("./static/sp/pq_definido_2_sp.jpg")
        st.image(image5, width=300)
        st.markdown('<p style="text-align:center; font-weight:bold; color:#1E3A8A; margin-top:0.5rem;"></p>', unsafe_allow_html=True)

    st.write("""
    se agrega al denominador a penalizar segmentos sin coincidencias.

    Si PQ se multiplica y se divide por el tamaño del conjunto de TP, entonces PQ se puede ver como la multiplicación de a calidad de segmentación (SQ) término y a calidad de reconocimiento (RQ) término:
    """)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image6 = Image.open("./static/sp/pq_definido_3_sp.jpg")
        st.image(image6, width=350)
        st.markdown('<p style="text-align:center; font-weight:bold; color:#1E3A8A; margin-top:0.5rem;">Descomposición de PQ en SQ y RQ</p>', unsafe_allow_html=True)

    st.write("""
    RQ es lo familiar Puntuación de F1 ampliamente utilizado para la estimación de calidad en configuraciones de detección. SQ es simplemente el IoU promedio de segmentos coincidentes.
    """)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image7 = Image.open("./static/sp/pq_definido_4_sp.jpg")
        st.image(image7, width=400)
        st.markdown('<p style="text-align:center; font-weight:bold; color:#1E3A8A; margin-top:0.5rem;">Tabla de descomposición de métricas</p>', unsafe_allow_html=True)

    st.write("La descomposición anterior proporciona información adicional para el análisis.")

with st.container():
    st.markdown("### 2.3. NMS Post-Procesamiento")

    st.write("""
    Para medir PQ, primero debemos resolver estas superposiciones. Un simple supresión no máxima (NMS) se realiza un procedimiento similar.

    Nosotros primero ordene los segmentos predichos por sus puntajes de confianza y eliminar instancias con puntajes bajos.

    Entonces, nosotros iterar sobre instancias ordenadas, empezando por los más seguros.

    Para cada caso nosotros primero elimine los píxeles que se han asignado a segmentos anteriores, entonces si queda una fracción suficiente del segmento, nosotros acepte la parte no superpuesta, de lo contrario descartamos todo el segmento.
    """)

# 3. Resultados Experimentales
with st.container():
    st.markdown("## 3. Resultados Experimentales")

    st.markdown("### 3.1. PQ para la Cosa")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image8 = Image.open("./static/sp/result_sp.jpg")
        st.image(image8, width=650)
        st.markdown('<p style="text-align:center; font-weight:bold; color:#1E3A8A; margin-top:0.5rem;">Resultados de la máquina en la segmentación de instancias (clases de cosas ignoradas)</p>', unsafe_allow_html=True)

    st.write("""
    Algunos enfoques SOTA como Máscara R-CNN y se evalúan G-RMI.

    AP^NO: AP de las predicciones no superpuestas.

    La eliminación de superposiciones perjudica a AP ya que los detectores se benefician de predecir múltiples hipótesis superpuestas.

    Los métodos con mejor AP también tienen mejor AP^NO y también PQ mejorado.
    """)

with st.container():
    st.markdown("### 3.2. PQ para Cosas")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image9 = Image.open("./static/sp/result_1_sp.jpg")
        st.image(image9, width=450)
        st.markdown('<p style="text-align:center; font-weight:bold; color:#1E3A8A; margin-top:0.5rem;">Resultados de la máquina en segmentación semántica (clases de cosas ignoradas)</p>', unsafe_allow_html=True)

    st.write("""
    Los métodos con mejor IoU media también muestran mejores resultados de PQ.
    """)

with st.container():
    st.markdown("### 3.3. PQ general")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image10 = Image.open("./static/sp/sq_general_sp.jpg")
        st.image(image10, width=450)
        st.markdown('<p style="text-align:center; font-weight:bold; color:#1E3A8A; margin-top:0.5rem;">Rendimiento humano vs. máquina</p>', unsafe_allow_html=True)

    st.write("""
    Para SQ, las máquinas rastrean a los humanos solo ligeramente.

    Por otro lado, la máquina RQ es dramáticamente más baja que la RQ humana, especialmente en ADE20k y Vistas.

    Esto implica que el reconocimiento, es decir, la clasificación, es el principal desafío para los métodos actuales. En general, existe una brecha significativa entre el rendimiento humano y de la máquina.
    """)

with st.container():
    st.markdown("### 3.4. Visualización")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image11 = Image.open("./static/sp/visualizacion_sp.jpg")
        st.image(image11, width=650)
        st.markdown('<p style="text-align:center; font-weight:bold; color:#1E3A8A; margin-top:0.5rem;">Resultados de segmentación panóptica en Cityscapes (izquierda dos) y ADE20k (derecha tres)</p>', unsafe_allow_html=True)

    st.write("""
    Las visualizaciones de las salidas panópticas se muestran arriba.

    La segmentación panóptica es un tipo de combinación de segmentación semántica y segmentación de instancias.

    El propósito principal de esta historia es introduzca las métricas PQ, SQ y RQ.
    """)