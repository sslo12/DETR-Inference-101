import streamlit as st

# Configuración de página
st.set_page_config(
    page_title="DETR Panoptic",
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


st.markdown('<div class="title">End-to-End Object Detection<br>with Transformers</div>', unsafe_allow_html=True)

# Autores originales
st.markdown("""
    <div class="authors">
        Nicolas Carion, Francisco Massa, Gabriel Synnaeve,<br>
        Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko
    </div>
""", unsafe_allow_html=True)

# Equipo implementador
st.markdown("""
    <div class="team">
        Presentado por:<br>
        María José Clavijo, Shirley Lombana, Santiago Valencia<br>
        Universidad Autónoma de Occidente
    </div>
""", unsafe_allow_html=True)


st.markdown('<div class="nav-button">', unsafe_allow_html=True)
st.page_link("pages/about_detr.py", label="Explorar DETR →")
st.markdown('</div>', unsafe_allow_html=True)
