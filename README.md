# DETR-ResNet-101: Un Enfoque para la Segmentación Panóptica
Este proyecto es una implementación simplificada del modelo DETR (DEtection TRansformer) propuesto por Facebook AI Research, aplicado específicamente a la tarea de segmentación panóptica.  
Utiliza Streamlit para la visualización interactiva de los resultados y permite probar el modelo de manera local en un entorno aislado.

La aplicación permite cargar una imagen y obtener como salida una segmentación panóptica que combina detección de instancias y segmentación semántica, todo en una sola arquitectura basada en transformers.


## 🔍 ¿Qué hace este modelo?
**DETR-ResNet-101 Panoptic** es un modelo basado en *Transformers* que permite:

- Realizar **segmentación panóptica** precisa combinando detección de objetos y segmentación semántica.
- Identificar tanto **cosas** (objetos individuales) como **stuff** (regiones amorfas).
- Generar salidas con **máscaras, clases e instancias únicas** por píxel.
- Usar una arquitectura **end-to-end** sin necesidad de postprocesamiento como NMS.

```
## 📁 Estructura del proyecto
DETR-Inference-101/
│
├── pages/                             # Scripts para las diferentes secciones de la aplicación
│   ├── about_detr.py                  # Página con información sobre el modelo DETR
│   ├── inference_imgs.py              # Lógica para inferencia en imágenes cargadas
│   ├── panoptic_inference_camera.py   # Inferencia panóptica utilizando la cámara
│   └── panoptic_segmentation.py       # información sobre la segmentación panóptica
│
├── static/                            # Carpeta para archivos estáticos
│
├── .gitignore                         # Archivos y carpetas ignoradas por Git
├── Dockerfile                         # Configuración para contenedor Docker
├── README.md                          # Documentación principal del proyecto 
├── home.py                            # Script de página principal de la aplicación
└── requirements.txt                   # Lista de dependencias del proyecto Python 
```
## 👥 Créditos de Desarrollo

Esta implementación fue realizada como una prueba académica de inferencia y visualización del modelo **DETR-ResNet-101** aplicado a la **segmentación panóptica**.  
No somos autores del modelo original, únicamente replicamos su funcionamiento como parte de una actividad de aprendizaje.

**Desarrollado por:**

- María José Clavijo Rojas  
- Shirley Stefany Lombana
- Santiago Valencia
---

## 📚 Fuente del Modelo

Este proyecto está basado en el modelo DETR publicado por Facebook AI Research.

- 📄 **Referencia APA:**  
  Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). *End-to-End Object Detection with Transformers*. arXiv preprint arXiv:2005.12872. https://doi.org/10.48550/arXiv.2005.12872

- 🔗 **Repositorio oficial de GitHub:**  
  [https://github.com/facebookresearch/detr](https://github.com/facebookresearch/detr)
