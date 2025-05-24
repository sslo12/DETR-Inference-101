# DETR-ResNet-101: Un Enfoque para la Segmentación Panóptica


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
├── home.py                            # Script de inicio o página principal de la aplicación
└── requirements.txt                   # Lista de dependencias del proyecto Python 
```
