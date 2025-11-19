# Documentación del Proyecto: Vision CEX Backend

## 1. Resumen Ejecutivo (Stakeholders)

**Vision CEX** es una plataforma de inteligencia artificial diseñada para la inspección automatizada de flotas de autobuses. Su objetivo principal es detectar daños físicos (abolladuras, rayones, roturas) en tiempo real o mediante análisis de video, optimizando el mantenimiento y reduciendo costos operativos.

### Flujo de Valor
1.  **Captura**: El sistema se conecta a cámaras de seguridad (RTSP) o procesa archivos de video.
2.  **Detección Inteligente**: Utiliza modelos de visión computacional para identificar autobuses y segmentar sus partes.
3.  **Análisis de Daños**: Cuando se detecta un autobús, el sistema emplea "Modelos de Lenguaje Visual" (VLM) avanzados (como ERNIE o LLaVA) para "mirar" el vehículo y razonar si existe algún daño visible.
4.  **Reporte**: Se generan alertas y reportes detallados con la ubicación y severidad del daño.

---

## 2. Arquitectura Técnica

El proyecto sigue una **Clean Architecture** estricta para garantizar la escalabilidad y mantenibilidad.

### 2.1. Estructura Modular
El backend está construido con **FastAPI** y organizado en capas:

*   **Domain Layer (`app/domain`)**:
    *   Contiene la lógica de negocio pura.
    *   Define *Entidades* (ej. `DamageReport`, `BoundingBox`) y *Interfaces de Repositorio* (ej. `ErnieRepository`, `DamageAnalysisRepository`).
    *   No depende de librerías externas ni frameworks.
*   **Data Layer (`app/data`)**:
    *   Implementa las interfaces del dominio.
    *   Contiene *Adaptadores* para servicios externos (ej. `YoloUltralyticsAdapter`, `ErnieClient`, `DeepSeekClient`).
    *   Maneja la interacción con bases de datos o APIs.
*   **Presentation Layer (`app/presentation`)**:
    *   Expone la funcionalidad a través de una API REST.
    *   Maneja la entrada/salida HTTP (Routers, Request/Response Models).
*   **Core Layer (`app/core`)**:
    *   Configuración global e inyección de dependencias (`ServiceLocator`).

### 2.2. Contenedores y GPU (Docker)
El sistema está "dockerizado" para asegurar consistencia entre entornos de desarrollo y producción.

*   **Docker Compose**: Orquesta el servicio de la API y dependencias (Redis, etc.).
*   **Soporte GPU (NVIDIA RTX 5070)**:
    *   Se utiliza la imagen base `nvidia/cuda` para acceso directo al hardware.
    *   Configuración `runtime: nvidia` en `docker-compose.yml` permite al contenedor acceder a la GPU del host.
    *   Librerías como `torch`, `ultralytics` y `autoawq` están configuradas para usar CUDA.

---

## 3. Pipeline de Detección (Paso a Paso)

El núcleo del sistema es un pipeline de procesamiento de video diseñado para ser eficiente y preciso.

### Paso 1: Ingesta y Detección de Movimiento
*   **Entrada**: Flujo RTSP en tiempo real o archivo de video.
*   **Optimización**: Antes de procesar con modelos pesados, se aplica una detección de movimiento (background subtraction) para ignorar frames estáticos y ahorrar recursos de GPU.

### Paso 2: Detección y Segmentación de Buses (YOLO v11-seg)
*   **Modelo**: Se utiliza un modelo **YOLO (v8/v11)** entrenado específicamente para segmentación de instancias.
*   **Acción**:
    *   Identifica la presencia de un autobús en el frame.
    *   Genera una "máscara" (segmentación) que delimita exactamente los píxeles que corresponden al autobús, separándolo del fondo.
*   **Implementación**: `YoloUltralyticsAdapter` carga el modelo `.pt` localmente y ejecuta la inferencia en la GPU.

### Paso 3: Extracción de Snapshots (Casos Sospechosos)
*   Si se detecta un autobús con alta confianza, el sistema extrae un "snapshot" (recorte de la imagen) de la región de interés.
*   Esta imagen se prepara para un análisis más profundo.

### Paso 4: Análisis con Modelos de Lenguaje Visual (LVM)
Aquí reside la inteligencia "cognitiva" del sistema. Se utilizan modelos masivos para razonar sobre la imagen.

*   **Modelos Explorados**:
    *   **ERNIE-4.5-VL (Baidu)**: Modelo potente para descripción detallada y VQA (Visual Question Answering). Integrado vía `ErnieClient`.
    *   **DinoGround / LLaVA**: Alternativas open-source para grounding y descripción.
*   **Proceso**:
    *   Se envía el snapshot al modelo LVM con un prompt específico: *"¿Detectas abolladuras, rayones o daños en la carrocería de este autobús?"*.
    *   El modelo analiza la imagen semánticamente y devuelve una descripción textual o un veredicto.

### Paso 5: Verificación de Daños (Custom YOLO)
*   Si el LVM marca el caso como "Sospechoso", se puede activar un segundo modelo YOLO especializado, entrenado exclusivamente con dataset de daños (golpes, vidrios rotos).
*   Esto confirma la ubicación exacta (Bounding Box) del daño para el reporte.

### Paso 6: Reporte y Almacenamiento
*   Los hallazgos se estructuran en un objeto `DamageReport`.
*   Se almacenan en base de datos y se envían alertas al dashboard de control.

---

## 4. Guía de Construcción y Despliegue

### Prerrequisitos
*   **Hardware**: PC/Servidor con GPU NVIDIA (RTX 3090/4090/5070 recomendado) + 16GB+ VRAM.
*   **Software**: Drivers NVIDIA actualizados, Docker Desktop, NVIDIA Container Toolkit.

### Configuración del Entorno
1.  **Clonar Repositorio**:
    ```bash
    git clone <repo_url>
    cd vision_cex_backend
    ```
2.  **Variables de Entorno**:
    Crear `.env` basado en `.env.example`:
    ```env
    YOLO_WEIGHTS=yolov8n-seg.pt  # O tu modelo custom v11
    HF_TOKEN=tu_token_huggingface
    ```

### Ejecución
El sistema NO se ejecuta localmente con Python directo, sino encapsulado:

```bash
# Construir y levantar servicios
docker compose up --build
```

*   La API estará disponible en `http://localhost:8000`.
*   Swagger UI: `http://localhost:8000/docs`.

### Uso de Modelos Específicos
*   **YOLO Local**: Coloca tus pesos `.pt` en la raíz y actualiza `YOLO_WEIGHTS` en el `.env`.
*   **ERNIE**: El adaptador `ErnieClient` descargará automáticamente el modelo cuantizado (AWQ) desde HuggingFace en el primer uso. **Nota**: Requiere ~16GB de VRAM.

---

## 5. Tecnologías Clave

| Tecnología | Uso | Justificación |
| :--- | :--- | :--- |
| **FastAPI** | Backend API | Alto rendimiento, asíncrono, tipado estático. |
| **Docker** | Contenedorización | Portabilidad y gestión de entorno GPU complejo. |
| **YOLO v11** | Detección/Segm. | Estado del arte en velocidad/precisión para video. |
| **AutoAWQ** | Inferencia LLM | Ejecución eficiente de modelos grandes (ERNIE) en 4-bit. |
| **OpenCV** | Procesamiento Imagen | Manipulación de frames, RTSP, HLS. |

---

## 6. Guía de Consumo de API

A continuación se detallan los endpoints principales para interactuar con el sistema.

### 6.1. Gestión de Fuentes de Video

**Endpoint**: `POST /api/v1/video/source/rtsp`
**Descripción**: Configura una cámara IP o stream RTSP como fuente de entrada.

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/video/source/rtsp' \
  -H 'Content-Type: application/json' \
  -d '{
  "url": "rtsp://admin:password@192.168.1.10:554/stream1"
}'
```

**Endpoint**: `POST /api/v1/video/source/file`
**Descripción**: Configura un archivo de video local (dentro del contenedor) como fuente.

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/video/source/file' \
  -H 'Content-Type: application/json' \
  -d '{
  "path": "samples/bus_inspection.mp4"
}'
```

### 6.2. Streaming con Detección en Tiempo Real

**Endpoint**: `GET /api/v1/video/stream`
**Descripción**: Devuelve un stream MJPEG continuo. Cada frame procesado incluye las "bounding boxes" dibujadas por YOLO.

*   **Parámetros**:
    *   `fps`: Limita los cuadros por segundo (útil para ahorrar ancho de banda).
    *   `roi`: Región de interés `x,y,w,h` (ej. `100,100,640,480`). Solo detecta dentro de ese recuadro.

```bash
# Abrir en navegador o VLC
http://localhost:8000/api/v1/video/stream?fps=15&roi=0,0,1280,720
```

### 6.3. Análisis Cognitivo con ERNIE (VLM)

**Endpoint**: `POST /api/v1/ernie/generate`
**Descripción**: Envía una imagen estática al modelo ERNIE-4.5-VL para obtener una descripción detallada o respuesta a una pregunta.

*   **Payload**:
    *   `image_url`: Puede ser una URL web, una ruta local del servidor, o una imagen en Base64 (`data:image/...`).
    *   `prompt`: La pregunta o instrucción para el modelo.

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/ernie/generate' \
  -H 'Content-Type: application/json' \
  -d '{
  "image_url": "https://ejemplo.com/foto_bus_dano.jpg",
  "prompt": "Describe detalladamente el daño visible en el parachoques delantero. ¿Es una abolladura o un rayón?"
}'
```

**Respuesta Ejemplo**:
```json
{
  "response": "La imagen muestra una abolladura significativa en el lado derecho del parachoques delantero. La pintura está desconchada y hay signos de óxido..."
}
```

---

## 7. Fragmentos de Código Relevantes

La robustez del sistema se basa en componentes clave bien aislados.

### 7.1. Cliente ERNIE (Soporte Multimodal)
Ubicación: `backend/app/data/adapters/ernie_client.py`

Este adaptador maneja la complejidad de cargar un modelo cuantizado (AWQ) y procesar diferentes tipos de entrada de imagen.

```python
class ErnieClient:
    def _load_image(self, image_source: str) -> Image.Image:
        """Carga imagen desde URL, Base64 o ruta local de forma transparente."""
        if image_source.startswith("http"):
            # Descarga desde web
            response = requests.get(image_source)
            return Image.open(BytesIO(response.content))
        elif image_source.startswith("data:image"):
            # Decodifica Base64
            header, encoded = image_source.split(",", 1)
            data = base64.b64decode(encoded)
            return Image.open(BytesIO(data))
        else:
            # Carga desde sistema de archivos local
            return Image.open(image_source)

    def generate(self, image_url: str, prompt: str) -> str:
        # ... lógica de inferencia con transformers y autoawq ...
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        return self.tokenizer.decode(outputs[0])
```

### 7.2. Adaptador YOLO Ultralytics
Ubicación: `backend/app/data/adapters/yolo_ultralytics_adapter.py`

Encapsula la librería `ultralytics` para que el dominio no dependa de ella directamente. Soporta tanto detección como segmentación.

```python
class YoloUltralyticsAdapter:
    def __init__(self, weights_path, device="auto"):
        self._model = YOLO(weights_path)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def segment(self, frame) -> List[SegmentationObject]:
        # Ejecuta inferencia optimizada
        results = self._model.predict(source=frame, device=self._device, verbose=False)
        # Mapea resultados propietarios de YOLO a entidades de dominio limpias
        instances = []
        for r in results:
            # ... extracción de máscaras y polígonos ...
            instances.append(SegmentationObject(polygon=pts, cls=cls_name))
        return instances
```

### 7.3. Inyección de Dependencias (Service Locator)
Ubicación: `backend/app/core/di/service_locator.py`

Centraliza la creación de instancias, permitiendo cambiar implementaciones (ej. de YOLO a otro detector) sin tocar el código de los controladores.

```python
class ServiceLocator:
    @classmethod
    def ernie_repo(cls) -> ErnieRepositoryImpl:
        if cls._ernie_repo is None:
            # Inyecta el cliente en el repositorio
            cls._ernie_repo = ErnieRepositoryImpl(client=cls.ernie_client())
        return cls._ernie_repo
```

