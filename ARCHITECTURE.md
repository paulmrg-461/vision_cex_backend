# ARCHITECTURE.md

## **Backend Application Architecture with FastAPI**

This document outlines the **inquebrantable rules** for structuring and developing backend applications using **FastAPI**. It enforces **Clean Architecture**, modular design, and best practices to ensure scalability, maintainability, and testability. All projects must adhere to this architecture, and deviations are not allowed.

---

IMPORTANTE: Política de ejecución
- El backend NO debe ejecutarse directamente en la máquina local con `pip`/`uvicorn`.
- TODA la ejecución y desarrollo deben realizarse mediante **Docker** y **docker-compose** con soporte para GPU (NVIDIA Container Toolkit).
- Las dependencias se instalan dentro del contenedor; el host solo necesita Docker, Docker Compose y (opcional) NVIDIA Toolkit para GPU.

---

### **1. Project Structure**

The project follows a **modular directory structure** based on **Clean Architecture**. Each module (feature) is self-contained and adheres to the following layers:

```
backend/
├── app/                  # Core application logic
│   ├── domain/           # Business logic and use cases
│   ├── data/             # Data layer implementation
│   ├── presentation/     # API endpoints and request/response models
│   └── core/             # Core utilities and dependency injection
├── docker/               # Docker-related files
│   ├── Dockerfile        # Dockerfile for the service
│   ├── docker-compose.yml# Orchestration of multiple services
│   └── nvidia/           # NVIDIA-specific configurations
├── .env                  # Environment variables for credentials
├── requirements.txt      # Python dependencies
└── README.md             # Documentation for the project
```

---

### **2. Layers in Detail**

#### **Domain Layer**
- **Purpose:** Contains the business logic and abstract definitions of repositories and use cases.
- **Components:**
  - **Entities:** Plain Python objects representing core business models.
  - **Repositories:** Abstract interfaces defining contracts for data access.
  - **Use Cases:** Classes implementing specific business logic (e.g., `GetUserUseCase`, `ProcessImageUseCase`).

**Example Directory Structure:**
```
domain/
├── entities/
│   └── user_entity.py
├── repositories/
│   └── user_repository.py
└── usecases/
    └── get_user_usecase.py
```

---

#### **Data Layer**
- **Purpose:** Implements the data access logic and provides concrete implementations of repositories.
- **Components:**
  - **Repositories Implementation (`Impl`):** Concrete classes implementing the repository interfaces.
  - **Models:** Data models used for serialization/deserialization.
  - **Data Sources:** Interfaces and implementations for local and remote data sources (e.g., database queries, external APIs).
  - **Data Sources Implementation (`Impl`):** Concrete implementations of data sources.

**Example Directory Structure:**
```
data/
├── datasources/
│   ├── user_datasource.py
│   └── user_datasource_impl.py
├── models/
│   └── user_model.py
├── repositories/
│   └── user_repository_impl.py
└── ...
```

---

#### **Presentation Layer**
- **Purpose:** Handles the API endpoints and request/response models.
- **Components:**
  - **API Endpoints:** FastAPI routes for handling HTTP requests.
  - **Request/Response Models:** Pydantic models for validating input/output data.
  - **Middleware:** Custom middleware for logging, authentication, etc.

**Example Directory Structure:**
```
presentation/
├── api/
│   ├── v1/
│   │   ├── user_router.py
│   │   └── healthcheck_router.py
├── models/
│   └── user_request_model.py
└── middleware/
    └── auth_middleware.py
```

---

#### **Core Layer**
- **Purpose:** Centralized utilities, configurations, and dependency injection.
- **Components:**
  - **Dependency Injection:** Use `fastapi.DependencyInjection` or `dependency_injector` for managing dependencies.
  - **Utilities:** Helper functions, constants, and extensions.
  - **Configurations:** Environment-specific configurations (e.g., API endpoints, app settings).

**Example Directory Structure:**
```
core/
├── di/
│   └── service_locator.py
├── utils/
│   └── constants.py
├── config/
│   └── environment_config.py
└── ...
```

---

### **3. Docker and GPU Support**

#### **Docker Configuration**
- **Purpose:** Ensure the application runs in a containerized environment and supports GPU acceleration.
- **Components:**
  - **Dockerfile:** Build the application image with CUDA and cuDNN support.
  - **docker-compose.yml:** Orchestrate multiple services (e.g., FastAPI, database, cache).

**Key Rules:**
- Use the **NVIDIA Container Toolkit** to enable GPU support in Docker.
- Specify the base image with CUDA and cuDNN pre-installed (e.g., `nvidia/cuda:12.0-base`).
- Siempre iniciar el servicio vía `docker compose`. No se debe usar `uvicorn` directamente en el host.

**Example Dockerfile:**
```dockerfile
FROM nvidia/cuda:12.0-base

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app/
WORKDIR /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port and run the app
EXPOSE 8000
CMD ["uvicorn", "app.presentation.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Example docker-compose.yml:**
```yaml
version: '3.8'

services:
  fastapi:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    env_file:
      - .env
    volumes:
      - .:/app
  redis:
    image: redis:latest
    ports:
      - "6379:6379"
```

---

### **4. Environment Variables**

- Store all sensitive credentials and configuration in a `.env` file.
- Use the `.env` file to configure API keys for HuggingFace and Deepseek.

**Example .env File:**
```
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
REDIS_HOST=redis
REDIS_PORT=6379
APP_ENV=development
```

---

### **5. Testing**

- Follow **Test-Driven Development (TDD)** principles.
- Write unit tests, integration tests, and end-to-end tests for all layers.
- Use the following tools:
  - **Unit Tests:** Test business logic and use cases.
  - **Integration Tests:** Test API endpoints and interactions with external services.
  - **End-to-End Tests:** Test the entire application flow.

**Key Rules:**
- Place tests in the `tests/` directory, mirroring the `app/` structure.
- Ensure all tests pass before merging code into the main branch.

**Example Directory Structure:**
```
tests/
├── domain/
│   └── usecases/
│       └── get_user_usecase_test.py
├── data/
│   └── repositories/
│       └── user_repository_impl_test.py
├── presentation/
│   └── api/
│       └── user_router_test.py
└── ...
```

---

### **6. Additional Guidelines**

- **Code Quality:**  
  - Follow **Clean Code** principles.
  - Write self-explanatory, readable, and maintainable code.
  - Avoid unnecessary complexity or redundancy.

- **Documentation:**  
  - Document all modules, components, and their interactions in the `README.md` file.
  - Update this `ARCHITECTURE.md` file if new patterns, technologies, or architectural decisions are introduced.

- **Version Control:**  
  - Use Git for version control and follow branching strategies like Git Flow or Trunk-Based Development.
  - Ensure commit messages are descriptive and follow conventional commit standards (e.g., `feat:`, `fix:`, `docs:`).

---

### **7. Ejecución con Docker y docker-compose (obligatoria)**

#### Prerrequisitos (host)
- Docker Desktop (o Docker Engine) y Docker Compose.
- NVIDIA GPU y drivers con **NVIDIA Container Toolkit** para habilitar CUDA dentro del contenedor. En Windows, asegúrate de tener WSL2 habilitado y soporte de GPU en Docker Desktop.

#### Variables de entorno
- Copia `.env.example` a `.env` y ajusta según tu entorno:
```
APP_ENV=development
VIDEO_SOURCE=0
YOLO_WEIGHTS=yolov8n.pt
DEVICE=auto
```
- `.env` se inyectará al servicio FastAPI vía `docker-compose.yml`.

#### Construcción y arranque
- Construir e iniciar el servicio con GPU (si disponible):
```
docker compose up --build
```
- El contenedor instala las dependencias definidas en `requirements.txt` y ejecuta:
```
uvicorn app.presentation.api.main:app --host 0.0.0.0 --port 8000
```

#### Acceso a la API
- Health/root: `http://localhost:8000/`
- Streaming MJPEG con detección: `http://localhost:8000/api/v1/video/stream`
- ROI opcional para limitar la detección a una región: `http://localhost:8000/api/v1/video/stream?roi=x,y,w,h`

#### Notas importantes
- No ejecutar `pip install` ni `uvicorn` en el host; todo se ejecuta dentro del contenedor.
- Si no se detecta GPU, Ultralytics usará CPU automáticamente; el streaming seguirá funcionando pero con menor rendimiento.
- Si se desea usar un modelo distinto (p. ej., ONNX con OpenCV DNN), actualizar `YOLO_WEIGHTS` y el adaptador correspondiente en la capa `data/adapters/`.
