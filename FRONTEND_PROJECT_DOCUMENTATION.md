# Documentación del Proyecto Vision CEX Frontend

## 1. Visión General del Proyecto (Stakeholders)

**Vision CEX** es una plataforma de inteligencia artificial de vanguardia diseñada para la inspección y detección automatizada de daños en la flota vehicular de Consorcio Express. Utilizando tecnologías avanzadas de visión por computador, la plataforma permite un monitoreo en tiempo real y un análisis detallado del estado de los vehículos.

### Propuesta de Valor
- **Automatización**: Reduce la necesidad de inspecciones manuales, ahorrando tiempo y recursos.
- **Monitoreo en Tiempo Real**: Visualización en vivo de las cámaras de inspección (Live Views).
- **Trazabilidad**: Registro histórico de grabaciones y reportes detallados de daños.
- **Escalabilidad**: Arquitectura modular diseñada para crecer con nuevas funcionalidades.

### Módulos Principales
1.  **Live Views (Vistas en Vivo)**: Panel de control para visualizar múltiples streams de video en tiempo real desde los arcos de inspección.
2.  **Grabaciones**: Acceso a un archivo histórico de inspecciones pasadas para auditoría y revisión.
3.  **Reportes**: Generación de informes detallados sobre los daños detectados, incluyendo estadísticas y evidencia visual.

---

## 2. Arquitectura Técnica

Este proyecto está construido sobre **Flutter**, utilizando **Clean Architecture** para garantizar la mantenibilidad, testabilidad y escalabilidad del código.

### Principios de Diseño
- **Separación de Responsabilidades**: El código se divide en capas (Presentación, Dominio, Datos) para que la lógica de negocio sea independiente de la UI y de las fuentes de datos.
- **Modularidad**: Cada funcionalidad principal (`auth`, `liveviews`, `recordings`, `reports`) reside en su propio módulo dentro de `lib/features`.
- **Inyección de Dependencias**: Uso de `get_it` para gestionar las dependencias de forma desacoplada.

### Estructura de Carpetas
```
lib/
├── core/                   # Componentes compartidos y configuración base
│   ├── constants/          # Strings, assets, configuraciones globales
│   ├── di/                 # Inyección de dependencias (Service Locator)
│   ├── error/              # Manejo de errores y excepciones
│   ├── navigation/         # Configuración de rutas (GoRouter)
│   ├── network/            # Clientes HTTP/WebSocket
│   └── usecases/           # Interfaz base para casos de uso
├── features/               # Módulos funcionales
│   ├── auth/               # Autenticación (Login, Sesión)
│   ├── home/               # Pantalla principal y dashboard
│   ├── liveviews/          # Visualización de cámaras (WebRTC/MJPEG)
│   ├── recordings/         # Historial de grabaciones
│   └── reports/            # Reportes de daños
└── main.dart               # Punto de entrada de la aplicación
```

---

## 3. Detalles de Implementación y Código

A continuación, profundizamos en cómo se implementan los patrones clave con ejemplos reales del código.

### 3.1. Inyección de Dependencias (`lib/core/di/injection_container.dart`)
Utilizamos `get_it` como Service Locator. Este archivo es el "cerebro" que conecta todas las piezas de la arquitectura.

```dart
final sl = GetIt.instance;

Future<void> init() async {
  // ! Features - Auth
  
  // 1. Bloc (Presentación)
  // Se usa registerFactory para crear una NUEVA instancia cada vez que se solicita.
  // Esto es crucial para los Blocs, ya que pueden tener estados efímeros que queremos reiniciar.
  sl.registerFactory(
    () => AuthBloc(
      loginUseCase: sl(), // Inyectamos los casos de uso necesarios
      logoutUseCase: sl(),
      getCurrentAuthUseCase: sl(),
      checkAuthStatusUseCase: sl(),
    ),
  );

  // 2. Use Cases (Dominio)
  // Se usa registerLazySingleton para crear una ÚNICA instancia compartida.
  // Los casos de uso suelen ser stateless, por lo que no necesitamos múltiples copias.
  sl.registerLazySingleton(() => LoginUseCase(sl()));
  sl.registerLazySingleton(() => LogoutUseCase(sl()));

  // 3. Repository (Datos)
  // Registramos la INTERFAZ (AuthRepository) pero devolvemos la IMPLEMENTACIÓN (AuthRepositoryImpl).
  // Esto permite cambiar la implementación (ej. de Firebase a REST API) sin tocar el dominio.
  sl.registerLazySingleton<AuthRepository>(
    () => AuthRepositoryImpl(
      remoteDataSource: sl(),
      localDataSource: sl(),
    ),
  );
}
```

### 3.2. Navegación (`lib/core/navigation/app_router.dart`)
Usamos `go_router` para una gestión de rutas declarativa y robusta.

```dart
static final GoRouter router = GoRouter(
  initialLocation: splash,
  routes: [
    GoRoute(
      path: login,
      name: 'login',
      builder: (context, state) => const LoginScreen(),
    ),
    GoRoute(
      path: home,
      name: 'home',
      // Aquí podríamos añadir sub-rutas o parámetros
      builder: (context, state) => const home_feature.HomeScreen(),
    ),
  ],
  // Manejo global de errores 404
  errorBuilder: (context, state) => const NotFoundScreen(),
);
```

### 3.3. Clean Architecture: Ejemplo Módulo Auth

#### Capa de Dominio (Lógica de Negocio Pura)
Aquí definimos *qué* hace la app, sin importar *cómo* se guardan los datos o *cómo* se ven.

**Caso de Uso (`LoginUseCase`)**:
Encapsula una regla de negocio específica. En este caso, validar credenciales antes de llamar al repositorio.

```dart
// lib/features/auth/domain/usecases/login_usecase.dart
class LoginUseCase {
  final AuthRepository repository;

  const LoginUseCase(this.repository);

  // El método 'call' permite usar la instancia como una función: loginUseCase(...)
  Future<AuthEntity> call({
    required String email,
    required String password,
  }) async {
    // Reglas de negocio: Validación previa
    if (!_isValidEmail(email)) {
      throw Exception('Formato de email inválido');
    }
    if (password.length < 6) {
      throw Exception('La contraseña debe tener al menos 6 caracteres');
    }

    // Delegar la operación de I/O al repositorio
    final auth = await repository.login(
      email: email.trim().toLowerCase(),
      password: password,
    );
    
    await repository.saveAuth(auth);
    return auth;
  }
}
```

#### Capa de Datos (Implementación e Infraestructura)
Aquí definimos *cómo* se obtienen los datos.

**Modelo (`UserModel`)**:
Es la representación de datos que viene de la API (JSON). Extiende de la Entidad de dominio para ser compatible.

```dart
// lib/features/auth/data/models/user_model.dart
class UserModel extends UserEntity {
  const UserModel({
    required super.id,
    required super.email,
    // ...
  });

  // Convierte JSON a Objeto Dart
  factory UserModel.fromJson(Map<String, dynamic> json) {
    return UserModel(
      id: json['id'] as String,
      email: json['email'] as String,
      // Parseo seguro de fechas
      createdAt: DateTime.parse(json['created_at'] as String),
      // Manejo de nulos
      isActive: json['is_active'] as bool? ?? true,
    );
  }

  // Convierte Objeto Dart a JSON (para enviar a API)
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'email': email,
      // ...
    };
  }
}
```

#### Capa de Presentación (UI y Estado)
Aquí definimos *cómo* se ve y se comporta la app.

**BLoC (`AuthBloc`)**:
Gestiona el estado de la UI basándose en eventos.

```dart
// lib/features/auth/presentation/bloc/auth_bloc.dart
class AuthBloc extends Bloc<AuthEvent, AuthState> {
  final LoginUseCase loginUseCase;
  // ... otras dependencias

  AuthBloc({required this.loginUseCase, ...}) : super(const AuthInitial()) {
    on<LoginEvent>(_onLogin);
  }

  Future<void> _onLogin(LoginEvent event, Emitter<AuthState> emit) async {
    try {
      // 1. Emitir estado de carga (UI muestra spinner)
      emit(const AuthLoading());

      // 2. Ejecutar caso de uso
      final auth = await loginUseCase(
        email: event.email,
        password: event.password,
      );

      // 3. Emitir éxito (UI navega a Home)
      emit(AuthAuthenticated(auth: auth, user: auth.user));
    } catch (e) {
      // 4. Emitir error (UI muestra snackbar)
      // Aquí mapeamos excepciones técnicas a mensajes amigables
      String errorMessage = 'Error al iniciar sesión';
      if (e.toString().contains('Usuario no encontrado')) {
        errorMessage = 'Usuario no encontrado';
      }
      emit(AuthError(message: errorMessage));
    }
  }
}
```

---

## 4. Stack Tecnológico

- **Framework**: Flutter (Dart)
- **Gestión de Estado**: `flutter_bloc` (Patrón BLoC)
- **Navegación**: `go_router` (Ruteo declarativo)
- **Inyección de Dependencias**: `get_it`
- **Comparación de Objetos**: `equatable`
- **Almacenamiento Local**: `shared_preferences`
- **Streaming en Tiempo Real**:
    - `flutter_webrtc`: Para conexiones de baja latencia y alta calidad (Arquitectura Objetivo).
    - Soporte para streams MJPEG/HLS para compatibilidad.
- **Internacionalización (i18n)**: `flutter_localizations` (Soporte multi-idioma).

---

## 5. Guía de Construcción Paso a Paso

Esta sección detalla cómo se construye la aplicación desde cero.

### Paso 1: Configuración Inicial
Creación del proyecto e instalación de dependencias clave en `pubspec.yaml`.

### Paso 2: Capa Core
Establecemos los cimientos de la aplicación (`injection_container.dart`, `app_router.dart`).

### Paso 3: Módulo de Autenticación (`features/auth`)
Implementación del flujo de Login siguiendo el flujo: Entidad -> Repositorio -> Caso de Uso -> Bloc -> UI.

### Paso 4: Módulo de Live Views (`features/liveviews`)
Implementación de la visualización de cámaras.
- **Infraestructura de Video**: Configuración de `flutter_webrtc` para recibir streams.
- **Adaptación Web**: Uso de `HtmlElementView` o `Image.network` (MJPEG) como fallback.

### Paso 5: Internacionalización
Configuración de `l10n` para soportar español e inglés.

### Paso 6: Módulos de Grabaciones y Reportes
Siguiendo el mismo patrón que Auth:
- **Recordings**: Listado de videos históricos.
- **Reports**: Vistas detalladas de incidentes.
