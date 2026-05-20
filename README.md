# 🤖 EmoBot API

> Sistema de chat conversacional inteligente con análisis de emociones en tiempo real y evaluación de riesgo psicológico (ansiedad/depresión) usando Deep Learning y NLP.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características](#-características)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Módulos de la API](#-módulos-de-la-api)
- [Modelos](#-modelos)
- [Configuración](#-configuración)
---

## 🎯 Descripción

EmoBot es un sistema de asistencia conversacional empático que:
- **Detecta emociones** en texto usando un modelo RoBERTa fine-tuned
- **Evalúa riesgos** de ansiedad y depresión mediante aprendizaje automático
- **Genera respuestas** empáticas y contextuales usando Gemini LLM
- **Orienta al usuario** sin diagnosticar, respetando límites éticos y clínicos

## ✨ Características

- 🧠 **Análisis Emocional Multi-etiqueta**: Detecta 6 emociones simultáneamente (disgusto, sorpresa, ira, tristeza, felicidad, miedo)
- 🎯 **Predicción de Riesgo**: Clasifica entre ninguno, ansiedad o depresión usando características lingüísticas y emocionales
- 💬 **Respuestas Inteligentes**: Integración con Gemini LLM para conversaciones naturales y empáticas
- ⚡ **Tiempo Real**: Análisis instantáneo de cada mensaje del usuario
- 🔒 **Ética y Seguridad**: Instrucciones cuidadosas para evitar diagnósticos y ofrecer recursos apropiados
- 📊 **Historial de Sesión**: Seguimiento completo de la conversación y evolución del usuario

---

## 📁 Estructura del Proyecto

```
EmoBot_api/
│
├── API/                           # 🐍 Código fuente de la aplicación
│   ├── config.py                  # Configuración central del sistema
│   ├── emotion_analyzer.py        # Análisis de emociones con RoBERTa
│   ├── risk_evaluator.py          # Evaluación de riesgo (ansiedad/depresión)
│   ├── llm_adapter.py             # Integración con Gemini LLM
│   ├── main.py                    # Aplicación principal de chat
│   └── __pycache__/               # Archivos compilados de Python
│
├── Models/                        # 🧠 Modelos y recursos guardados
│   ├── best_hyperparameters.json      # Hiperparámetros óptimos del modelo
│   ├── evaluator_model.pkl            # Modelo evaluador (LogReg)
│   └── scaler.pkl                     # Escalador de características
│
├── .env                           # 🔐 Variables de entorno (API keys y paths)
└── README.md                      # 📖 Este archivo
```

> **Nota:** El modelo RoBERTa con *fine-tuning* utilizado para el análisis de emociones no se incluye directamente en el repositorio por su tamaño. Su ubicación se define mediante la variable `MODEL_DIR` en el archivo `.env`.

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                         USUARIO INPUT                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │   main.py      │  ← Orquestador principal
                    └────────┬───────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
                ▼            ▼            ▼
    ┌──────────────┐  ┌──────────┐  ┌──────────────┐
    │ Emotion      │  │ Risk     │  │ LLM          │
    │ Analyzer     │  │ Evaluator│  │ Adapter      │
    └──────┬───────┘  └────┬─────┘  └──────┬───────┘
           │               │                │
           ▼               ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ RoBERTa  │    │ LogReg + │    │ Gemini   │
    │ Model    │    │ Features │    │ API      │
    └──────────┘    └──────────┘    └──────────┘
           │               │                │
           └───────────────┴────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  RESPUESTA EMPÁTICA    │
              │  + Análisis de Riesgo  │
              └────────────────────────┘
```

---

## 🚀 Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- GPU con CUDA (opcional, pero recomendado para mejor rendimiento)
- Tener localmente los Models y el transformer RoBERTa 

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone <repository-url>
cd EmoBot_api
```

2. **Crear entorno virtual** (opcional, pero recomendado)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Instalar dependencias**

Es recomendable instalar PyTorch primero con soporte CUDA si tienes una GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Luego instala el resto de las dependencias usando el archivo `requirements.txt`:
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**
```bash
# Crear archivo .env en la raíz del proyecto y agregar las claves
echo "GEMINI_API_KEY=tu_api_key_aqui" > .env
echo "MODEL_DIR=../RoBERTa_local" >> .env
```

5. **Verificar estructura de modelos**
Asegúrate de que las carpetas `Models/` y `RoBERTa_local/` contengan todos los archivos necesarios.

---

## 💻 Uso

### Iniciar el Servidor API

El proyecto está diseñado como una API REST usando FastAPI. Para iniciar el servidor:

```bash
cd API
uvicorn main:app --reload
```

El servidor estará disponible en `http://127.0.0.1:8000`. Puedes acceder a la documentación interactiva (Swagger UI) en `http://127.0.0.1:8000/docs`.

### Ejemplo de Petición al Chat

Puedes probar la API haciendo una solicitud POST al endpoint `/chat`:

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "Me siento muy ansioso últimamente"}'
```

**Respuesta Esperada:**
```json
{
  "response": "Entiendo que la ansiedad puede ser muy abrumadora. ¿Hay algo en particular que sientas que está desencadenando esa ansiedad? ¿Cómo has estado manejando estos sentimientos?",
  "session_id": "893c5fa41a94..."
}
```

Puedes usar el `session_id` devuelto en peticiones futuras a `/chat` incluyéndolo en el cuerpo del JSON para mantener el contexto de tu conversación.

---

## 📦 Módulos de la API

### 1. 🔧 `config.py`

**Propósito**: Configuración centralizada del sistema

**Contenido**:
- 📂 Rutas a modelos y archivos
- 🎭 Nombres de emociones y umbrales de detección
- 🏷️ Mapeo de clases del evaluador
- 💻 Configuración de dispositivo (CPU/GPU)
- ⏱️ Rate limits para APIs externas
- 📜 Instrucciones del sistema para el LLM

---

### 2. 🎭 `emotion_analyzer.py`

**Propósito**: Análisis de emociones usando RoBERTa fine-tuned

**Clase principal**: `EmotionAnalyzer`

**Funcionalidades**:
- Carga del modelo RoBERTa local desde checkpoint
- Tokenización de texto con manejo de longitud máxima
- Predicción multi-etiqueta de 6 emociones
- Retorna vectores de probabilidades normalizados



**Tecnologías**:
- 🤗 Transformers (Hugging Face)
- 🔥 PyTorch
- 📊 NumPy

---

### 3. ⚠️ `risk_evaluator.py`

**Propósito**: Evaluación de riesgo psicológico (ansiedad/depresión)

**Clase principal**: `RiskEvaluator`

**Funcionalidades**:
- Extracción de 16 características del texto
- Aplicación de reglas heurísticas (conteo de signos, patrones)
- Predicción con modelo de regresión logística
- Escalado automático de características

**Pipeline de características**:
1. **Emocionales**: Probabilidades de las 6 emociones
2. **Flags**: Banderas binarias si emoción > umbral
3. **Heurísticas**: Conteo de "!", "?", patrones lingüísticos
4. **Scores normalizados**: Puntajes compuestos para ansiedad/depresión

**Clases de predicción**:
- 🟢 **none**: Sin señales significativas de riesgo
- 🟡 **anxiety**: Indicadores de ansiedad
- 🔴 **depression**: Indicadores de depresión

---

### 4. 🤖 `llm_adapter.py`

**Propósito**: Integración con Gemini LLM para respuestas empáticas

**Clase principal**: `GeminiAdapter`

**Funcionalidades**:
- Conexión con API de Google Gemini
- Creación de sesiones de chat con contexto
- Generación de respuestas contextuales y empáticas
- Control automático de rate limiting (14 requests/min)

**Características de seguridad**:
- ❌ No diagnostica ni etiqueta
- ✅ Máximo 2 preguntas por turno
- 🆘 Ofrece recursos en casos críticos
- 🛡️ Respeta límites éticos y clínicos


**Instrucciones del sistema**: Ver `config.py` para el prompt completo que guía el comportamiento del LLM.

---

### 5. 🎬 `main.py`

**Propósito**: Exponer los servicios del proyecto mediante una API REST

**Funcionalidades**:
- Inicialización de todos los componentes a través del ciclo de vida (`lifespan`) de FastAPI
- Endpoints para evaluar riesgos y conectar con el modelo Gemini
- Análisis en tiempo real a través del endpoint `/chat`
- Gestión de la sesión de conversación en memoria

**Flujo de una petición HTTP al chat**:

```python
1. Se recibe un mensaje mediante POST en /chat con un session_id (o se crea uno temporal)
2. Se evalúan las emociones y el riesgo del mensaje utilizando RiskEvaluator
3. Se envian los análisis extraídos y el texto del usuario a GeminiAdapter
4. FastAPI registra la entrada, el análisis y la salida en el historial de sesiones en memoria
5. Se devuelve al cliente la respuesta del LLM y el session_id
```

**Endpoints principales**:
- `GET /`: Mensaje de bienvenida y status básico de la API.
- `GET /health`: Estado de los componentes (modelos e IA local).
- `POST /chat`: Interfaz principal para interactuar enviando mensajes de chat.
- `DELETE /session/{session_id}`: Limpia y elimina el contexto de una sesión en memoria.

---

## 🧠 Modelos

### RoBERTa de Emociones (Fine-Tuned)

- **Directorio**: Definido por la variable `MODEL_DIR` en el `.env` (ruta externa al proyecto).
- **Base**: Modelo RoBERTa pre-entrenado adaptado (fine-tuning) específicamente para detección de emociones en texto.
- **Tarea**: Clasificación multi-etiqueta
- **Clases**: 6 emociones (disgusto, sorpresa, ira, tristeza, felicidad, miedo)
- **Función de activación**: Sigmoid
- **Métricas**: F1-score multi-label

### Evaluador de Riesgo

- **Archivo**: `Models/evaluator_model.pkl`
- **Algoritmo**: Regresión Logística (LogisticRegression)
- **Features**: 18+ características emocionales y lingüísticas
- **Scaler**: `Models/scaler.pkl` (StandardScaler)
- **Clases**: 3 (none, anxiety, depression)

### Tokenizer y Configuración

- **Directorio**: Definido por `MODEL_DIR` en el `.env` (comparte ruta con el modelo fine-tuned).
- **Tipo**: BPE (Byte-Pair Encoding)
- **Vocabulario**: ~50k tokens
- **Tokens especiales**: `<s>`, `</s>`, `<pad>`, `<unk>`, `<mask>`

---

## ⚙️ Configuración

### Variables de Entorno

Crear archivo `.env` en la raíz del proyecto:

```bash
# API Keys
GEMINI_API_KEY=tu_clave_api_aqui

# Opcional: Configuración adicional
DEVICE=cuda  # o 'cpu'
MAX_LENGTH=256
RATE_LIMIT=14
```

### Ajustar Umbrales

En `config.py`, puedes modificar los umbrales de detección:

```python
THRESHOLDS = {
    "disgusto": 0.7,    # Aumentar para menos sensibilidad
    "sorpresa": 0.7,
    "ira": 0.7,
    "tristeza": 0.7,
    "felicidad": 0.7,
    "miedo": 0.8,       # Mayor umbral para emociones críticas
}
```

---