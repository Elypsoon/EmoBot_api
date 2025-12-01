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
├── Models/                        # 🧠 Modelos entrenados
│   ├── RoBERTa_Emociones_Model.pth    # Pesos del modelo de emociones
│   ├── evaluator_model.pkl            # Modelo evaluador (LogReg)
│   └── scaler.pkl                     # Escalador de características
│
├── RoBERTa_local/                 # 🤗 Modelo RoBERTa base local
│   ├── config.json                # Configuración del transformer
│   ├── pytorch_model.bin          # Pesos base del modelo
│   ├── tokenizer.json             # Tokenizer entrenado
│   ├── vocab.json                 # Vocabulario
│   ├── merges.txt                 # BPE merges
│   ├── special_tokens_map.json    # Tokens especiales
│   └── tokenizer_config.json      # Config del tokenizer
│
├── .env                           # 🔐 Variables de entorno (API keys)
└── README.md                      # 📖 Este archivo
```

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
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers numpy scikit-learn joblib google-genai
```

4. **Configurar variables de entorno**
```bash
# Crear archivo .env en la raíz del proyecto
echo "GEMINI_API_KEY=tu_api_key_aqui" > .env
```

5. **Verificar estructura de modelos**
Asegúrate de que las carpetas `Models/` y `RoBERTa_local/` contengan todos los archivos necesarios.

---

## 💻 Uso

### Ejecución Básica

```bash
cd API
python main.py
```

### Comandos Disponibles en el Chat

- 📝 **Escribe tu mensaje**: Simplemente escribe y presiona Enter
- ❌ `/quit` o `/exit`: Salir del chat
- 📊 `/summary`: Ver resumen estadístico de la sesión actual

### Ejemplo de Sesión

```
=== Chat EmoBot (Transformer + Evaluador + LLM) ===

[INFO] Usando dispositivo: cuda
[INFO] Cargando tokenizer y modelo emocional desde local...
[OK ] Modelo emocional cargado.
[INFO] Cargando evaluador (LogReg) y scaler...
[OK ] Scaler cargado.
[OK ] LLM (Gemini) inicializado.

Escribe tu mensaje y presiona Enter.
Comandos: /quit para salir, /summary para ver resumen de la sesión.

Tú: Me siento muy ansioso últimamente
[ANÁLISIS] emocionesTop: [('miedo', 0.85), ('tristeza', 0.62), ('ira', 0.23)]
[RIESGO] none=0.12 | anxiety=0.78 | depression=0.10 => pred: anxiety
Asistente: Entiendo que la ansiedad puede ser muy abrumadora. ¿Hay algo en particular que sientas que está desencadenando esa ansiedad? ¿Cómo has estado manejando estos sentimientos?
```

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

**Propósito**: Aplicación principal y orquestador del sistema

**Funcionalidades**:
- Inicialización de todos los componentes
- Bucle de conversación interactivo
- Análisis en tiempo real
- Gestión de historial de sesión
- Comandos especiales del usuario

**Flujo de ejecución**:

```python
1. Cargar EmotionAnalyzer
2. Cargar RiskEvaluator
3. Inicializar GeminiAdapter
4. Loop:
   a. Recibir input del usuario
   b. Analizar emociones
   c. Evaluar riesgo
   d. Generar respuesta con LLM
   e. Mostrar resultados
   f. Guardar en historial
```

**Salida en consola**:
- 📊 Análisis de emociones (top 3)
- 🎯 Probabilidades de riesgo
- 🏷️ Predicción de clase
- 💬 Respuesta del asistente

---

## 🧠 Modelos

### RoBERTa de Emociones

- **Archivo**: `Models/RoBERTa_Emociones_Model.pth`
- **Base**: RoBERTa pre-entrenado (transformers)
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

- **Directorio**: `RoBERTa_local/`
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