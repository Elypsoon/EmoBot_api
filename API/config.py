"""
Configuración central del sistema EmoBot.
Contiene rutas, constantes, umbrales, listas léxicas y configuraciones del sistema.
"""
import os
import re
from dotenv import load_dotenv
import torch
load_dotenv()

# === Rutas de modelos ===
# Ruta del modelo RoBERTa local (incluye tokenizer y pesos)
LOCAL_MODEL_DIR = os.getenv("MODEL_DIR")
EVALUATOR_MODEL_PKL = "../Models/evaluator_model.pkl"
SCALER_PKL = "../Models/scaler.pkl"

# === Configuración de emociones ===
EMOTION_NAMES = ["disgusto", "sorpresa", "ira", "tristeza", "felicidad", "miedo"]
NUM_EMOTIONS = len(EMOTION_NAMES)

# === Mapeo de clases del evaluador ===
LABEL_MAP = {"none": 0, "anxiety": 1, "depression": 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# === Configuración de dispositivo ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Diccionarios léxicos para extracción de características ===
# Mapeo de abreviaciones comunes en español
ABREVIACIONES_COMUNES = {
    "k": "que", "q": "que", "xq": "porque", "xk": "porque", "pq": "porque",
    "kiero": "quiero", "tmb": "también", "d": "de", "xa": "para", "pa": "para",
    "tb": "también", "x": "por", "bn": "bien", "profe": "profesor",
    "ntp": "no te preocupes", "qndo": "cuando"
}

# Pronombres de primera persona (indicadores de autoenfoque)
PRONOMBRES_1RA = {
    "yo", "me", "mi", "mí", "conmigo", "mío", "mía", "míos", "mías"
}

# Palabras de negación
NEGACIONES = {
    "no", "nunca", "jamás", "tampoco", "nadie", "nada", 
    "ningún", "ninguno", "ninguna"
}

# Palabras absolutistas (indicadores de pensamiento dicotómico)
ABSOLUTISTAS = {
    "siempre", "nunca", "jamás", "todo", "toda", "todos", "todas", 
    "nada", "nadie", "totalmente", "absolutamente", "completamente", 
    "definitivamente", "eternamente"
}

# Preposiciones
PREPOSICIONES = {
    "a", "ante", "bajo", "cabe", "con", "contra", "de", "desde", 
    "durante", "en", "entre", "hacia", "hasta", "mediante", "para", 
    "por", "según", "sin", "so", "sobre", "tras", "versus", "vía"
}

# Conjunciones
CONJUNCIONES = {
    "y", "e", "ni", "que", "o", "u", "pero", "mas", "sino", 
    "aunque", "porque", "pues", "si", "como"
}

# Determinantes
DETERMINANTES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", 
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas", 
    "aquel", "aquella", "aquellos", "aquellas", "mi", "tu", "su", 
    "mis", "tus", "sus", "nuestro", "nuestra", "nuestros", "nuestras"
}


def limpiar_texto(texto: str) -> str:
    """
    Limpia y normaliza el texto expandiendo abreviaciones y eliminando 
    espacios múltiples.
    
    Args:
        texto: Texto a limpiar
        
    Returns:
        Texto normalizado en minúsculas
    """
    if not isinstance(texto, str):
        return ""
    texto = texto.lower().strip()
    tokens = texto.split()
    tokens_normalizados = [ABREVIACIONES_COMUNES.get(tok, tok) for tok in tokens]
    texto = " ".join(tokens_normalizados)
    texto = re.sub(r"\s+", " ", texto)
    return texto

# === Rate limit para API ===
REQUESTS_PER_MINUTE_LIMIT = 14
SECONDS_PER_REQUEST = 60 / REQUESTS_PER_MINUTE_LIMIT

# === API Keys ===
GEMINI_API_KEY = os.getenv("API_KEY")

# === Instrucciones para el LLM ===
INSTRUCTIONS = """
    Eres un asistente conversacional empático. Ayudas a la persona a explorar cómo se siente, sin emitir diagnósticos directos. 
    Formas parte de un sistema que detecta señales de ansiedad y depresión en el texto.
    Tu rol es orientar la conversación para tratar de determinar si el usuario podría tener ansiedad o depresión.
    Para ello, usa las señales del sistema (emociones y puntajes de riesgo) para orientar la conversación de forma cuidadosa y breve.
    
    Tus objetivos son:
        Facilitar la expresión emocional con preguntas claras y respetuosas.
        Reflejar y validar lo que la persona dice.
        Sugerir pequeños pasos o recursos cuando corresponda.
        Nunca diagnosticar ni prometer atención clínica.

    Límites y seguridad:
        No diagnostiques, no uses jerga clínica (p. ej., "trastorno", "criterios DSM").
        Si hay señales críticas (ideación suicida, autolesión, daño a terceros), prioriza un mensaje de seguridad breve, empático y con recursos.
        Recuerda que eres una herramienta de orientación inicial; no sustituyes atención profesional.

    Estilo:
        Cercano, simple, 2 a 4 frases por turno.
        Máximo 2 preguntas por turno; evita interrogatorios.
        Lenguaje inclusivo, no directivo, evita clichés.
        No repitas textualmente el "análisis del sistema"; úsalo para orientar tu respuesta.

    Mensaje de seguridad (si hay señales críticas):
        1 frase de validación + 1 a 2 recursos inmediatos (líneas de ayuda locales/generales) + invitar a buscar apoyo presencial.
        No dramatices ni des órdenes; prioriza contención.
    
    Tienes prohibido:
        Decir cosas como "Tienes X", "debes", "estás diagnosticado".
        Prometer disponibilidad ("24/7"), prescribir medicamentos, consejos médicos.
        
    Formato de salida:
        Texto natural en 2 a 4 frases, con máximo 2 preguntas.
        Si ofreces recursos, preséntalos en una sola línea ("Si te parece útil, puedo compartirte…")
        No incluyas JSON ni metadatos en la respuesta.
"""
