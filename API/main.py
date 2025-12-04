"""
API REST de EmoBot.
Sistema de chat conversacional con análisis de emociones y evaluación de riesgo.
"""

import time
import uuid
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import GEMINI_API_KEY
from emotion_analyzer import EmotionAnalyzer
from risk_evaluator import RiskEvaluator
from llm_adapter import GeminiAdapter


# === Modelos Pydantic para request/response ===
class ChatRequest(BaseModel):
    """Modelo para la solicitud de chat."""
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Modelo para la respuesta del chat."""
    response: str
    session_id: str


# === Variables globales para los modelos (se inicializan al arrancar) ===
emotion_analyzer: Optional[EmotionAnalyzer] = None
risk_evaluator: Optional[RiskEvaluator] = None
llm: Optional[GeminiAdapter] = None

# Diccionario para almacenar sesiones en memoria
sessions: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Contexto de vida de la aplicación para inicializar modelos."""
    global emotion_analyzer, risk_evaluator, llm
    
    print("[INICIO] Cargando modelos...")
    
    # Inicializar analizador de emociones
    emotion_analyzer = EmotionAnalyzer()
    
    # Inicializar evaluador de riesgo
    risk_evaluator = RiskEvaluator(emotion_analyzer)
    
    # Inicializar LLM si hay API key
    if GEMINI_API_KEY:
        try:
            llm = GeminiAdapter(GEMINI_API_KEY)
            print("[OK ] LLM (Gemini) inicializado.")
        except Exception as e:
            print(f"[WARN] No se pudo inicializar el LLM: {e}")
            llm = None
    else:
        print("[WARN] GEMINI_API_KEY no definido. El chat funcionará sin LLM.")
        llm = None
    
    print("[OK ] API EmoBot lista para recibir solicitudes.")
    
    yield
    
    # Limpieza al cerrar (si es necesario)
    print("[FIN] Cerrando API EmoBot.")


# === Crear aplicación FastAPI ===
app = FastAPI(
    title="EmoBot API",
    description="API para el chatbot de diagnóstico de ansiedad y depresión",
    version="1.0.0",
    lifespan=lifespan
)

# === Configurar CORS para permitir conexiones desde el frontend ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Endpoint raíz para verificar que la API está funcionando."""
    return {"message": "EmoBot API está funcionando", "status": "ok"}


@app.get("/health")
async def health_check():
    """Endpoint de salud para verificar el estado de los componentes."""
    return {
        "status": "healthy",
        "emotion_analyzer": emotion_analyzer is not None,
        "risk_evaluator": risk_evaluator is not None,
        "llm_available": llm is not None
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint principal del chat.
    Recibe un mensaje del usuario y devuelve la respuesta del LLM.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío")
    
    if risk_evaluator is None:
        raise HTTPException(status_code=503, detail="El servicio no está inicializado correctamente")
    
    # Obtener o crear session_id
    session_id = request.session_id or uuid.uuid4().hex
    
    # Inicializar sesión si no existe
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "created_at": time.time()
        }
    
    user_text = request.message.strip()
    
    # 1) Análisis local (transformer + evaluador)
    analysis = risk_evaluator.predict_risk(user_text)
    
    # 2) Respuesta del LLM (si disponible)
    if llm:
        try:
            llm_text = llm.respond(user_text, analysis)
        except Exception as e:
            llm_text = "Lo siento, hubo un problema al procesar tu mensaje. ¿Podrías intentarlo de nuevo?"
            print(f"[ERROR] Error en LLM: {e}")
    else:
        llm_text = "El servicio de chat no está disponible en este momento."
    
    # 3) Guardar en historial de sesión
    sessions[session_id]["history"].append({
        "role": "user",
        "text": user_text,
        "analysis": analysis,
        "ts": time.time(),
    })
    sessions[session_id]["history"].append({
        "role": "assistant",
        "text": llm_text,
        "ts": time.time(),
    })
    
    return ChatResponse(
        response=llm_text,
        session_id=session_id
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Elimina una sesión de chat."""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Sesión eliminada correctamente"}
    raise HTTPException(status_code=404, detail="Sesión no encontrada")
