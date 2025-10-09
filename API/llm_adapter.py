"""
Módulo para integración con LLM (Gemini).
Maneja la comunicación con el modelo de lenguaje para generar respuestas empáticas.
"""

import time
import json
from typing import Dict, Any
from google import genai
from google.genai import types

from config import INSTRUCTIONS, SECONDS_PER_REQUEST


class GeminiAdapter:
    """Adaptador para comunicación con Gemini LLM."""
    
    def __init__(self, api_key: str, instructions: str = INSTRUCTIONS):
        """
        Inicializa el adaptador de Gemini.
        
        Args:
            api_key: Clave de API de Gemini
            instructions: Instrucciones del sistema para el LLM
            
        Raises:
            RuntimeError: Si no se puede inicializar el SDK de Gemini
        """
        if genai is None or types is None:
            raise RuntimeError(
                "No se encontró el SDK de Gemini esperado (genai). "
                "Instala/usa la misma lib que tu proyecto ya utiliza o adapta este adapter."
            )
        
        self.client = genai.Client(api_key=api_key)
        self.chat = self.client.chats.create(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=instructions,
                temperature=1.0
            )
        )
    
    def respond(self, user_text: str, analysis: Dict[str, Any]) -> str:
        """
        Genera una respuesta empática basada en el texto del usuario y el análisis del sistema.
        
        Args:
            user_text: Mensaje del usuario
            analysis: Diccionario con el análisis de riesgo y emociones
            
        Returns:
            Respuesta generada por el LLM
        """
        # Prepara un bloque compacto para dar contexto al LLM
        meta = {
            "pred_label": analysis["pred_label"],
            "class_proba": analysis["class_proba"],
            "emotions_top3": sorted(
                analysis["emotions"].items(), key=lambda x: x[1], reverse=True
            )[:3],
            "hints": [
                "máximo 2 preguntas por turno",
                "no diagnosticar, no etiquetar, solo guiar",
                "ofrecer recursos si ansiedad/depresión > 0.65 durante un periodo prolongado (más de 8 mensajes)",
            ],
        }
        
        prompt = (
            "Usuario:\n"
            f"{user_text}\n\n"
            "Análisis del sistema (no lo repitas literal, úsalo para guiar tu respuesta):\n"
            f"{json.dumps(meta, ensure_ascii=False)}\n\n"
            "Responde de forma empática y breve. Si corresponde, haz 1 a 2 preguntas útiles."
        )
        
        resp = self.chat.send_message(prompt)
        # Respetar rate-limit básico
        time.sleep(SECONDS_PER_REQUEST)
        return (resp.text or "").strip()
