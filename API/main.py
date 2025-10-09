"""
Aplicación principal de chat EmoBot.
Sistema de chat conversacional con análisis de emociones y evaluación de riesgo.
"""

import time
import uuid
from typing import Dict, Any, List, Optional
from config import GEMINI_API_KEY
from emotion_analyzer import EmotionAnalyzer
from risk_evaluator import RiskEvaluator
from llm_adapter import GeminiAdapter


def main():
    """Función principal del chat EmoBot."""
    print("=== Chat EmoBot (Transformer + Evaluador + LLM) ===")
    session_id = uuid.uuid4().hex
    history: List[Dict[str, Any]] = []
    
    # Inicializar componentes
    print("\n[INICIO] Cargando modelos...")
    emotion_analyzer = EmotionAnalyzer()
    risk_evaluator = RiskEvaluator(emotion_analyzer)
    
    # Inicializar LLM si hay API key
    llm: Optional[GeminiAdapter] = None
    if GEMINI_API_KEY:
        try:
            llm = GeminiAdapter(GEMINI_API_KEY)
            print("[OK ] LLM (Gemini) inicializado.")
        except Exception as e:
            print(f"[WARN] No se pudo inicializar el LLM: {e}")
    else:
        print("[WARN] GEMINI_API_KEY no definido. El chat funcionará sin LLM (solo análisis).")
    
    print("\nEscribe tu mensaje y presiona Enter.")
    print("Comandos: /quit para salir, /summary para ver resumen de la sesión.\n")
    
    while True:
        try:
            user_text = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[SALIENDO]")
            break
        
        if not user_text:
            continue
        
        if user_text.lower() in ("/quit", "/exit"):
            print("[FIN] Gracias por usar EmoBot.")
            break
        
        if user_text.lower() == "/summary":
            # Pequeño resumen de la sesión
            n_msgs = sum(1 for m in history if m["role"] == "user")
            print(f"[RESUMEN] Sesión {session_id} – mensajes del usuario: {n_msgs}")
            if history:
                last = history[-1].get("analysis")
                if last:
                    print(f"  Último riesgo: {last['class_proba']}")
            continue
        
        # 1) Análisis local (transformer + evaluador)
        analysis = risk_evaluator.predict_risk(user_text)
        
        # Mostrar resultados al usuario (compacto)
        probs = analysis["class_proba"]
        print(
            f"[ANÁLISIS] emocionesTop: "
            f"{sorted(analysis['emotions'].items(), key=lambda x:x[1], reverse=True)[:3]}"
        )
        print(
            f"[RIESGO] none={probs['none']:.2f} | anxiety={probs['anxiety']:.2f} | "
            f"depression={probs['depression']:.2f} => pred: {analysis['pred_label']}"
        )
        
        # 2) Respuesta del LLM (si disponible)
        if llm:
            try:
                llm_text = llm.respond(user_text, analysis)
            except Exception as e:
                llm_text = f"(No se pudo obtener respuesta del LLM: {e})"
        else:
            llm_text = "No hay LLM configurado."
        
        print(f"Asistente: {llm_text}\n")
        
        # 3) Guardar en historial en memoria
        history.append({
            "role": "user",
            "text": user_text,
            "analysis": analysis,
            "ts": time.time(),
        })
        history.append({
            "role": "assistant",
            "text": llm_text,
            "ts": time.time(),
        })


if __name__ == "__main__":
    main()
