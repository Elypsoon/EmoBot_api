"""
Módulo para evaluación de riesgo (ansiedad/depresión).
Extrae características del texto y usa un modelo evaluador para predecir riesgo.
"""

import os
import numpy as np
import joblib
from typing import Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

from config import (
    EVALUATOR_MODEL_PKL,
    SCALER_PKL,
    EMOTION_NAMES,
    THRESHOLDS,
    LABEL_MAP,
    IDX_TO_LABEL
)
from emotion_analyzer import EmotionAnalyzer


class RiskEvaluator:
    """Clase para evaluar riesgo de ansiedad/depresión a partir de texto."""
    
    def __init__(self, emotion_analyzer: EmotionAnalyzer):
        """
        Inicializa el evaluador de riesgo.
        
        Args:
            emotion_analyzer: Instancia de EmotionAnalyzer para extraer emociones
        """
        self.emotion_analyzer = emotion_analyzer
        
        # Cargar modelo evaluador
        print("[INFO] Cargando evaluador (LogReg) y scaler...")
        self.clf = joblib.load(EVALUATOR_MODEL_PKL)
        
        # Cargar scaler si existe
        if os.path.exists(SCALER_PKL):
            self.scaler: Optional[StandardScaler] = joblib.load(SCALER_PKL)
            print("[OK ] Scaler cargado.")
        else:
            self.scaler = None
            print("[WARN] No se encontró scaler.pkl. Se inferirá SIN escalar.")
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extrae características del texto para el evaluador.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con todas las características extraídas
        """
        # Obtener vector de emociones
        emo_vec = self.emotion_analyzer.get_emotion_vector(text)
        feats: Dict[str, float] = {}
        
        # Probabilidades y banderas por emoción
        for i, emo in enumerate(EMOTION_NAMES):
            val = float(emo_vec[i])
            feats[f"{emo}_prob"] = val
            thr = THRESHOLDS.get(emo, 0.7)
            feats[f"{emo}_flag"] = int(val >= thr)
        
        # Reglas simples dep/ans
        score_dep = float(emo_vec[EMOTION_NAMES.index("tristeza")])
        score_ans = float(emo_vec[EMOTION_NAMES.index("miedo")])
        count_excla = text.count("!")
        count_question = text.count("?")
        
        if count_excla > 2:
            score_ans += 0.1
        if count_question > 2:
            score_ans += 0.1
        score_ans = min(score_ans, 1.0)
        
        feats["rule_dep_score_norm"] = min(score_dep, 1.0)
        feats["rule_ans_score_norm"] = score_ans
        feats["exclam_count"] = count_excla
        feats["question_count"] = count_question
        
        return feats
    
    def vectorize_features(self, feats: Dict[str, float]) -> np.ndarray:
        """
        Alinea el vector de features al orden usado en entrenamiento.
        
        Args:
            feats: Diccionario de características
            
        Returns:
            Array numpy con características ordenadas y escaladas
        """
        if hasattr(self.clf, "feature_names_in_"):
            ordered_cols = list(self.clf.feature_names_in_)
        else:
            ordered_cols = sorted(feats.keys())
        
        row = [feats.get(col, 0.0) for col in ordered_cols]
        X = np.array(row, dtype=np.float32).reshape(1, -1)
        
        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except Exception as e:
                print(f"[WARN] Error al escalar features: {e}. Continuando sin escalar.")
        
        return X
    
    def predict_risk(self, text: str) -> Dict[str, Any]:
        """
        Predice el riesgo de ansiedad/depresión a partir de un texto.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con emociones, probabilidades por clase y predicción
        """
        # Extraer características
        feats = self.extract_features(text)
        X = self.vectorize_features(feats)
        
        # Predecir
        proba = self.clf.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = IDX_TO_LABEL[pred_idx]
        
        # Ordenar emociones para display
        emotions = {emo: float(feats[f"{emo}_prob"]) for emo in EMOTION_NAMES}
        
        return {
            "emotions": emotions,
            "features": feats,
            "class_proba": {
                "none": float(proba[LABEL_MAP["none"]]),
                "anxiety": float(proba[LABEL_MAP["anxiety"]]),
                "depression": float(proba[LABEL_MAP["depression"]]),
            },
            "pred_label": pred_label,
        }
