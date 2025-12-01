"""
Módulo para evaluación de riesgo (ansiedad/depresión).
Extrae características lingüísticas y emocionales del texto usando un enfoque híbrido
(reglas + modelo ML) para predecir riesgo de trastornos.
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
    LABEL_MAP,
    IDX_TO_LABEL,
    limpiar_texto,
    PRONOMBRES_1RA,
    NEGACIONES,
    ABSOLUTISTAS,
    PREPOSICIONES,
    CONJUNCIONES,
    DETERMINANTES
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
        Extrae características lingüísticas y emocionales del texto.
        Implementa el mismo pipeline de extracción del notebook de entrenamiento.
        
        Args:
            text: Texto original a analizar
            
        Returns:
            Diccionario con todas las características extraídas
        """
        # Guardar texto original para conteo de signos
        raw_text = text
        
        # Limpiar texto para análisis
        cleaned_text = limpiar_texto(text)
        tokens = cleaned_text.split()
        num_tokens = len(tokens) if len(tokens) > 0 else 1
        
        # Obtener vector de emociones
        emo_vec = self.emotion_analyzer.get_emotion_vector(text)
        feats: Dict[str, float] = {}
        
        # === Características emocionales ===
        neg_polarity_sum = 0.0
        for i, emo in enumerate(EMOTION_NAMES):
            val = float(emo_vec[i])
            feats[f"{emo}_prob"] = val
            # Sumar emociones negativas para polaridad
            if emo in ["disgusto", "ira", "tristeza", "miedo"]:
                neg_polarity_sum += val
        
        feats["negative_polarity"] = neg_polarity_sum
        
        # === Características lingüísticas (ratios) ===
        count_pron_1st = sum(1 for t in tokens if t in PRONOMBRES_1RA)
        count_neg = sum(1 for t in tokens if t in NEGACIONES)
        count_abs = sum(1 for t in tokens if t in ABSOLUTISTAS)
        count_prep = sum(1 for t in tokens if t in PREPOSICIONES)
        count_conj = sum(1 for t in tokens if t in CONJUNCIONES)
        count_det = sum(1 for t in tokens if t in DETERMINANTES)
        count_adv_mente = sum(1 for t in tokens if t.endswith("mente"))
        
        feats["pronoun_1st_ratio"] = count_pron_1st / num_tokens
        feats["negation_ratio"] = count_neg / num_tokens
        feats["absolutist_ratio"] = count_abs / num_tokens
        feats["preposition_ratio"] = count_prep / num_tokens
        feats["conjunction_ratio"] = count_conj / num_tokens
        feats["determinant_ratio"] = count_det / num_tokens
        feats["adverb_mente_ratio"] = count_adv_mente / num_tokens
        feats["avg_word_len"] = sum(len(t) for t in tokens) / num_tokens
        feats["word_count"] = num_tokens
        
        # === Reglas híbridas para depresión y ansiedad ===
        # Score de depresión basado en tristeza + indicadores lingüísticos
        score_dep = float(emo_vec[EMOTION_NAMES.index("tristeza")])
        feats["rule_dep_score_norm"] = (
            score_dep + 
            feats["negation_ratio"] + 
            feats["absolutist_ratio"] + 
            feats["pronoun_1st_ratio"]
        ) / 4.0
        
        # Score de ansiedad basado en miedo + signos de puntuación
        score_ans = float(emo_vec[EMOTION_NAMES.index("miedo")])
        count_excla = raw_text.count("!")
        count_question = raw_text.count("?")
        
        if count_excla > 2:
            score_ans += 0.1
        if count_question > 2:
            score_ans += 0.1
        score_ans = min(score_ans, 1.0)
        
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
            # Orden por defecto basado en el notebook
            ordered_cols = [
                "disgusto_prob", "sorpresa_prob", "ira_prob", "tristeza_prob",
                "felicidad_prob", "miedo_prob", "negative_polarity",
                "pronoun_1st_ratio", "negation_ratio", "absolutist_ratio",
                "preposition_ratio", "conjunction_ratio", "determinant_ratio",
                "adverb_mente_ratio", "avg_word_len", "word_count",
                "rule_dep_score_norm", "rule_ans_score_norm",
                "exclam_count", "question_count"
            ]
        
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
            Diccionario con emociones, características, probabilidades y predicción
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
