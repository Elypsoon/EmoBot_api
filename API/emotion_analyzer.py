"""
Módulo para análisis de emociones usando el modelo RoBERTa entrenado.
Maneja la carga del modelo local y la predicción de emociones desde texto.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict

from config import (
    LOCAL_MODEL_DIR,
    NUM_EMOTIONS,
    EMOTION_NAMES,
    DEVICE,
    limpiar_texto
)


class EmotionAnalyzer:
    """Clase para cargar y usar el modelo de detección de emociones."""
    
    def __init__(self):
        """Inicializa y carga el modelo emocional y el tokenizer desde local."""
        print(f"[INFO] Usando dispositivo: {DEVICE}")
        print(f"[INFO] Cargando tokenizer y modelo desde: {LOCAL_MODEL_DIR}")
        
        # Cargar tokenizer desde directorio local
        self.tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_MODEL_DIR,
            local_files_only=True,
            use_fast=True
        )
        
        # Cargar modelo completo desde directorio local (incluye pesos)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            LOCAL_MODEL_DIR,
            local_files_only=True
        )
        
        # Actualizar configuración del modelo para las emociones
        self.model.config.num_labels = NUM_EMOTIONS
        self.model.config.id2label = {i: lab for i, lab in enumerate(EMOTION_NAMES)}
        self.model.config.label2id = {lab: i for i, lab in enumerate(EMOTION_NAMES)}
        
        self.model.to(DEVICE)
        self.model.eval()
        
        print("[OK ] Modelo emocional cargado correctamente.")
    
    def get_emotion_vector(self, text: str, max_length: int = 256) -> np.ndarray:
        """
        Analiza un texto y devuelve un vector de probabilidades emocionales.
        
        Args:
            text: Texto a analizar
            max_length: Longitud máxima para el tokenizer
            
        Returns:
            Array numpy con las probabilidades de cada emoción (shape: NUM_EMOTIONS,)
        """
        # Limpiar texto antes de tokenizar
        cleaned_text = limpiar_texto(text)
        
        encoding = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
        return probs
    
    def analyze_emotions(self, text: str) -> Dict[str, float]:
        """
        Analiza un texto y devuelve un diccionario con las emociones detectadas.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con nombre de emoción como clave y probabilidad como valor
        """
        probs = self.get_emotion_vector(text)
        return {emo: float(probs[i]) for i, emo in enumerate(EMOTION_NAMES)}
