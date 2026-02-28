"""
utils.py — Version gratuite (sans OpenAI)
Utilise Ollama à la place de GPT pour get_vehicle_details et get_procedure.
"""
import json
import re
import os
import logging
import requests

log = logging.getLogger(__name__)

OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


def _ollama(prompt: str, system: str = "", max_tokens: int = 500) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": OLLAMA_MODEL, "messages": messages,
                  "stream": False, "options": {"num_predict": max_tokens}},
            timeout=60,
        )
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        log.error(f"Erreur Ollama : {e}")
        return ""


def format_transcript(transcript: list) -> str:
    """Formate la transcription YouTube."""
    items = [
        {"start": str(item["start"]),
         "end":   str(item["start"] + item["duration"]),
         "text":  item["text"]}
        for item in transcript
    ]
    return json.dumps(items)


def get_vehicle_details(owners_manual: str) -> dict:
    """Extrait marque/modèle/année depuis un manuel — via Ollama (gratuit)."""
    prompt = (
        f"From this vehicle manual extract the make, model and year. "
        f"Reply ONLY with JSON: {{\"make\":\"...\",\"model\":\"...\",\"year\":\"...\"}}\n\n"
        f"Manual (first 2000 chars): {owners_manual[:2000]}"
    )
    raw = _ollama(prompt, system="You are a vehicle manual parser. Reply with JSON only.")
    try:
        # Extraire le JSON même si entouré de texte
        match = re.search(r'\{[^}]+\}', raw)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {"make": "Unknown", "model": "Unknown", "year": "Unknown"}


def get_procedure(transcript: str, query: str) -> list:
    """Extrait une procédure depuis une transcription YouTube — via Ollama."""
    prompt = (
        f"From this YouTube transcript, extract exactly 10 procedure steps for: '{query}'\n"
        f"Reply ONLY with a JSON array: "
        f"[{{\"start\":\"0\",\"end\":\"30\",\"content\":\"Step description\"}},...]\n\n"
        f"Transcript: {transcript[:4000]}"
    )
    raw = _ollama(prompt, system="You are a procedure extractor. Reply with JSON array only.")
    try:
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            procedure = json.loads(match.group())
            if procedure:
                return procedure
    except Exception:
        pass
    raise ValueError("Impossible d'extraire la procédure depuis la vidéo")


def answer_question_with_context(question: str, context_chunks: list) -> str:
    """Répond à une question avec du contexte — via Ollama."""
    context = "\n\n".join(context_chunks[:4]) if context_chunks else ""
    prompt  = (
        f"Contexte :\n{context}\n\nQuestion : {question}\n\n"
        "Réponds de façon précise et pratique en français."
    ) if context else question
    result = _ollama(
        prompt,
        system="Tu es un expert en mécanique automobile spécialisé Afrique de l'Ouest.",
        max_tokens=800,
    )
    return result or "Je n'ai pas pu générer une réponse. Veuillez réessayer."
