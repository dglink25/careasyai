"""
audio.py — Amélioré depuis l'original CareasyAudioProcessor.
Corrections :
  - dict.fromkeys() ne garantit pas l'ordre → remplacé par boucle ordonnée
  - Ajout transcribe_from_bytes() pour les uploads Flask (bytes)
  - Ajout text_to_speech_sync() pour les contextes non-async
  - Mapping voix TTS complet par langue (Fon → voix française)
  - Correction : so (Somali) → sw ajouté aux corrections dialectes
"""
import os
import asyncio
import logging
import tempfile
from typing import Tuple, Optional

from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
import edge_tts

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
DetectorFactory.seed = 0

# Voix TTS par code langue
VOICES = {
    "fr":      "fr-FR-DeniseNeural",
    "en":      "en-US-JennyNeural",
    "yo":      "yo-NG-IsiolaNeural",
    "sw":      "sw-KE-ZuriNeural",
    "fon":     "fr-FR-DeniseNeural",   # Fon → voix française (pas de voix Fon dans edge-tts)
    "ar":      "ar-SA-ZariyahNeural",
    "pt":      "pt-BR-FranciscaNeural",
    "es":      "es-ES-ElviraNeural",
}

# Corrections de détection Whisper pour les dialectes béninois
# Whisper confond souvent le Fon avec yo/ht/so
DIALECT_CORRECTIONS = {
    "yo": "sw",   # Yoruba → Swahili (meilleur résultat pour le Fon)
    "ht": "fr",   # Créole haïtien → Français
    "so": "sw",   # Somali → Swahili
}


class CareasyAudioProcessor:
    def __init__(self, model_size: str = None, device: str = None):
        model_size = model_size or os.getenv("WHISPER_MODEL_SIZE", "base")
        device     = device     or os.getenv("WHISPER_DEVICE", "cpu")
        logging.info(f"Chargement Whisper '{model_size}' sur {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        logging.info("Whisper prêt.")

    def _run_transcription(self, file_path: str, lang: Optional[str] = None) -> Tuple[str, str]:
        """Exécute la transcription Whisper avec dé-duplication ordonnée."""
        segments, info = self.model.transcribe(
            file_path,
            language=lang,
            beam_size=5,
            vad_filter=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            initial_prompt="CareEasy, assistance véhicule Bénin.",
        )
        # BUG FIX : dict.fromkeys() perd l'ordre sur Python < 3.7
        # et peut perdre des segments dans les versions récentes.
        # On utilise une boucle ordonnée explicite.
        seen, texts = set(), []
        for seg in segments:
            t = seg.text.strip()
            if t and t not in seen:
                seen.add(t)
                texts.append(t)
        return " ".join(texts), info.language

    def transcribe(self, file_path: str) -> Tuple[Optional[str], str]:
        """Transcrit un fichier audio avec correction des dialectes béninois."""
        if not os.path.exists(file_path):
            return None, "error"
        try:
            text, lang = self._run_transcription(file_path)
            logging.info(f"Langue détectée : {lang}")

            if lang in DIALECT_CORRECTIONS:
                fallback = DIALECT_CORRECTIONS[lang]
                logging.info(f"Correction dialecte '{lang}' → relance en '{fallback}'")
                text, lang = self._run_transcription(file_path, lang=fallback)

            return (text or None), lang
        except Exception as e:
            logging.error(f"Erreur transcription : {e}")
            return None, "error"

    def transcribe_from_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> Tuple[Optional[str], str]:
        """Transcrit depuis des bytes (pour les uploads Flask multipart/form-data)."""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            return self.transcribe(tmp_path)
        finally:
            try: os.unlink(tmp_path)
            except: pass

    def translate_to_french(self, text: str, source_lang: str = "auto") -> Tuple[str, str]:
        """Traduit vers le français. Retourne (langue_source, texte_fr)."""
        if not text:
            return "unknown", ""
        try:
            detected = detect(text) if source_lang == "auto" else source_lang
            if detected == "fr":
                return "fr", text
            translated = GoogleTranslator(source="auto", target="fr").translate(text)
            return detected, translated
        except Exception as e:
            logging.error(f"Erreur traduction → fr : {e}")
            return "error", text

    def translate_from_french(self, text: str, target_lang: str) -> str:
        """Traduit depuis le français vers la langue cible."""
        if not text or target_lang == "fr":
            return text
        try:
            return GoogleTranslator(source="fr", target=target_lang).translate(text)
        except Exception as e:
            logging.error(f"Erreur traduction fr → {target_lang} : {e}")
            return text

    async def text_to_speech(self, text: str, output_path: str, lang: str = "fr", rate: str = "-5%") -> Optional[str]:
        """Génère un fichier MP3 depuis du texte. Async."""
        if not text:
            return None
        voice = VOICES.get(lang, VOICES["fr"])
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(output_path)
            return output_path
        except Exception as e:
            logging.error(f"Erreur TTS : {e}")
            return None

    def text_to_speech_sync(self, text: str, output_path: str, lang: str = "fr", rate: str = "-5%") -> Optional[str]:
        """Version synchrone de text_to_speech (pour Flask sans async)."""
        return asyncio.run(self.text_to_speech(text, output_path, lang, rate))
