"""
audio.py — CareEasy Audio (Bénin natif)
────────────────────────────────────────
Voix personnalisée CareEasy :
  - Français béninois : fr-CA-AntoineNeural (voix masculine chaleureuse)
    Ton plus proche de l'accent béninois que le français de France
  - Débit légèrement ralenti (-15%) pour l'accent béninois
  - Pitch neutre, style naturel et accessible
  - Fon/Swahili : sw-KE-ZuriNeural
"""

import asyncio
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import edge_tts
from deep_translator import GoogleTranslator
from langdetect import detect as langdetect_detect
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE     = os.getenv("WHISPER_DEVICE",     "cpu")

# ── Voix CareEasy personnalisées ──────────────────────────────────────────────
# fr-CA-AntoineNeural : voix masculine, accent canadien-français
#   → Proche de l'intonation béninoise, chaleureux, naturel
# fr-CA-SylvieNeural  : voix féminine alternative
# Débit -15% pour simuler le rythme de parole béninois

VOICES = {
    "fr":  "fr-CA-AntoineNeural",   # Voix CareEasy principale (béninois)
    "fon": "fr-CA-AntoineNeural",   # Fon → voix CareEasy (même voix, texte traduit)
    "en":  "en-US-GuyNeural",       # Anglais (voix masculine naturelle)
    "sw":  "sw-KE-ZuriNeural",      # Swahili (Kenya)
    "yo":  "yo-NG-IsiolaNeural",    # Yoruba (Nigeria)
    "ar":  "ar-SA-HamedNeural",
    "pt":  "pt-BR-AntonioNeural",
    "es":  "es-MX-JorgeNeural",
}

# Débit de parole par langue (ralenti pour meilleure compréhension)
VOICE_RATE = {
    "fr":  "-12%",   # Rythme béninois — légèrement plus lent
    "fon": "-18%",   # Plus lent pour le Fon
    "en":  "-8%",
    "sw":  "-10%",
    "yo":  "-10%",
}

# Corrections Whisper pour les accents béninois
# Whisper confond souvent : Fon → haïtien (ht) ou Yoruba (yo)
DIALECT_CORRECTIONS = {
    "ht": "fr",   # Créole haïtien → Français (accent béninois confondu)
    "yo": "sw",   # Yoruba détecté → traiter comme Swahili pour le Fon
    "ig": "fr",   # Igbo → Français
    "ha": "fr",   # Haoussa → Français (contexte Bénin)
}

# Vocabulaire béninois commun pour améliorer la reconnaissance Whisper
# Ces substitutions aident à corriger les transcriptions
BENIN_VOCAB_CORRECTIONS = {
    "zemidjan": "zémidjan",
    "keke": "kéké",
    "tokpa": "Dantokpa",
    "dakar": "Dakar",
    "cotonou": "Cotonou",
    "lokossa": "Lokossa",
    "parakou": "Parakou",
}


class CareasyAudioProcessor:

    def __init__(self):
        log.info(f"Chargement Whisper '{WHISPER_MODEL_SIZE}' sur {WHISPER_DEVICE}...")
        self.whisper = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE,
                                   compute_type="int8")
        log.info("Whisper prêt.")

    # ── Transcription ─────────────────────────────────────────────────────────

    def transcribe_from_bytes(self, audio_bytes: bytes,
                               suffix: str = ".wav") -> Tuple[str, str]:
        """
        Transcrit un fichier audio en texte.
        Gère les accents béninois : Fon, Français local, Yoruba.

        Returns: (texte_transcrit, langue_détectée)
        """
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            # Whisper avec beam_size=5 pour meilleure précision accents
            segments, info = self.whisper.transcribe(
                tmp_path,
                beam_size=5,
                vad_filter=True,           # Filtre silence
                vad_parameters={"min_silence_duration_ms": 500},
                initial_prompt=(           # Guide Whisper vers le contexte béninois
                    "Transcription en français béninois. "
                    "Mots clés : zémidjan, kéké, Cotonou, FCFA, Dantokpa, "
                    "mécanique, garage, voiture, moto."
                ),
            )
            text  = " ".join(s.text.strip() for s in segments if s.text.strip())
            lang  = info.language or "fr"
            lang  = DIALECT_CORRECTIONS.get(lang, lang)

            # Corrections vocabulaire béninois
            text_corr = text
            for wrong, right in BENIN_VOCAB_CORRECTIONS.items():
                text_corr = text_corr.replace(wrong, right)

            log.info(f"Transcription : lang={lang}, texte='{text_corr[:80]}...'")
            return text_corr, lang
        except Exception as e:
            log.error(f"Erreur transcription : {e}")
            return "", "fr"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ── Traduction ────────────────────────────────────────────────────────────

    def translate_to_french(self, text: str,
                             source_lang: str = "auto") -> Tuple[str, str]:
        """Traduit vers le français."""
        if source_lang == "fr" or not text.strip():
            return source_lang, text
        try:
            # Normaliser les codes de langue
            lang_map = {"sw": "sw", "yo": "yo", "fon": "fr", "ht": "fr"}
            src = lang_map.get(source_lang, source_lang)
            translated = GoogleTranslator(source=src, target="fr").translate(text)
            return "fr", translated or text
        except Exception as e:
            log.warning(f"Traduction vers FR : {e}")
            return source_lang, text

    def translate_from_french(self, text: str, target_lang: str = "fr") -> str:
        """Traduit depuis le français vers la langue cible."""
        if target_lang == "fr" or not text.strip():
            return text
        try:
            lang_map = {"fon": "fr", "sw": "sw", "yo": "yo"}
            tgt = lang_map.get(target_lang, target_lang)
            if tgt == "fr":
                return text  # Pas de traduction Fon disponible
            translated = GoogleTranslator(source="fr", target=tgt).translate(text)
            return translated or text
        except Exception as e:
            log.warning(f"Traduction depuis FR : {e}")
            return text

    # ── TTS Béninois ──────────────────────────────────────────────────────────

    def text_to_speech_sync(self, text: str, output_path: str,
                             lang: str = "fr") -> Optional[str]:
        """
        Synthèse vocale avec voix CareEasy personnalisée.
        Voix masculine fr-CA-AntoineNeural — accent proche du béninois.
        """
        try:
            return asyncio.run(self._tts_async(text, output_path, lang))
        except RuntimeError:
            # Dans un loop asyncio existant
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._tts_async(text, output_path, lang))
            finally:
                loop.close()

    async def _tts_async(self, text: str, output_path: str,
                          lang: str = "fr") -> Optional[str]:
        voice = VOICES.get(lang, VOICES["fr"])
        rate  = VOICE_RATE.get(lang, "-10%")

        # Nettoyer le texte pour la synthèse (retirer les emojis et markdown)
        text_clean = self._clean_for_tts(text)
        if not text_clean.strip():
            return None

        try:
            communicate = edge_tts.Communicate(text_clean, voice, rate=rate)
            await communicate.save(output_path)
            log.info(f"TTS généré : {voice} → {output_path}")
            return output_path
        except Exception as e:
            log.error(f"Erreur TTS {voice} : {e}")
            # Fallback sur voix standard
            try:
                communicate = edge_tts.Communicate(text_clean, "fr-FR-HenriNeural", rate="-10%")
                await communicate.save(output_path)
                return output_path
            except Exception as e2:
                log.error(f"TTS fallback : {e2}")
                return None

    def _clean_for_tts(self, text: str) -> str:
        """Nettoie le texte pour la synthèse vocale."""
        import re
        # Retirer les emojis
        emoji_pattern = re.compile(
            "[\U00002600-\U000027BF\U0001F300-\U0001F9FF"
            "\U00002700-\U000027BF\U0001FA00-\U0001FA9F]+",
            flags=re.UNICODE)
        text = emoji_pattern.sub("", text)
        # Retirer le markdown
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)   # **gras**
        text = re.sub(r'\*(.+?)\*',     r'\1', text)    # *italique*
        text = re.sub(r'#{1,6}\s',      '',    text)     # # titres
        text = re.sub(r'━+|─+|={3,}',  '',    text)     # séparateurs
        text = re.sub(r'\[|\]',         '',    text)     # crochets
        # Remplacer les symboles parlés
        text = text.replace("🔴", "urgent").replace("🟡", "attention").replace("🟢", "ok")
        text = text.replace("FCFA", "francs CFA").replace("km", "kilomètres")
        # Retirer les lignes vides multiples
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


# ── Singleton ─────────────────────────────────────────────────────────────────

_processor: Optional[CareasyAudioProcessor] = None

def get_audio_processor() -> CareasyAudioProcessor:
    global _processor
    if _processor is None:
        _processor = CareasyAudioProcessor()
    return _processor