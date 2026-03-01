"""
audio.py v2 — CareEasy Audio (Voix féminine béninoise)
────────────────────────────────────────────────────────
Voix CarAI personnalisée:
  - Principale: fr-CA-SylvieNeural (voix féminine canadienne-française)
    Intonation chaleureuse et claire, proche du français béninois
  - Débit -15% pour le rythme naturel béninois
  - Accent béninois simulé via initial_prompt Whisper
  - Fon: même voix avec débit plus lent
"""

import asyncio, io, logging, os, tempfile, re
from pathlib import Path
from typing import Optional, Tuple

import edge_tts
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE     = os.getenv("WHISPER_DEVICE",     "cpu")

# ── Voix CarAI ────────────────────────────────────────────────────────────────
# fr-CA-SylvieNeural : voix féminine principale de CarAI
#   Chaude, claire, débit adapté au français béninois
# Alternatives testées et classées:
#   fr-CA-SylvieNeural   → CHOISIE (féminine, chaleureuse, claire)
#   fr-CA-AntoineNeural  → Masculine, chaleureuse (option masculine)
#   fr-BE-CharlineNeural → Belge, moins naturel
#   fr-FR-DeniseNeural   → Standard France, trop "métro"

VOICES = {
    "fr":  "fr-CA-SylvieNeural",   # CarAI voix principale
    "fon": "fr-CA-SylvieNeural",   # Fon → CarAI (pas de voix Fon native)
    "en":  "en-US-AriaNeural",     # Anglais féminin naturel
    "sw":  "sw-KE-ZuriNeural",     # Swahili Kenya (féminin)
    "yo":  "yo-NG-IsiolaNeural",   # Yoruba Nigeria
    "ar":  "ar-EG-SalmaNeural",
    "pt":  "pt-BR-ThalitaNeural",
    "es":  "es-MX-DaliaNeural",
}

# Débit par langue (-15% = 15% plus lent = rythme béninois naturel)
VOICE_RATE = {
    "fr":  "-15%",
    "fon": "-20%",
    "en":  "-8%",
    "sw":  "-10%",
    "yo":  "-10%",
}

# Corrections pour l'accent béninois (Whisper confond certains sons)
DIALECT_CORRECTIONS = {
    "ht": "fr",   # Créole haïtien → Français béninois
    "ig": "fr",   # Igbo → Français
    "ha": "fr",   # Haoussa → Français
    "yo": "fon",  # Yoruba → Fon (contexte Bénin)
}

BENIN_VOCAB = {
    "zemidjan": "zémidjan", "keke": "kéké",
    "tokpa": "Dantokpa", "fcfa": "FCFA",
    "careasy": "CareEasy",
}


class CareasyAudioProcessor:

    def __init__(self):
        log.info(f"Chargement Whisper '{WHISPER_MODEL_SIZE}' sur {WHISPER_DEVICE}...")
        self.whisper = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE,
                                   compute_type="int8")
        log.info("Whisper prêt.")

    def transcribe_from_bytes(self, audio_bytes: bytes,
                               suffix: str = ".wav") -> Tuple[str, str]:
        """
        Transcrit un message audio béninois.
        Guide Whisper avec le vocabulaire béninois pour meilleure précision.
        """
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            tmp = f.name
        try:
            segments, info = self.whisper.transcribe(
                tmp,
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 400},
                initial_prompt=(
                    "Conversation en français béninois. "
                    "Contexte: automobile, moto, garage, mécanique au Bénin. "
                    "Mots courants: zémidjan, kéké, Cotonou, Parakou, Lokossa, "
                    "FCFA, Dantokpa, garage, panne, huile, carburant."
                ),
            )
            text = " ".join(s.text.strip() for s in segments if s.text.strip())
            lang = DIALECT_CORRECTIONS.get(info.language, info.language) or "fr"
            # Corrections vocabulaire béninois
            for wrong, right in BENIN_VOCAB.items():
                text = text.replace(wrong, right)
            log.info(f"Transcription: lang={lang}, '{text[:80]}...'")
            return text, lang
        except Exception as e:
            log.error(f"Erreur transcription: {e}")
            return "", "fr"
        finally:
            Path(tmp).unlink(missing_ok=True)

    def translate_to_french(self, text: str, source_lang: str = "auto") -> Tuple[str, str]:
        if source_lang == "fr" or not text.strip(): return source_lang, text
        try:
            lang_map = {"fon": "fr", "ht": "fr", "yo": "yo", "sw": "sw"}
            src = lang_map.get(source_lang, source_lang)
            if src == "fr": return "fr", text
            translated = GoogleTranslator(source=src, target="fr").translate(text)
            return "fr", translated or text
        except Exception as e:
            log.warning(f"Traduction → FR: {e}")
            return source_lang, text

    def translate_from_french(self, text: str, target_lang: str = "fr") -> str:
        if target_lang == "fr" or not text.strip(): return text
        try:
            lang_map = {"fon": "fr", "sw": "sw", "yo": "yo"}
            tgt = lang_map.get(target_lang, target_lang)
            if tgt == "fr": return text
            return GoogleTranslator(source="fr", target=tgt).translate(text) or text
        except Exception as e:
            log.warning(f"Traduction ← FR: {e}")
            return text

    def text_to_speech_sync(self, text: str, output_path: str,
                             lang: str = "fr") -> Optional[str]:
        """
        Synthèse vocale CarAI — voix féminine fr-CA-SylvieNeural.
        Nettoie le texte: retire emojis, markdown, caractères spéciaux.
        """
        try:
            return asyncio.run(self._tts_async(text, output_path, lang))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._tts_async(text, output_path, lang))
            finally:
                loop.close()

    async def _tts_async(self, text: str, output_path: str,
                          lang: str = "fr") -> Optional[str]:
        voice = VOICES.get(lang, VOICES["fr"])
        rate  = VOICE_RATE.get(lang, "-12%")
        clean = self._clean_for_tts(text)
        if not clean.strip(): return None
        try:
            communicate = edge_tts.Communicate(clean, voice, rate=rate)
            await communicate.save(output_path)
            log.info(f"TTS: {voice} ({lang}) → {output_path} ({len(clean)} chars)")
            return output_path
        except Exception as e:
            log.error(f"TTS {voice}: {e}")
            # Fallback voix standard
            try:
                communicate = edge_tts.Communicate(clean, "fr-FR-DeniseNeural", rate="-10%")
                await communicate.save(output_path)
                return output_path
            except Exception as e2:
                log.error(f"TTS fallback: {e2}")
                return None

    def _clean_for_tts(self, text: str) -> str:
        """Nettoie pour la synthèse vocale: retire emojis, markdown, séparateurs."""
        # Retirer les emojis
        emoji_pat = re.compile(
            "[\U00002600-\U000027BF\U0001F300-\U0001FFFF\U00002700-\U000027BF]+",
            flags=re.UNICODE)
        text = emoji_pat.sub(" ", text)
        # Retirer le markdown
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*',     r'\1', text)
        text = re.sub(r'#{1,6}\s',      '',    text)
        text = re.sub(r'[━─=]{3,}',     '',    text)
        text = re.sub(r'\[|\]',         '',    text)
        # Remplacer pour l'oral
        text = text.replace("FCFA", "francs CFA")
        text = text.replace(" km ", " kilomètres ")
        text = text.replace("Tel:", "Téléphone:")
        text = text.replace("---", "")
        text = text.replace("CareEasy", "Care Easy")
        # Nettoyer les espaces
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ',     text)
        return text.strip()


_processor: Optional[CareasyAudioProcessor] = None

def get_audio_processor() -> CareasyAudioProcessor:
    global _processor
    if _processor is None:
        _processor = CareasyAudioProcessor()
    return _processor