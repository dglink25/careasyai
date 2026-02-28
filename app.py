import os
import uuid
import logging
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# LangChain gratuit (ChromaDB + HuggingFace)
from langchain_text_splitters import CharacterTextSplitter

import googleapiclient.discovery
from youtube_transcript_api import (
    YouTubeTranscriptApi as YTA,
    NoTranscriptFound,
    TranscriptsDisabled,
)

from pdfloader     import PDFLoader
from utils         import format_transcript, get_vehicle_details, get_procedure
from audio         import CareasyAudioProcessor
from agent         import get_agent
from laravel_bridge import (
    get_nearby_services,
    resolve_location,
    format_services_text,
    save_ai_message,
    get_conversation_history,
    log_ai_interaction,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OLLAMA_URL      = os.getenv("OLLAMA_URL",    "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",  "qwen2.5:3b")
CHROMA_DIR      = os.getenv("CHROMA_DIR",    "./chroma_db")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
AUDIO_TMP       = Path(tempfile.mkdtemp(prefix="careasy_audio_"))

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# Singletons
audio_processor = CareasyAudioProcessor()


@app.errorhandler(404)
def not_found(e):
    return jsonify({"message": "Not Found"}), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({"message": "Fichier trop volumineux (max 50 MB)"}), 413

@app.route("/")
def index():
    return jsonify({
        "message": "CareEasy AI — Vehicle Maintenance API (gratuit)",
        "llm":     OLLAMA_MODEL,
        "db":      "ChromaDB (local)",
    }), 200

@app.route("/api/v1/status", methods=["GET"])
def status():
    """Vérification santé — Ollama + ChromaDB."""
    import requests as req
    ollama_ok   = False
    ollama_models = []
    try:
        r = req.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if r.status_code == 200:
            ollama_ok     = True
            ollama_models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass

    # Collections ChromaDB
    try:
        agent = get_agent()
        collections = agent.get_collections()
    except Exception:
        collections = []

    return jsonify({
        "ollama":      {"ok": ollama_ok, "url": OLLAMA_URL, "models": ollama_models},
        "chromadb":    {"collections": len(collections), "names": collections},
        "whisper":     "ready",
        "tts":         "ready",
    }), 200



@app.route("/api/v1/chat", methods=["POST"])
def chat():
    """
    Point d'entrée principal — texte / audio / photo.
    100% gratuit, aucune API payante.

    Body JSON ou Form-data :
      message          : texte
      file             : audio (.wav/.mp3) ou image (.jpg/.png/.webp)
      lang             : fr | en | fon (défaut: fr)
      vehicle_make     : Dayang, Toyota...
      vehicle_model    : DY110-3, Corolla...
      collection_name  : nom collection ChromaDB (optionnel)
      lat, lng         : GPS pour services proches
      location_text    : "Cotonou, Akpakpa" (alternative au GPS)
      conversation_id  : ID Laravel pour historique
      return_audio     : true → retourne MP3
    """
    try:
        body = request.get_json(silent=True) or {}

        def field(name, default=""):
            v = body.get(name) or request.form.get(name) or default
            return v.strip() if isinstance(v, str) else v

        message          = field("message")
        lang             = field("lang", "fr")
        vehicle_make     = field("vehicle_make")
        vehicle_model    = field("vehicle_model")
        collection_name  = field("collection_name")
        lat              = float(field("lat") or 0)
        lng              = float(field("lng") or 0)
        location_text    = field("location_text")
        conversation_id  = int(field("conversation_id") or 0) or None
        return_audio     = str(field("return_audio", "false")).lower() == "true"

        history_raw = body.get("history") or request.form.get("history")
        history     = (history_raw if isinstance(history_raw, list)
                       else __import__("json").loads(history_raw or "[]"))

        # ── Fichier joint ─────────────────────────────────────────────────────
        image_bytes = None
        image_mime  = "image/jpeg"
        lang_detected = lang
        file = request.files.get("file")

        if file:
            filename = file.filename or ""
            ext      = Path(filename).suffix.lower()
            content  = file.content_type or ""

            if ext in (".jpg", ".jpeg", ".png", ".webp") or "image" in content:
                image_bytes = file.read()
                image_mime  = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                                ".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")
                log.info(f"Image reçue ({len(image_bytes)//1024} KB)")

            elif "audio" in content or ext in (".wav", ".mp3", ".ogg", ".m4a", ".webm"):
                audio_bytes = file.read()
                transcribed, detected = audio_processor.transcribe_from_bytes(audio_bytes, ext or ".wav")
                if not transcribed:
                    return jsonify({"message": "Impossible de transcrire l'audio."}), 422
                lang_detected = detected
                if not message:
                    message = transcribed

        # ── Traduction → français ─────────────────────────────────────────────
        message_fr = message or ""
        if lang_detected != "fr" and message:
            _, message_fr = audio_processor.translate_to_french(message, source_lang=lang_detected)

        # ── Géolocalisation ───────────────────────────────────────────────────
        user_location = None
        if lat and lng:
            user_location = {"lat": lat, "lng": lng}
        elif location_text:
            resolved = resolve_location(location_text)
            if resolved:
                user_location = {
                    "lat":     float(resolved.get("latitude",  0)),
                    "lng":     float(resolved.get("longitude", 0)),
                    "address": f"{resolved.get('arrondissement','')}, {resolved.get('commune','')}".strip(", "),
                }

        # ── Historique conversation ───────────────────────────────────────────
        if conversation_id and not history:
            history = get_conversation_history(conversation_id, limit=6)

        # ── Appel agent ───────────────────────────────────────────────────────
        agent = get_agent()

        if image_bytes and not message_fr:
            analysis_fr, urgency = agent.analyze_photo(
                image_bytes, image_mime, vehicle_make, vehicle_model, message_fr
            )
            result = {
                "answer_fr":  analysis_fr,
                "intent":     "diagnostic",
                "urgency":    urgency,
                "vehicle":    {"make": vehicle_make, "model": vehicle_model, "collection_name": collection_name},
                "services_proches": [],
                "sources":    [],
            }
        else:
            result = agent.chat(
                message=message,
                message_fr=message_fr,
                image_bytes=image_bytes,
                image_mime=image_mime,
                collection_name=collection_name or None,
                vehicle_make=vehicle_make or None,
                vehicle_model=vehicle_model or None,
                user_location=user_location,
                user_lang=lang_detected,
                history=history,
            )

        answer_fr = result["answer_fr"]
        urgency   = result.get("urgency", "unknown")
        intent    = result.get("intent",  "info_generale")

        answer_final = answer_fr
        if lang_detected not in ("fr",) and answer_fr:
            answer_final = audio_processor.translate_from_french(answer_fr, target_lang=lang_detected)

        if return_audio and answer_final:
            audio_path = str(AUDIO_TMP / f"resp_{uuid.uuid4().hex}.mp3")
            result_path = audio_processor.text_to_speech_sync(answer_final, audio_path, lang_detected)
            if result_path and Path(result_path).exists():
                return send_file(result_path, mimetype="audio/mpeg",
                                 as_attachment=True, download_name="careasy_response.mp3")

        if conversation_id and answer_fr:
            ai_msg_id = save_ai_message(
                conversation_id=conversation_id,
                content=answer_final,
                ai_metadata={
                    "intent": intent, "urgency": urgency, "lang": lang_detected,
                    "vehicle_make": result.get("vehicle", {}).get("make", ""),
                    "model":  OLLAMA_MODEL,
                },
            )
            log_ai_interaction(
                message_id=ai_msg_id, ai_input=message_fr or "",
                ai_output=answer_fr, intent=intent, language=lang_detected,
                model=OLLAMA_MODEL,
            )

        return jsonify({
            "answer":           answer_final,
            "answer_fr":        answer_fr,
            "intent":           intent,
            "urgency":          urgency,
            "lang_detected":    lang_detected,
            "vehicle":          result.get("vehicle", {}),
            "services_proches": result.get("services_proches", []),
            "sources":          result.get("sources", []),
        }), 200

    except Exception as e:
        log.error(f"Erreur /chat : {e}", exc_info=True)
        return jsonify({"message": f"Erreur : {str(e)}"}), 500


@app.route("/api/v1/embed", methods=["POST", "DELETE"])
def embed():
    """
    POST  : Indexe un PDF dans ChromaDB (local, gratuit).
    DELETE: Supprime une collection ChromaDB.
    """
    if request.method == "POST":
        try:
            file = request.files.get("file")
            if not file:
                return jsonify({"message": "Fichier 'file' manquant"}), 400

            pdf       = file.read()
            loader    = PDFLoader(pdf)
            documents = loader.load()
            doc_id    = loader.docId

            # Vérifier si déjà indexé
            agent = get_agent()
            existing = agent.get_collections()
            if doc_id in existing:
                return jsonify({"documentId": doc_id, "already_exists": True}), 200

            # Extraction détails véhicule
            first_pages = "".join(d.page_content for d in documents[:5])
            vehicle_details = get_vehicle_details(first_pages)

            # Découpage en chunks
            splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
            chunks   = splitter.split_documents(documents)
            texts    = [c.page_content for c in chunks]
            metas    = [{
                **c.metadata,
                "vehicle_make":  vehicle_details["make"],
                "vehicle_model": vehicle_details["model"],
                "vehicle_year":  vehicle_details["year"],
            } for c in chunks]

            # Indexation ChromaDB
            indexed = agent.index_document(texts, doc_id, metas)

            return jsonify({
                "documentId":     doc_id,
                "vehicleDetails": vehicle_details,
                "chunks":         indexed,
                "storage":        "ChromaDB (local)",
            }), 200

        except Exception as e:
            log.error(f"Erreur POST /embed : {e}", exc_info=True)
            return jsonify({"message": f"Erreur : {str(e)}"}), 500

    if request.method == "DELETE":
        try:
            body   = request.get_json(silent=True) or {}
            doc_id = body.get("id", "").strip()
            if not doc_id:
                return jsonify({"message": "Fournir 'id'"}), 400

            agent = get_agent()
            if doc_id not in agent.get_collections():
                return jsonify({"message": "Collection non trouvée"}), 404

            agent.chroma.delete_collection(doc_id)
            return jsonify({"message": doc_id, "deleted": True}), 200

        except Exception as e:
            log.error(f"Erreur DELETE /embed : {e}", exc_info=True)
            return jsonify({"message": str(e)}), 500

@app.route("/api/v1/qa", methods=["POST"])
def qa():
    """Q&A sur un document ChromaDB (local, gratuit)."""
    try:
        body  = request.get_json(silent=True) or {}
        doc_id = body.get("id",    "").strip()
        query  = body.get("query", "").strip()
        lang   = body.get("lang",  "fr").strip()

        if not doc_id: return jsonify({"message": "Fournir 'id'"}), 400
        if not query:  return jsonify({"message": "Fournir 'query'"}), 400

        agent = get_agent()
        if doc_id not in agent.get_collections():
            return jsonify({"message": "Document non trouvé"}), 404

        # Traduction → français
        query_fr = query
        if lang != "fr":
            _, query_fr = audio_processor.translate_to_french(query, source_lang=lang)

        # RAG + réponse Ollama
        context, _ = agent._rag(query_fr, doc_id)
        from utils import answer_question_with_context
        answer_fr = answer_question_with_context(query_fr, [context] if context else [])

        # Traduction réponse
        answer = answer_fr
        if lang != "fr":
            answer = audio_processor.translate_from_french(answer_fr, target_lang=lang)

        return jsonify({"answer": answer, "answer_fr": answer_fr}), 200

    except Exception as e:
        log.error(f"Erreur /qa : {e}", exc_info=True)
        return jsonify({"message": str(e)}), 500

@app.route("/api/v1/services/nearby", methods=["GET"])
def nearby_services():
    try:
        lat    = float(request.args.get("lat", 0))
        lng    = float(request.args.get("lng", 0))
        radius = float(request.args.get("radius", 10))
        domaine = request.args.get("domaine", "")
        if not lat or not lng:
            return jsonify({"message": "Fournir lat et lng"}), 400
        services = get_nearby_services(lat, lng, radius_km=radius, domaine=domaine or None)
        return jsonify({"data": services, "count": len(services)}), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route("/api/v1/audio/tts", methods=["POST"])
def tts():
    try:
        body = request.get_json(silent=True) or {}
        text = body.get("text", "").strip()
        lang = body.get("lang", "fr").strip()
        if not text:
            return jsonify({"message": "Fournir 'text'"}), 400
        output = str(AUDIO_TMP / f"tts_{uuid.uuid4().hex}.mp3")
        result = audio_processor.text_to_speech_sync(text, output, lang)
        if result and Path(result).exists():
            return send_file(result, mimetype="audio/mpeg",
                             as_attachment=True, download_name="tts.mp3")
        return jsonify({"message": "Erreur TTS"}), 500
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route("/api/v1/audio/transcribe", methods=["POST"])
def transcribe():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"message": "Fichier audio manquant"}), 400
        translate = request.form.get("translate_to_fr", "false").lower() == "true"
        suffix    = Path(file.filename or "audio.wav").suffix or ".wav"
        text, lang = audio_processor.transcribe_from_bytes(file.read(), suffix)
        if not text:
            return jsonify({"message": "Transcription impossible"}), 422
        result = {"text": text, "lang_detected": lang}
        if translate and lang != "fr":
            _, result["text_fr"] = audio_processor.translate_to_french(text, lang)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500



@app.route("/api/v1/video", methods=["GET"])
def get_video():
    try:
        prompt = request.args.get("prompt", "").strip()
        if not prompt:
            return jsonify({"message": "Fournir 'prompt'"}), 400
        yt = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        resp = yt.search().list(part="snippet", maxResults=3, q=prompt).execute()
        return jsonify(resp.get("items", [])), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route("/api/v1/video-procedure/<video_id>", methods=["POST"])
def get_procedure_from_video(video_id):
    try:
        body  = request.get_json(silent=True) or {}
        query = body.get("query", "").strip()
        if not query:
            return jsonify({"message": "Fournir 'query'"}), 400
        try:
            data = YTA.get_transcript(video_id, languages=["fr", "en"])
        except (NoTranscriptFound, TranscriptsDisabled):
            return jsonify({"message": "Transcript non trouvé"}), 404
        transcript = format_transcript(data)
        procedure  = get_procedure(transcript, query)
        return jsonify({"procedure": procedure}), 200
    except ValueError as e:
        return jsonify({"message": str(e)}), 422
    except Exception as e:
        return jsonify({"message": str(e)}), 500

if __name__ == "__main__":
    port  = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_ENV", "production") == "development"
    log.info(f"CareEasy AI (gratuit) démarré sur le port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)